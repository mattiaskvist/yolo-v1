"""Training script for YOLO v1 with Modal support."""

import argparse
from pathlib import Path

import modal
import torch
import torch.optim as optim
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from yolo import ResNetBackbone, YOLOv1
from yolo.dataset import create_voc_datasets
from yolo.loss import YOLOLoss
from yolo.training import (
    log_hyperparameters,
    print_dataset_info,
    print_model_info,
    print_tensorboard_info,
    print_training_config,
    train,
)


# ============================================================================
# Modal Setup
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

yolo_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.13")
    .entrypoint([])
    .uv_sync(uv_project_dir=PROJECT_ROOT)
    .env({"PYTHONPATH": "/root"})
    .add_local_dir(PROJECT_ROOT / "src" / "yolo", remote_path="/root/yolo")
)

kaggle_volume = modal.Volume.from_name("yolo-data", create_if_missing=True)
KAGGLE_VOLUME_PATH = (  # the path to the volume from within the container
    Path("/root") / ".cache/kagglehub/datasets/"
)

model_volume = modal.Volume.from_name("yolo-checkpoints", create_if_missing=True)
MODEL_DIR = Path("/models")

MINUTES = 60
HOURS = 20
TRAIN_GPU_COUNT = 1
TRAIN_GPU = f"L4:{TRAIN_GPU_COUNT}"

app = modal.App(
    name="yolo-v1-train",
    image=yolo_image,
    volumes={KAGGLE_VOLUME_PATH: kaggle_volume, MODEL_DIR: model_volume},
)


@app.function(
    gpu=TRAIN_GPU,
    timeout=60 * MINUTES * HOURS,
    secrets=[
        modal.Secret.from_name("KAGGLE_USERNAME", required_keys=["KAGGLE_USERNAME"]),
        modal.Secret.from_name("KAGGLE_KEY", required_keys=["KAGGLE_KEY"]),
    ],
)
def run_training(args):
    """Execute the training pipeline with Modal.

    This function runs on Modal infrastructure and handles volume syncing.
    """
    # Make sure volumes are synced (Modal-specific)
    if not modal.is_local():
        model_volume.reload()
        kaggle_volume.reload()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Set up TensorBoard
    writer = None
    log_dir = None
    if args.tensorboard:
        from datetime import datetime

        if args.experiment_name:
            exp_name = args.experiment_name
        else:
            # Auto-generate experiment name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"yolo_{timestamp}"

        log_dir = Path(args.log_dir) / exp_name
        writer = SummaryWriter(log_dir=str(log_dir))
        print_tensorboard_info(log_dir, args.log_dir)

    # Create datasets
    # Training: VOC 2007 trainval + VOC 2012 train
    print("\nCreating training dataset (VOC 2007 trainval + VOC 2012 train)...")
    train_dataset = create_voc_datasets(
        years_and_splits=[("2007", "trainval"), ("2012", "train")],
        download=True,
        S=7,
        B=2,
        augment=not args.no_augment,  # Enable augmentation by default
    )

    # Validation: VOC 2012 val
    print("Creating validation dataset (VOC 2012 val)...")
    val_dataset = create_voc_datasets(
        years_and_splits=[("2012", "val")],
        download=True,
        S=7,
        B=2,
        augment=False,  # Never augment validation set
    )

    print_dataset_info(len(train_dataset), len(val_dataset), not args.no_augment)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == "cuda" else False,
    )

    # Create model
    backbone = ResNetBackbone(pretrained=True, freeze=args.freeze_backbone)
    model = YOLOv1(backbone=backbone, num_classes=args.num_classes, S=7, B=2)
    model = model.to(device)

    # Count and display parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_model_info(total_params, trainable_params)

    # Prepare hyperparameters for logging (will be logged after training with actual metrics)
    hparams = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_coord": args.lambda_coord,
        "lambda_noobj": args.lambda_noobj,
        "freeze_backbone": args.freeze_backbone,
        "num_epochs": args.epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "use_amp": args.use_amp,
    }

    # Create loss function
    criterion = YOLOLoss(
        S=7,
        B=2,
        C=args.num_classes,
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_factor
    )

    # Create GradScaler for automatic mixed precision training
    scaler = None
    if args.use_amp and device == "cuda":
        scaler = GradScaler("cuda")
        print("Automatic Mixed Precision (AMP) enabled")
    elif args.use_amp and device != "cuda":
        print("Warning: AMP requested but not using CUDA device. AMP will be disabled.")

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss_resume = None
    best_map_resume = None

    if args.resume:
        # If resume is True but no path provided, or if it's "true"/"True", use latest checkpoint
        if args.resume == "true" or args.resume == "True" or args.resume is True:
            resume_path = checkpoint_dir / "yolo_latest.pth"
        else:
            resume_path = Path(args.resume)

        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1

            # Try to load best metrics from checkpoint if available
            best_val_loss_resume = checkpoint.get("val_loss", None)
            best_map_resume = checkpoint.get("mAP50:95", None)

            print(
                f"Resumed from epoch {checkpoint.get('epoch', 0)}, starting at epoch {start_epoch}"
            )
            if best_val_loss_resume is not None:
                print(f"  Previous best val_loss: {best_val_loss_resume:.4f}")
            if best_map_resume is not None:
                print(f"  Previous best mAP@0.5:0.95: {best_map_resume:.4f}")
        else:
            print(f"\nWarning: Resume checkpoint not found at {resume_path}")
            print("Starting training from scratch")

    # Train
    print_training_config(args)

    try:
        final_metrics = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            save_frequency=args.save_frequency,
            writer=writer,
            compute_map=args.compute_map,
            map_frequency=args.map_frequency,
            num_classes=args.num_classes,
            start_epoch=start_epoch,
            best_val_loss_init=best_val_loss_resume,
            best_map_init=best_map_resume,
            scaler=scaler,
        )

        # Log hyperparameters with final metrics to TensorBoard
        log_hyperparameters(writer, hparams, final_metrics)

    finally:
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
            if log_dir is not None:
                print(f"\nTensorBoard logs saved to: {log_dir}")

    print("\nTraining completed!")


@app.local_entrypoint()
def main(
    data_root: str = "../data",
    batch_size: int = 64,
    num_workers: int = 32,
    no_augment: bool = False,
    freeze_backbone: bool = False,
    num_classes: int = 20,
    epochs: int = 135,
    lr: float = 1e-4,
    weight_decay: float = 5e-4,
    lr_decay_epochs: str = "75,105",  # Comma-separated epochs
    lr_decay_factor: float = 0.1,
    lambda_coord: float = 5.0,
    lambda_noobj: float = 0.5,
    checkpoint_dir: str = "checkpoints",
    save_frequency: int = 10,
    resume: str = None,
    log_dir: str = "runs",
    experiment_name: str = None,
    tensorboard: bool = False,
    compute_map: bool = False,
    map_frequency: int = 5,
    device: str = None,  # Auto-detect device (mps/cuda/cpu)
    download_data: bool = False,
    remote: bool = False,
    use_amp: bool = False,
):
    """Main entry point when using Modal.

    All training arguments are passed as function parameters to work with Modal's CLI.
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Parse lr_decay_epochs from comma-separated string to list of ints
    lr_decay_epochs_list = [int(x.strip()) for x in lr_decay_epochs.split(",")]

    args = argparse.Namespace(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        no_augment=no_augment,
        freeze_backbone=freeze_backbone,
        num_classes=num_classes,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay_epochs=lr_decay_epochs_list,
        lr_decay_factor=lr_decay_factor,
        lambda_coord=lambda_coord,
        lambda_noobj=lambda_noobj,
        checkpoint_dir=checkpoint_dir,
        save_frequency=save_frequency,
        resume=resume,
        log_dir=log_dir,
        experiment_name=experiment_name,
        tensorboard=tensorboard,
        compute_map=compute_map,
        map_frequency=map_frequency,
        device=device,
        download_data=download_data,
        remote=remote,
        use_amp=use_amp,
    )

    if args.remote and args.checkpoint_dir == "checkpoints":
        args.checkpoint_dir = str(MODEL_DIR / checkpoint_dir)
        print(
            f"Running remotely. Setting checkpoint directory to persistent volume: {args.checkpoint_dir}"
        )

    if remote:
        run_training.remote(args)
    else:
        run_training.local(args)
