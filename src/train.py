"""Training script for YOLO v1 with ResNet backbone."""

import argparse
import time
from pathlib import Path

import modal
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from yolo import ResNetBackbone, YOLOv1, evaluate_model
from yolo.dataset import create_voc_datasets
from yolo.loss import YOLOLoss


# ============================================================================
# Logging/Printing Utilities
# ============================================================================


def print_epoch_header(epoch: int, num_epochs: int):
    """Print epoch header with separators."""
    print(f"\n{'=' * 70}")
    print(f"Epoch {epoch}/{num_epochs}")
    print(f"{'=' * 70}")


def print_loss_metrics(
    prefix: str, losses: dict[str, float], epoch: int = None
) -> None:
    """Print loss metrics in a formatted way.

    Args:
        prefix: Prefix label (e.g., "Training", "Validation")
        losses: Dictionary of loss values
        epoch: Optional epoch number to include in output
    """
    epoch_str = f" - Epoch {epoch}" if epoch is not None else ""
    print(f"\n{prefix}{epoch_str} Average Loss: {losses['total']:.4f}")
    print(f"  Coord: {losses['coord']:.4f}")
    print(f"  Conf (obj): {losses['conf_obj']:.4f}")
    print(f"  Conf (noobj): {losses['conf_noobj']:.4f}")
    print(f"  Class: {losses['class']:.4f}")


def print_map_metrics(metrics: dict[str, float]) -> None:
    """Print mAP metrics in a formatted way.

    Args:
        metrics: Dictionary containing mAP, precision, recall values
    """
    if "mAP50:95" not in metrics:
        return

    print(f"  mAP@0.5:0.95: {metrics['mAP50:95']:.4f}")
    print(f"  mAP@0.5: {metrics['mAP50']:.4f}")
    print(f"  mAP@0.75: {metrics['mAP75']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")

    # Print size-based metrics if available
    if "mAP50:95_small" in metrics:
        print(f"  mAP@0.5:0.95 (small): {metrics['mAP50:95_small']:.4f}")
    if "mAP50:95_medium" in metrics:
        print(f"  mAP@0.5:0.95 (medium): {metrics['mAP50:95_medium']:.4f}")
    if "mAP50:95_large" in metrics:
        print(f"  mAP@0.5:0.95 (large): {metrics['mAP50:95_large']:.4f}")


def print_dataset_info(
    train_size: int, val_size: int, augmentation_enabled: bool
) -> None:
    """Print dataset loading information.

    Args:
        train_size: Number of training images
        val_size: Number of validation images
        augmentation_enabled: Whether data augmentation is enabled
    """
    print("\nLoading datasets...")
    print(f"Train dataset: {train_size} images")
    print(f"Val dataset: {val_size} images")
    print(f"Data augmentation: {'ENABLED' if augmentation_enabled else 'DISABLED'}")
    if augmentation_enabled:
        print("  - Random scaling and translation (up to 20%)")
        print("  - HSV color adjustments (exposure & saturation up to 1.5x)")


def print_model_info(total_params: int, trainable_params: int) -> None:
    """Print model parameter information.

    Args:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    print("\nCreating model...")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")


def print_training_config(args) -> None:
    """Print training configuration parameters.

    Args:
        args: Parsed command line arguments
    """
    print("\nStarting training...")
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Lambda coord: {args.lambda_coord}")
    print(f"  Lambda noobj: {args.lambda_noobj}")
    print(f"  Mixed Precision (AMP): {'ENABLED' if args.use_amp else 'DISABLED'}")


def print_tensorboard_info(log_dir: Path, log_root: str) -> None:
    """Print TensorBoard logging information.

    Args:
        log_dir: Directory where logs are being saved
        log_root: Root directory for tensorboard command
    """
    print("\nTensorBoard logging enabled")
    print(f"Log directory: {log_dir}")
    print(f"To view logs, run: tensorboard --logdir={log_root}\n")


def print_checkpoint_saved(
    checkpoint_path: Path, metric_name: str = None, metric_value: float = None
) -> None:
    """Print checkpoint save notification.

    Args:
        checkpoint_path: Path where checkpoint was saved
        metric_name: Optional metric name (e.g., "val_loss", "mAP@0.5:0.95")
        metric_value: Optional metric value
    """
    if metric_name and metric_value is not None:
        print(
            f"Best model saved to {checkpoint_path} ({metric_name}: {metric_value:.4f})"
        )
    else:
        print(f"Checkpoint saved to {checkpoint_path}")


# ============================================================================
# Checkpoint Saving Functions
# ============================================================================


def save_checkpoint(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    train_losses: dict[str, float],
    val_losses: dict[str, float],
) -> None:
    """Save a regular checkpoint during training.

    Args:
        checkpoint_path: Path where checkpoint will be saved
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save
        train_losses: Dictionary containing training loss values
        val_losses: Dictionary containing validation loss and metric values
    """
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_losses["total"],
        "val_loss": val_losses["total"],
    }
    if "mAP50:95" in val_losses:
        checkpoint_data["mAP50:95"] = val_losses["mAP50:95"]
        checkpoint_data["mAP50"] = val_losses["mAP50"]
        checkpoint_data["mAP75"] = val_losses["mAP75"]

    torch.save(checkpoint_data, checkpoint_path)
    print_checkpoint_saved(checkpoint_path)


def save_best_model(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    val_losses: dict[str, float],
    metric_name: str,
    metric_value: float,
) -> None:
    """Save the best model checkpoint.

    Args:
        checkpoint_path: Path where checkpoint will be saved
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        val_losses: Dictionary containing validation loss and metric values
        metric_name: Name of the metric being optimized (e.g., "val_loss", "mAP@0.5:0.95")
        metric_value: Value of the metric
    """
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses["total"],
    }
    if "mAP50:95" in val_losses:
        checkpoint_data["mAP50:95"] = val_losses["mAP50:95"]
        checkpoint_data["mAP50"] = val_losses["mAP50"]
        checkpoint_data["mAP75"] = val_losses["mAP75"]

    torch.save(checkpoint_data, checkpoint_path)
    print_checkpoint_saved(checkpoint_path, metric_name, metric_value)


def save_best_map_model(
    checkpoint_path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    val_losses: dict[str, float],
    map_value: float,
) -> None:
    """Save the best mAP model checkpoint.

    Args:
        checkpoint_path: Path where checkpoint will be saved
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        val_losses: Dictionary containing validation loss and metric values
        map_value: mAP@0.5:0.95 value
    """
    checkpoint_data = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_losses["total"],
        "mAP50:95": val_losses["mAP50:95"],
        "mAP50": val_losses["mAP50"],
        "mAP75": val_losses["mAP75"],
    }
    torch.save(checkpoint_data, checkpoint_path)
    print_checkpoint_saved(checkpoint_path, "mAP@0.5:0.95", map_value)


# ============================================================================
# TensorBoard Writer Functions
# ============================================================================


def log_batch_metrics(
    writer: SummaryWriter,
    loss_dict: dict[str, float],
    epoch: int,
    batch_idx: int,
    num_batches: int,
) -> None:
    """Log batch-level metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        loss_dict: Dictionary containing loss values
        epoch: Current epoch number
        batch_idx: Current batch index (0-indexed)
        num_batches: Total number of batches in epoch
    """
    if writer is None:
        return

    global_step = (epoch - 1) * num_batches + batch_idx
    writer.add_scalar("batch/loss_total", loss_dict["total"], global_step)
    writer.add_scalar("batch/loss_coord", loss_dict["coord"], global_step)
    writer.add_scalar("batch/loss_conf_obj", loss_dict["conf_obj"], global_step)
    writer.add_scalar("batch/loss_conf_noobj", loss_dict["conf_noobj"], global_step)
    writer.add_scalar("batch/loss_class", loss_dict["class"], global_step)


def log_epoch_metrics(
    writer: SummaryWriter,
    train_losses: dict[str, float],
    val_losses: dict[str, float],
    learning_rate: float,
    epoch: int,
) -> None:
    """Log epoch-level metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        train_losses: Dictionary containing training loss values
        val_losses: Dictionary containing validation loss and metric values
        learning_rate: Current learning rate
        epoch: Current epoch number
    """
    if writer is None:
        return

    # Log training losses
    writer.add_scalar("epoch/train_loss_total", train_losses["total"], epoch)
    writer.add_scalar("epoch/train_loss_coord", train_losses["coord"], epoch)
    writer.add_scalar("epoch/train_loss_conf_obj", train_losses["conf_obj"], epoch)
    writer.add_scalar("epoch/train_loss_conf_noobj", train_losses["conf_noobj"], epoch)
    writer.add_scalar("epoch/train_loss_class", train_losses["class"], epoch)

    # Log validation losses
    writer.add_scalar("epoch/val_loss_total", val_losses["total"], epoch)
    writer.add_scalar("epoch/val_loss_coord", val_losses["coord"], epoch)
    writer.add_scalar("epoch/val_loss_conf_obj", val_losses["conf_obj"], epoch)
    writer.add_scalar("epoch/val_loss_conf_noobj", val_losses["conf_noobj"], epoch)
    writer.add_scalar("epoch/val_loss_class", val_losses["class"], epoch)

    # Log learning rate
    writer.add_scalar("epoch/learning_rate", learning_rate, epoch)

    # Log mAP metrics if computed
    if "mAP50:95" in val_losses:
        writer.add_scalar("epoch/mAP50:95", val_losses["mAP50:95"], epoch)
        writer.add_scalar("epoch/mAP50", val_losses["mAP50"], epoch)
        writer.add_scalar("epoch/mAP75", val_losses["mAP75"], epoch)
        writer.add_scalar("epoch/precision", val_losses["precision"], epoch)
        writer.add_scalar("epoch/recall", val_losses["recall"], epoch)

        # Log size-based metrics if available
        if "mAP50:95_small" in val_losses:
            writer.add_scalar(
                "epoch/mAP50:95_small", val_losses["mAP50:95_small"], epoch
            )
        if "mAP50:95_medium" in val_losses:
            writer.add_scalar(
                "epoch/mAP50:95_medium", val_losses["mAP50:95_medium"], epoch
            )
        if "mAP50:95_large" in val_losses:
            writer.add_scalar(
                "epoch/mAP50:95_large", val_losses["mAP50:95_large"], epoch
            )


def log_hyperparameters(
    writer: SummaryWriter,
    hparams: dict,
    final_metrics: dict[str, float],
) -> None:
    """Log hyperparameters and final metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        hparams: Dictionary containing hyperparameter values
        final_metrics: Dictionary containing final metric values
    """
    if writer is None:
        return

    metric_dict = {
        "hparam/best_val_loss": final_metrics["best_val_loss"],
        "hparam/final_train_loss": final_metrics["final_train_loss"],
    }
    if "best_mAP50:95" in final_metrics:
        metric_dict["hparam/best_mAP50:95"] = final_metrics["best_mAP50:95"]

    writer.add_hparams(hparams, metric_dict)


# ============================================================================
# Training Functions
# ============================================================================


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: YOLOLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter = None,
    scaler: GradScaler = None,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        scaler: GradScaler for automatic mixed precision training (optional)
    """
    model.train()

    total_loss = 0
    coord_loss = 0
    conf_obj_loss = 0
    conf_noobj_loss = 0
    class_loss = 0
    num_batches = 0

    start_time = time.time()
    use_amp = scaler is not None

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()

        # Use automatic mixed precision if enabled
        if use_amp:
            with autocast("cuda"):
                predictions = model(images)
                loss, loss_dict = criterion(predictions, targets)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient clipping (unscale gradients first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

        # Accumulate losses
        total_loss += loss_dict["total"]
        coord_loss += loss_dict["coord"]
        conf_obj_loss += loss_dict["conf_obj"]
        conf_noobj_loss += loss_dict["conf_noobj"]
        class_loss += loss_dict["class"]
        num_batches += 1

        # Print progress and log to tensorboard
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss_dict['total']:.4f} "
                f"(coord: {loss_dict['coord']:.4f}, "
                f"conf_obj: {loss_dict['conf_obj']:.4f}, "
                f"conf_noobj: {loss_dict['conf_noobj']:.4f}, "
                f"class: {loss_dict['class']:.4f}) "
                f"Time: {elapsed:.2f}s"
            )

            # Log batch losses to TensorBoard
            log_batch_metrics(writer, loss_dict, epoch, batch_idx, len(dataloader))

            start_time = time.time()

    # Return average losses
    return {
        "total": total_loss / num_batches,
        "coord": coord_loss / num_batches,
        "conf_obj": conf_obj_loss / num_batches,
        "conf_noobj": conf_noobj_loss / num_batches,
        "class": class_loss / num_batches,
    }


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: YOLOLoss,
    device: str,
    compute_map: bool = False,
    num_classes: int = 20,
) -> dict[str, float]:
    """Validate the model.

    Args:
        model: The model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to run validation on
        compute_map: Whether to compute mAP metrics (slower but more informative)
        num_classes: Number of object classes

    Returns:
        Dictionary containing validation metrics
    """
    model.eval()

    total_loss = 0
    coord_loss = 0
    conf_obj_loss = 0
    conf_noobj_loss = 0
    class_loss = 0
    num_batches = 0

    with torch.no_grad():
        with autocast(device_type=device):
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)

                # Forward pass
                predictions = model(images)

                # Calculate loss
                _, loss_dict = criterion(predictions, targets)

                # Accumulate losses
                total_loss += loss_dict["total"]
                coord_loss += loss_dict["coord"]
                conf_obj_loss += loss_dict["conf_obj"]
                conf_noobj_loss += loss_dict["conf_noobj"]
                class_loss += loss_dict["class"]
                num_batches += 1

    # Prepare results
    results = {
        "total": total_loss / num_batches,
        "coord": coord_loss / num_batches,
        "conf_obj": conf_obj_loss / num_batches,
        "conf_noobj": conf_noobj_loss / num_batches,
        "class": class_loss / num_batches,
    }

    # Compute mAP if requested
    if compute_map:
        print("\n  Computing mAP metrics...")
        map_results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            num_classes=num_classes,
            iou_thresholds=None,  # Use default 0.5:0.95 range
            conf_threshold=0.01,
            nms_threshold=0.4,
        )
        # Add all mAP metrics to results
        results["mAP50:95"] = map_results["mAP50:95"]
        results["mAP50"] = map_results["mAP50"]
        results["mAP75"] = map_results["mAP75"]
        results["precision"] = map_results["precision"]
        results["recall"] = map_results["recall"]

        # Add size-based metrics if available
        for key in ["mAP50:95_small", "mAP50:95_medium", "mAP50:95_large"]:
            if key in map_results:
                results[key] = map_results[key]

    return results


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: YOLOLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: str,
    num_epochs: int,
    checkpoint_dir: Path,
    save_frequency: int = 5,
    writer: SummaryWriter = None,
    compute_map: bool = False,
    map_frequency: int = 5,
    num_classes: int = 20,
    start_epoch: int = 1,
    best_val_loss_init: float = None,
    best_map_init: float = None,
    scaler: GradScaler = None,
) -> dict[str, float]:
    """Main training loop.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N epochs
        writer: TensorBoard writer
        compute_map: Whether to compute mAP during validation
        map_frequency: Compute mAP every N epochs
        num_classes: Number of object classes
        start_epoch: Epoch number to start from (for resuming training)
        best_val_loss_init: Initial best validation loss (for resuming)
        best_map_init: Initial best mAP (for resuming)
        scaler: GradScaler for automatic mixed precision training (optional)

    Returns:
        Dictionary containing final training metrics (best_val_loss, final_train_loss, etc.)
    """

    best_val_loss = (
        best_val_loss_init if best_val_loss_init is not None else float("inf")
    )
    best_map = best_map_init if best_map_init is not None else 0.0
    final_train_loss = None

    for epoch in range(start_epoch, num_epochs + 1):
        print_epoch_header(epoch, num_epochs)

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, scaler
        )
        print_loss_metrics("Training", train_losses, epoch)

        # Validate
        print("\nValidating...")
        should_compute_map = compute_map and (
            epoch % map_frequency == 0 or epoch == num_epochs
        )
        val_losses = validate(
            model,
            val_loader,
            criterion,
            device,
            compute_map=should_compute_map,
            num_classes=num_classes,
        )

        print_loss_metrics("Validation", val_losses, epoch)
        print_map_metrics(val_losses)

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")

        # Log epoch metrics to TensorBoard
        log_epoch_metrics(writer, train_losses, val_losses, current_lr, epoch)

        # Always save latest checkpoint after every epoch
        latest_checkpoint_path = checkpoint_dir / "yolo_latest.pth"
        save_checkpoint(
            latest_checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            train_losses,
            val_losses,
        )

        # Save checkpoint at specified frequency
        if epoch % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f"yolo_epoch_{epoch}.pth"
            save_checkpoint(
                checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                train_losses,
                val_losses,
            )

        # Save best model (by validation loss)
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_model_path = checkpoint_dir / "yolo_best.pth"
            save_best_model(
                best_model_path,
                epoch,
                model,
                optimizer,
                val_losses,
                "val_loss",
                best_val_loss,
            )

        # Track best mAP (using mAP50:95 as the primary metric)
        if "mAP50:95" in val_losses and val_losses["mAP50:95"] > best_map:
            best_map = val_losses["mAP50:95"]
            best_map_path = checkpoint_dir / "yolo_best_map.pth"
            save_best_map_model(
                best_map_path, epoch, model, optimizer, val_losses, best_map
            )

        # Update final training loss
        final_train_loss = train_losses["total"]

    # Return final metrics
    results = {
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
    }
    if best_map > 0:
        results["best_mAP50:95"] = best_map

    return results


# Modal setup
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
    device: str = "cuda",  # this will execute locally, so default to cuda
    download_data: bool = False,
    remote: bool = False,
    use_amp: bool = False,
):
    """Main entry point when using Modal.

    All training arguments are passed as function parameters to work with Modal's CLI.
    """
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
