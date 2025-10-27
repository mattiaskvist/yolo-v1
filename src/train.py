"""Training script for YOLO v1 with ResNet backbone."""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from yolo import ResNetBackbone, YOLOv1, evaluate_model
from yolo.dataset import VOCDetectionYOLO
from yolo.loss import YOLOLoss


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: YOLOLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    writer: SummaryWriter = None,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    coord_loss = 0
    conf_obj_loss = 0
    conf_noobj_loss = 0
    class_loss = 0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Calculate loss
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
            if writer is not None:
                global_step = (epoch - 1) * len(dataloader) + batch_idx
                writer.add_scalar("batch/loss_total", loss_dict["total"], global_step)
                writer.add_scalar("batch/loss_coord", loss_dict["coord"], global_step)
                writer.add_scalar(
                    "batch/loss_conf_obj", loss_dict["conf_obj"], global_step
                )
                writer.add_scalar(
                    "batch/loss_conf_noobj", loss_dict["conf_noobj"], global_step
                )
                writer.add_scalar("batch/loss_class", loss_dict["class"], global_step)

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
            iou_threshold=0.5,
            conf_threshold=0.01,
            nms_threshold=0.4,
        )
        results["mAP"] = map_results["mAP"]
        results["precision"] = map_results["precision"]
        results["recall"] = map_results["recall"]

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

    Returns:
        Dictionary containing final training metrics (best_val_loss, final_train_loss, etc.)
    """

    best_val_loss = float("inf")
    best_map = 0.0
    final_train_loss = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 70}")

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        print(f"\nTraining - Epoch {epoch} Average Loss: {train_losses['total']:.4f}")

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

        print(f"Validation - Epoch {epoch} Average Loss: {val_losses['total']:.4f}")
        print(f"  Coord: {val_losses['coord']:.4f}")
        print(f"  Conf (obj): {val_losses['conf_obj']:.4f}")
        print(f"  Conf (noobj): {val_losses['conf_noobj']:.4f}")
        print(f"  Class: {val_losses['class']:.4f}")

        # Print mAP metrics if computed
        if "mAP" in val_losses:
            print(f"  mAP@0.5: {val_losses['mAP']:.4f}")
            print(f"  Precision: {val_losses['precision']:.4f}")
            print(f"  Recall: {val_losses['recall']:.4f}")

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")

        # Log epoch metrics to TensorBoard
        if writer is not None:
            writer.add_scalar("epoch/train_loss_total", train_losses["total"], epoch)
            writer.add_scalar("epoch/train_loss_coord", train_losses["coord"], epoch)
            writer.add_scalar(
                "epoch/train_loss_conf_obj", train_losses["conf_obj"], epoch
            )
            writer.add_scalar(
                "epoch/train_loss_conf_noobj", train_losses["conf_noobj"], epoch
            )
            writer.add_scalar("epoch/train_loss_class", train_losses["class"], epoch)

            writer.add_scalar("epoch/val_loss_total", val_losses["total"], epoch)
            writer.add_scalar("epoch/val_loss_coord", val_losses["coord"], epoch)
            writer.add_scalar("epoch/val_loss_conf_obj", val_losses["conf_obj"], epoch)
            writer.add_scalar(
                "epoch/val_loss_conf_noobj", val_losses["conf_noobj"], epoch
            )
            writer.add_scalar("epoch/val_loss_class", val_losses["class"], epoch)

            writer.add_scalar("epoch/learning_rate", current_lr, epoch)

            # Log mAP metrics if computed
            if "mAP" in val_losses:
                writer.add_scalar("epoch/mAP", val_losses["mAP"], epoch)
                writer.add_scalar("epoch/precision", val_losses["precision"], epoch)
                writer.add_scalar("epoch/recall", val_losses["recall"], epoch)

        # Save checkpoint
        if epoch % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f"yolo_epoch_{epoch}.pth"
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_losses["total"],
                "val_loss": val_losses["total"],
            }
            if "mAP" in val_losses:
                checkpoint_data["mAP"] = val_losses["mAP"]

            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model (by validation loss)
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_model_path = checkpoint_dir / "yolo_best.pth"
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_losses["total"],
            }
            if "mAP" in val_losses:
                checkpoint_data["mAP"] = val_losses["mAP"]

            torch.save(checkpoint_data, best_model_path)
            print(
                f"Best model saved to {best_model_path} (val_loss: {best_val_loss:.4f})"
            )

        # Track best mAP
        if "mAP" in val_losses and val_losses["mAP"] > best_map:
            best_map = val_losses["mAP"]
            best_map_path = checkpoint_dir / "yolo_best_map.pth"
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_losses["total"],
                "mAP": val_losses["mAP"],
            }
            torch.save(checkpoint_data, best_map_path)
            print(f"Best mAP model saved to {best_map_path} (mAP: {best_map:.4f})")

        # Update final training loss
        final_train_loss = train_losses["total"]

    # Return final metrics
    results = {
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
    }
    if best_map > 0:
        results["best_mAP"] = best_map

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO v1 with ResNet backbone")

    # Data
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data",
        help="Path to VOC dataset root (where VOCdevkit will be created/exists)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of data loading workers"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation (scaling, translation, HSV adjustments)",
    )

    # Model
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet backbone weights",
    )
    parser.add_argument(
        "--num-classes", type=int, default=20, help="Number of object classes"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=135, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        nargs="+",
        default=[75, 105],
        help="Epochs to decay learning rate",
    )
    parser.add_argument(
        "--lr-decay-factor", type=float, default=0.1, help="LR decay factor"
    )

    # Loss
    parser.add_argument(
        "--lambda-coord", type=float, default=5.0, help="Coordinate loss weight"
    )
    parser.add_argument(
        "--lambda-noobj", type=float, default=0.5, help="No-object loss weight"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    # Experiment tracking
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (auto-generated if not provided)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )

    # Evaluation
    parser.add_argument(
        "--compute-map",
        action="store_true",
        help="Compute mAP metrics during validation (slower but more informative)",
    )
    parser.add_argument(
        "--map-frequency",
        type=int,
        default=5,
        help="Compute mAP every N epochs",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for training",
    )

    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download the VOC dataset from Kaggle (recommended - fast and reliable)",
    )

    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Set up TensorBoard
    writer = None
    log_dir = None
    if not args.no_tensorboard:
        from datetime import datetime

        if args.experiment_name:
            exp_name = args.experiment_name
        else:
            # Auto-generate experiment name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"yolo_{timestamp}"

        log_dir = Path(args.log_dir) / exp_name
        writer = SummaryWriter(log_dir=str(log_dir))
        print("\nTensorBoard logging enabled")
        print(f"Log directory: {log_dir}")
        print(f"To view logs, run: tensorboard --logdir={args.log_dir}\n")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = VOCDetectionYOLO(
        root=args.data_root,
        year="2007",
        image_set="train",
        download=args.download_data,
        S=7,
        B=2,
        augment=not args.no_augment,  # Enable augmentation by default
    )

    val_dataset = VOCDetectionYOLO(
        root=args.data_root,
        year="2007",
        image_set="val",
        download=False,  # Don't re-download for validation set
        S=7,
        B=2,
        augment=False,  # Never augment validation set
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")
    print(f"Data augmentation: {'ENABLED' if not args.no_augment else 'DISABLED'}")
    if not args.no_augment:
        print("  - Random scaling and translation (up to 20%)")
        print("  - HSV color adjustments (exposure & saturation up to 1.5x)")

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
    print("\nCreating model...")
    backbone = ResNetBackbone(pretrained=True, freeze=args.freeze_backbone)
    model = YOLOv1(backbone=backbone, num_classes=args.num_classes, S=7, B=2)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

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

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        resumed_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from epoch {resumed_epoch}")

    # Train
    print("\nStarting training...")
    print("Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Freeze backbone: {args.freeze_backbone}")
    print(f"  Lambda coord: {args.lambda_coord}")
    print(f"  Lambda noobj: {args.lambda_noobj}")

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
        )

        # Log hyperparameters with final metrics to TensorBoard
        if writer is not None:
            metric_dict = {
                "hparam/best_val_loss": final_metrics["best_val_loss"],
                "hparam/final_train_loss": final_metrics["final_train_loss"],
            }
            if "best_mAP" in final_metrics:
                metric_dict["hparam/best_mAP"] = final_metrics["best_mAP"]
            writer.add_hparams(hparams, metric_dict)

    finally:
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
            if log_dir is not None:
                print(f"\nTensorBoard logs saved to: {log_dir}")

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
