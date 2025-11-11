"""Logging utilities for YOLO v1 training."""

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


# ============================================================================
# Console Printing Utilities
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
