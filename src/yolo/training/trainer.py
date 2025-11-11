"""Core training logic for YOLO v1."""

import time

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..loss import YOLOLoss
from ..metrics import evaluate_model
from .checkpoints import save_best_map_model, save_best_model, save_checkpoint
from .logging import (
    log_batch_metrics,
    log_epoch_metrics,
    print_epoch_header,
    print_loss_metrics,
    print_map_metrics,
)


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
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        writer: TensorBoard writer (optional)
        scaler: GradScaler for automatic mixed precision training (optional)

    Returns:
        Dictionary containing average losses for the epoch
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
    checkpoint_dir,
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
