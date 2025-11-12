"""Checkpoint management utilities for YOLO v1 training."""

from pathlib import Path

import torch
import torch.optim as optim

from .logging import print_checkpoint_saved


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
