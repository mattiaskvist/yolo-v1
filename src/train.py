"""Training script for YOLO v1 with ResNet backbone."""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from yolo import ResNetBackbone, YOLOv1
from yolo.dataset import VOCDetectionYOLO
from yolo.loss import YOLOLoss


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: YOLOLoss,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
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

        # Print progress
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
) -> dict[str, float]:
    """Validate the model."""
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

    # Return average losses
    return {
        "total": total_loss / num_batches,
        "coord": coord_loss / num_batches,
        "conf_obj": conf_obj_loss / num_batches,
        "conf_noobj": conf_noobj_loss / num_batches,
        "class": class_loss / num_batches,
    }


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
):
    """Main training loop."""

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 70}")

        # Train
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        print(f"\nTraining - Epoch {epoch} Average Loss: {train_losses['total']:.4f}")

        # Validate
        print("\nValidating...")
        val_losses = validate(model, val_loader, criterion, device)

        print(f"Validation - Epoch {epoch} Average Loss: {val_losses['total']:.4f}")
        print(f"  Coord: {val_losses['coord']:.4f}")
        print(f"  Conf (obj): {val_losses['conf_obj']:.4f}")
        print(f"  Conf (noobj): {val_losses['conf_noobj']:.4f}")
        print(f"  Class: {val_losses['class']:.4f}")

        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint
        if epoch % save_frequency == 0:
            checkpoint_path = checkpoint_dir / f"yolo_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_losses["total"],
                    "val_loss": val_losses["total"],
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            best_model_path = checkpoint_dir / "yolo_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_losses["total"],
                },
                best_model_path,
            )
            print(
                f"Best model saved to {best_model_path} (val_loss: {best_val_loss:.4f})"
            )


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
        "--num-workers", type=int, default=4, help="Number of data loading workers"
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

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for training",
    )

    args = parser.parse_args()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = VOCDetectionYOLO(
        root=args.data_root,
        year="2007",
        image_set="train",
        download=False,
        S=7,
        B=2,
    )

    val_dataset = VOCDetectionYOLO(
        root=args.data_root,
        year="2007",
        image_set="val",
        download=False,
        S=7,
        B=2,
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

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

    train(
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
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
