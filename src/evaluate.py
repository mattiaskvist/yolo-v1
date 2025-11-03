"""
Evaluation script for YOLO v1 model.

This script evaluates a trained YOLO model on a dataset and computes mAP metrics.
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yolo import ResNetBackbone, YOLOv1, evaluate_model
from yolo.dataset import VOCDetectionYOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO v1 model")

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=20,
        help="Number of object classes",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ResNet backbone (should match training config)",
    )

    parser.add_argument(
        "--year",
        type=str,
        default="2007",
        help="VOC dataset year (2007, 2012)",
    )
    parser.add_argument(
        "--image-set",
        type=str,
        default="val",
        help="Image set to evaluate on (train, val, test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.01,
        help="Confidence threshold for filtering predictions",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="NMS threshold",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to use for evaluation",
    )

    args = parser.parse_args()

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.year} {args.image_set} dataset")
    dataset = VOCDetectionYOLO(
        year=args.year,
        image_set=args.image_set,
        download=True,
        S=7,
        B=2,
        augment=False,
    )
    print(f"Dataset size: {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
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

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if "epoch" in checkpoint:
        print(f"Checkpoint from epoch {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    if "mAP50:95" in checkpoint:
        print(f"mAP50:95 (from checkpoint): {checkpoint['mAP50:95']:.4f}")
    elif "mAP" in checkpoint:
        print(f"mAP (from checkpoint): {checkpoint['mAP']:.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Evaluate
    print(f"\nEvaluating on {len(dataset)} images...")

    results = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=args.num_classes,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        S=7,
        B=2,
    )

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"mAP50:95: {results['mAP50:95']:.4f}")
    print(f"mAP@0.5: {results['mAP50']:.4f}")
    print(f"mAP@0.75: {results['mAP75']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print()

    # Print per-class AP
    print("Per-class Average Precision (mAP50:95):")
    print("-" * 70)
    class_names = VOCDetectionYOLO.VOC_CLASSES

    # Sort classes by AP for better readability
    class_aps = [(i, results[f"AP50:95_class_{i}"]) for i in range(args.num_classes)]
    class_aps.sort(key=lambda x: x[1], reverse=True)

    for class_id, ap in class_aps:
        class_name = class_names[class_id]
        print(f"  {class_name:15s}: {ap:.4f}")

    print("=" * 70)

    # Save results to file
    results_file = Path(args.checkpoint).parent / "evaluation_results.txt"
    with open(results_file, "w") as f:
        f.write("YOLO v1 Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: VOC{args.year} {args.image_set}\n")
        f.write(f"Number of images: {len(dataset)}\n")
        f.write(f"\nmAP50:95: {results['mAP50:95']:.4f}\n")
        f.write(f"mAP@0.5: {results['mAP50']:.4f}\n")
        f.write(f"mAP@0.75: {results['mAP75']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write("\nPer-class Average Precision (mAP50:95):\n")
        f.write("-" * 70 + "\n")
        for class_id, ap in class_aps:
            class_name = class_names[class_id]
            f.write(f"  {class_name:15s}: {ap:.4f}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
