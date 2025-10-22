"""
Example script demonstrating mAP metrics usage.

This script shows how to use the mAPMetric class to evaluate model performance.
"""

import torch
from torch.utils.data import DataLoader

# Import from your yolo package
import sys

sys.path.append("./src")

from yolo import ResNetBackbone, YOLOv1, evaluate_model
from yolo.dataset import VOCDetectionYOLO
from yolo.metrics import mAPMetric


def example_basic_usage():
    """Basic usage of mAPMetric."""
    print("=" * 70)
    print("Example 1: Basic mAPMetric Usage")
    print("=" * 70)

    # Initialize metric
    metric = mAPMetric(
        num_classes=20,
        iou_threshold=0.5,
        conf_threshold=0.01,
        nms_threshold=0.4,
    )

    # Create dummy predictions and ground truths
    batch_size = 4
    predictions = torch.randn(batch_size, 7, 7, 30)  # S=7, B=2, C=20
    targets = torch.randn(batch_size, 7, 7, 30)

    # Ensure targets have some ground truth boxes
    targets[:, 3, 3, 4] = 1.0  # Set confidence for a box
    targets[:, 3, 3, 10] = 1.0  # Set class probability

    # Update metric
    metric.update(predictions, targets)

    # Compute results
    results = metric.compute()

    print(f"\nmAP: {results['mAP']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print()


def example_model_evaluation():
    """Example of evaluating a model on a dataset."""
    print("=" * 70)
    print("Example 2: Full Model Evaluation")
    print("=" * 70)

    # Setup device
    device = "cpu"  # Use CPU for example

    # Create model
    print("Creating model...")
    backbone = ResNetBackbone(pretrained=False, freeze=False)
    model = YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)
    model = model.to(device)
    model.eval()

    # Create a small dataset (just for demonstration)
    print("Loading dataset...")
    try:
        dataset = VOCDetectionYOLO(
            root="./data",
            year="2007",
            image_set="val",
            download=False,
            S=7,
            B=2,
        )

        # Use only a small subset for quick demo
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(10, len(dataset))))

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        print(f"Evaluating on {len(dataset)} images...")

        # Evaluate
        results = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=device,
            num_classes=20,
            iou_threshold=0.5,
        )

        print(f"\nmAP@0.5: {results['mAP']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")

        # Show per-class AP for first few classes
        print("\nPer-class AP (first 5 classes):")
        for i in range(5):
            ap = results[f"AP_class_{i}"]
            print(f"  Class {i}: {ap:.4f}")

    except Exception as e:
        print(f"Note: Could not load dataset - {e}")
        print("This is normal if you haven't downloaded the VOC dataset yet.")
        print("The metric implementation is still functional!")

    print()


def example_manual_metric_calculation():
    """Example showing manual step-by-step metric calculation."""
    print("=" * 70)
    print("Example 3: Manual Metric Calculation")
    print("=" * 70)

    # Create metric
    metric = mAPMetric(num_classes=20, S=7, B=2)
    metric.reset()

    # Simulate multiple batches
    num_batches = 3
    print(f"\nProcessing {num_batches} batches...")

    for batch_idx in range(num_batches):
        # Create predictions and targets for this batch
        predictions = torch.randn(2, 7, 7, 30)
        targets = torch.zeros(2, 7, 7, 30)

        # Add some ground truth objects
        targets[:, 2, 2, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])
        targets[:, 2, 2, 10] = 1.0  # Class 0

        # Update metric
        metric.update(predictions, targets)
        print(f"  Batch {batch_idx + 1} processed")

    # Compute final results
    print("\nComputing final metrics...")
    results = metric.compute()

    print(f"\nFinal Results:")
    print(f"  mAP: {results['mAP']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print()


def example_iou_calculation():
    """Example showing IoU calculation."""
    print("=" * 70)
    print("Example 4: IoU Calculation")
    print("=" * 70)

    metric = mAPMetric(num_classes=20)

    # Example 1: Identical boxes
    box1 = (0.5, 0.5, 0.2, 0.2)
    box2 = (0.5, 0.5, 0.2, 0.2)
    iou = metric._calculate_iou(box1, box2)
    print(f"\nIdentical boxes: IoU = {iou:.4f}")

    # Example 2: Non-overlapping boxes
    box1 = (0.2, 0.2, 0.1, 0.1)
    box2 = (0.8, 0.8, 0.1, 0.1)
    iou = metric._calculate_iou(box1, box2)
    print(f"Non-overlapping boxes: IoU = {iou:.4f}")

    # Example 3: Partially overlapping boxes
    box1 = (0.5, 0.5, 0.4, 0.4)
    box2 = (0.6, 0.6, 0.4, 0.4)
    iou = metric._calculate_iou(box1, box2)
    print(f"Partially overlapping boxes: IoU = {iou:.4f}")

    # Example 4: Contained box
    box1 = (0.5, 0.5, 0.6, 0.6)
    box2 = (0.5, 0.5, 0.3, 0.3)
    iou = metric._calculate_iou(box1, box2)
    print(f"Contained box: IoU = {iou:.4f}")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("YOLO v1 mAP Metrics - Usage Examples")
    print("*" * 70)
    print()

    # Run examples
    example_basic_usage()
    example_manual_metric_calculation()
    example_iou_calculation()
    example_model_evaluation()

    print("*" * 70)
    print("Examples completed!")
    print("*" * 70)
    print()
