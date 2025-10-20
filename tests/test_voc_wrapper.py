"""Tests for VOCDetectionYOLO wrapper."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yolo.dataset import VOCDetectionYOLO


def main():
    """Test the VOCDetectionYOLO wrapper."""

    print("=" * 70)
    print("Testing VOCDetectionYOLO wrapper")
    print("=" * 70)

    # Create dataset
    print("\nCreating dataset...")
    dataset = VOCDetectionYOLO(
        root="./data",
        year="2007",
        image_set="train",
        download=False,  # Set to True to download if needed
        S=7,
        B=2,
    )

    print("✓ Dataset created successfully")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Number of classes: {dataset.C}")
    print(f"  Grid size: {dataset.S}x{dataset.S}")
    print(f"  Boxes per cell: {dataset.B}")

    # Test loading a sample
    print("\nLoading first sample...")
    image, target = dataset[0]

    print("✓ Sample loaded successfully")
    print(f"  Image shape: {image.shape}")
    print(f"  Target shape: {target.shape}")
    print(
        f"  Expected target shape: ({dataset.S}, {dataset.S}, {5 * dataset.B + dataset.C})"
    )

    # Verify target shape
    expected_shape = (dataset.S, dataset.S, 5 * dataset.B + dataset.C)
    assert target.shape == expected_shape, (
        f"Target shape mismatch! Got {target.shape}, expected {expected_shape}"
    )

    # Count objects in the image
    num_objects = (target[:, :, 4] > 0).sum().item()
    print(f"  Number of objects detected: {num_objects}")

    # Test visualization
    print("\nTesting visualization...")
    sample_info = dataset.visualize_sample(0)
    print("✓ Visualization data retrieved")
    print(f"  Image path: {sample_info['image_path']}")
    print(f"  Number of objects: {len(sample_info['bboxes'])}")

    if len(sample_info["class_names"]) > 0:
        print(f"  Classes in image: {', '.join(sample_info['class_names'])}")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
