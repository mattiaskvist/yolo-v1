"""Example usage of YOLO v1 dataloaders."""

import sys
from pathlib import Path

from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.data import PascalVOCDataset
from yolo.data.download import setup_pascal_voc
from yolo.data.utils import draw_bboxes


def main():
    """Demonstrate basic usage of the Pascal VOC dataloader."""

    # Setup Pascal VOC dataset
    print("Setting up Pascal VOC 2007 dataset...")
    try:
        voc_root = setup_pascal_voc(
            root_dir=Path("data"),
            year="2007",
            split="trainval",
        )
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        return

    # Create dataset
    print("Creating Pascal VOC dataset...")
    dataset = PascalVOCDataset(
        root_dir=voc_root,
        split="train",
        S=7,  # Grid size
        B=2,  # Bounding boxes per cell
    )

    print(f"Dataset size: {len(dataset)} images")
    print(f"Number of classes: {dataset.C}")
    print(f"Classes: {', '.join(dataset.class_names)}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Set to > 0 for parallel loading
    )

    # Iterate through a few batches
    print("\n" + "=" * 60)
    print("Loading first batch...")
    for batch_idx, (images, targets) in enumerate(dataloader):
        if batch_idx >= 1:  # Just show first batch
            break

        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")  # (batch_size, 3, 448, 448)
        print(f"  Targets shape: {targets.shape}")  # (batch_size, 7, 7, 30)

        # Show statistics for first image in batch
        print("\n  First image:")
        print(f"    Min pixel value: {images[0].min():.3f}")
        print(f"    Max pixel value: {images[0].max():.3f}")
        print(f"    Mean pixel value: {images[0].mean():.3f}")

        # Count objects in first image
        num_objects = (targets[0, :, :, 4] > 0).sum().item()
        print(f"    Number of objects: {num_objects}")

    # Visualize a sample
    print("\n" + "=" * 60)
    print("Visualizing sample...")

    sample_idx = 0
    sample_info = dataset.visualize_sample(sample_idx)

    print(f"\nSample {sample_idx}:")
    print(f"  Image: {sample_info['image_path'].name}")
    print(f"  Objects: {len(sample_info['bboxes'])}")
    for i, (bbox, class_name) in enumerate(
        zip(sample_info["bboxes"], sample_info["class_names"])
    ):
        print(f"    {i + 1}. {class_name}: {bbox}")

    # Optionally save visualization
    try:
        from PIL import Image

        image = Image.open(sample_info["image_path"])
        image_with_boxes = draw_bboxes(
            image,
            sample_info["bboxes"],
            sample_info["class_names"],
            color="green",
            width=3,
        )

        output_path = Path("sample_visualization.jpg")
        image_with_boxes.save(output_path)
        print(f"\n  Saved visualization to: {output_path}")
    except Exception as e:
        print(f"\n  Could not save visualization: {e}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
