"""
Demo script showing how to load a trained model and make predictions.

This is a simple interactive example that demonstrates:
1. Loading a model checkpoint
2. Running inference on test images
3. Visualizing results

Example usage:
    python examples/inference_demo.py --checkpoint checkpoints/yolo_best.pth
    python examples/inference_demo.py --checkpoint checkpoints/yolo_best.pth --images-dir /path/to/images
    IMAGES_DIR=/path/to/images python examples/inference_demo.py --checkpoint checkpoints/yolo_best.pth
"""

import sys
from pathlib import Path
import argparse
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict import load_model, predict_single_image
import torch


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="YOLO v1 Inference Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/yolo_best.pth",
        help="Path to model checkpoint (default: checkpoints/yolo_best.pth)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to directory containing images for inference (default: uses IMAGES_DIR env var or Kaggle VOC test cache)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    print("=" * 70)
    print("YOLO v1 Inference Demo")
    print("=" * 70)
    print()

    # Verify checkpoint exists
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for ckpt in sorted(checkpoints_dir.glob("*.pth")):
                print(f"  - {ckpt}")
        print("\nPlease specify a valid checkpoint:")
        print(
            "  python examples/inference_demo.py --checkpoint checkpoints/your_model.pth"
        )
        return

    print(f"📦 Using checkpoint: {checkpoint_path.name}")
    print()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    print()

    # Load model
    # Try to load freeze_backbone from checkpoint, with fallback
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    freeze_backbone = checkpoint_data.get("freeze_backbone", False)
    print(f"ℹ️  freeze_backbone from checkpoint: {freeze_backbone}")

    model = load_model(
        checkpoint_path=str(checkpoint_path),
        num_classes=20,
        freeze_backbone=freeze_backbone,
        device=device,
    )

    # Find images
    print("\n" + "=" * 70)
    print("Looking for images...")
    print("=" * 70)

    # Determine images directory from CLI arg, env var, or default
    if args.images_dir:
        images_dir = Path(args.images_dir).expanduser()
    elif os.getenv("IMAGES_DIR"):
        images_dir = Path(os.getenv("IMAGES_DIR")).expanduser()
    else:
        # Default to Kaggle VOC test cache location
        images_dir = Path(
            "~/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
        ).expanduser()

    if not images_dir.exists():
        print("⚠️  Images directory not found at expected location")
        print(f"   Looked in: {images_dir}")
        print("\nPlease specify images directory via:")
        print("  1. CLI argument: --images-dir /path/to/images")
        print("  2. Environment variable: IMAGES_DIR=/path/to/images")
        print("\nOr use predict.py for single images:")
        print(
            "  python predict.py --checkpoint checkpoints/your_model.pth --image path/to/image.jpg"
        )
        return

    # Get images to process
    images = list(images_dir.glob("*.jpg"))[:50]

    if not images:
        print("⚠️  No JPEG images found in the directory")
        return

    print(f"\nFound {len(images)} images")
    print()

    # Run predictions
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Running predictions...")
    print("=" * 70)

    for i, img_path in enumerate(images, 1):
        output_path = output_dir / f"pred_{img_path.name}"

        print(f"\n[{i}/{len(images)}]")
        predict_single_image(
            model=model,
            image_path=str(img_path),
            output_path=str(output_path),
            conf_threshold=0.05,  # Lower threshold to see more detections
            nms_threshold=0.4,
            device=device,
        )

    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)
    print(f"\nPredictions saved to: {output_dir}/")
    print("\nTo run inference on your own images:")
    print(f"  python predict.py --checkpoint {checkpoint_path} --image your_image.jpg")
    print("\nTo process a directory of images:")
    print(f"  python predict.py --checkpoint {checkpoint_path} --image-dir your_dir/")
    print()


if __name__ == "__main__":
    main()
