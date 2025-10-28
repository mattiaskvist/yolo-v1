"""
Demo script showing how to load a trained model and make predictions.

This is a simple interactive example that demonstrates:
1. Loading a model checkpoint
2. Running inference on test images
3. Visualizing results

Example usage:
    python examples/inference_demo.py --checkpoint checkpoints/yolo_best.pth
"""

import sys
from pathlib import Path
import argparse

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

    # Find test images
    print("\n" + "=" * 70)
    print("Looking for test images...")
    print("=" * 70)

    test_images_dir = Path(
        "~/.cache/kagglehub/datasets/zaraks/pascal-voc-2007/versions/1/VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages"
    ).expanduser()
    if not test_images_dir.exists():
        print("⚠️  VOC test images not found at expected location")
        print(f"   Looked in: {test_images_dir}")
        print("\nPlease specify an image path manually:")
        print(
            "  python predict.py --checkpoint checkpoints/your_model.pth --image path/to/image.jpg"
        )
        return

    # Get a few test images
    test_images = list(test_images_dir.glob("*.jpg"))[:50]

    if not test_images:
        print("⚠️  No JPEG images found in the test directory")
        return

    print(f"\nFound {len(test_images)} test images")
    print()

    # Run predictions
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("Running predictions...")
    print("=" * 70)

    for i, img_path in enumerate(test_images, 1):
        output_path = output_dir / f"pred_{img_path.name}"

        print(f"\n[{i}/{len(test_images)}]")
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
