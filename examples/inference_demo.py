"""
Demo script showing how to load a trained model and make predictions.

This is a simple interactive example that demonstrates:
1. Loading a model checkpoint
2. Running inference on test images
3. Visualizing results

Example usage:
    python examples/inference_demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from predict import load_model, predict_single_image
import torch


def main():
    print("=" * 70)
    print("YOLO v1 Inference Demo")
    print("=" * 70)
    print()

    # Check for available checkpoint
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("⚠️  No checkpoints directory found.")
        print("\nTo use this demo, you need to train a model first:")
        print("  python train.py --freeze-backbone --epochs 50")
        print("\nOr download a pretrained checkpoint and place it in checkpoints/")
        return

    # Find latest checkpoint
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        print("⚠️  No checkpoint files (.pth) found in checkpoints/")
        print("\nTo use this demo, you need to train a model first:")
        print("  python train.py --freeze-backbone --epochs 50")
        return

    # Use the most recent checkpoint
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"📦 Using checkpoint: {latest_checkpoint.name}")
    print()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    print()

    # Load model
    model = load_model(
        checkpoint_path=str(latest_checkpoint),
        num_classes=20,
        freeze_backbone=False,  # Adjust if needed
        device=device,
    )

    # Find test images
    print("\n" + "=" * 70)
    print("Looking for test images...")
    print("=" * 70)

    test_images_dir = Path("data/VOCdevkit/VOC2007/JPEGImages")
    if not test_images_dir.exists():
        print("⚠️  VOC test images not found at expected location")
        print("\nPlease specify an image path manually:")
        print(
            "  python predict.py --checkpoint checkpoints/your_model.pth --image path/to/image.jpg"
        )
        return

    # Get a few test images
    test_images = list(test_images_dir.glob("*.jpg"))[:5]

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
            conf_threshold=0.3,  # Lower threshold to see more detections
            nms_threshold=0.4,
            device=device,
        )

    print("\n" + "=" * 70)
    print("✅ Demo completed!")
    print("=" * 70)
    print(f"\nPredictions saved to: {output_dir}/")
    print("\nTo run inference on your own images:")
    print(
        f"  python predict.py --checkpoint {latest_checkpoint} --image your_image.jpg"
    )
    print("\nTo process a directory of images:")
    print(f"  python predict.py --checkpoint {latest_checkpoint} --image-dir your_dir/")
    print()


if __name__ == "__main__":
    main()
