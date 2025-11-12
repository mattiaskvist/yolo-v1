"""
Load a trained YOLO model and make predictions on images.

This script demonstrates how to:
1. Load a trained model from checkpoint
2. Run inference on single or multiple images
3. Visualize predictions with bounding boxes
4. Save results
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from yolo import YOLOv1, ResNetBackbone
from yolo.inference import YOLOInference
from yolo.utils.visualization import draw_detections, VOC_CLASSES


def load_model(
    checkpoint_path: str,
    num_classes: int = 20,
    freeze_backbone: bool = False,
    device: str = "cuda",
) -> YOLOv1:
    """
    Load a trained YOLO model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        num_classes: Number of object classes
        freeze_backbone: Whether backbone was frozen during training
        device: Device to load model on

    Returns:
        Loaded YOLO model
    """
    print(f"Loading model from: {checkpoint_path}")

    # Create model with same architecture as training
    backbone = ResNetBackbone(pretrained=False, freeze=freeze_backbone)
    model = YOLOv1(backbone=backbone, num_classes=num_classes, S=7, B=2)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Print checkpoint info
    if "epoch" in checkpoint:
        print(f"  Loaded from epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")

    print("  Model loaded successfully!")

    return model


def predict_single_image(
    model: YOLOv1,
    image_path: str,
    output_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    class_names: list[str] = VOC_CLASSES,
    device: str = "cuda",
) -> list:
    """
    Run inference on a single image.

    Args:
        model: Trained YOLO model
        image_path: Path to input image
        output_path: Optional path to save visualization
        conf_threshold: Confidence threshold for detections
        nms_threshold: NMS threshold
        class_names: List of class names
        device: Device to run on

    Returns:
        List of detections
    """
    # Create inference engine
    inference = YOLOInference(model, device=device)

    # Run prediction
    print(f"\nProcessing: {image_path}")
    detections = inference.predict(
        image_path,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        class_names=class_names,
    )

    # Print results
    print(f"Found {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        print(
            f"  {i}. {det.class_name}: {det.confidence:.2f} at "
            f"({det.bbox.x:.2f}, {det.bbox.y:.2f}) size ({det.bbox.width:.2f}, {det.bbox.height:.2f})"
        )

    # Visualize if requested
    if output_path or len(detections) > 0:
        image = Image.open(image_path).convert("RGB")
        img_with_boxes = draw_detections(image, detections, class_names, conf_threshold)

        if output_path:
            img_with_boxes.save(output_path)
            print(f"Saved visualization to: {output_path}")
        else:
            # Show the image
            try:
                img_with_boxes.show()
            except Exception:
                print("Could not display image (no display available)")

    return detections


def predict_multiple_images(
    model: YOLOv1,
    image_dir: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    class_names: list[str] = VOC_CLASSES,
    device: str = "cuda",
):
    """
    Run inference on multiple images in a directory.

    Args:
        model: Trained YOLO model
        image_dir: Directory containing images
        output_dir: Directory to save visualizations
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold
        class_names: List of class names
        device: Device to run on
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Get all image files
    image_dir_path = Path(image_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [
        f for f in image_dir_path.iterdir() if f.suffix.lower() in image_extensions
    ]

    print(f"\nFound {len(image_files)} images in {image_dir}")

    # Process each image
    total_detections = 0
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")

        output_file = output_path / f"{image_file.stem}_pred{image_file.suffix}"

        detections = predict_single_image(
            model=model,
            image_path=str(image_file),
            output_path=str(output_file),
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            class_names=class_names,
            device=device,
        )

        total_detections += len(detections)

    print(f"\n{'=' * 70}")
    print(f"Processed {len(image_files)} images")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(image_files):.2f}")
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained YOLO model"
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)",
    )
    parser.add_argument(
        "--num-classes", type=int, default=20, help="Number of classes (default: 20)"
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Use if model was trained with frozen backbone",
    )

    # Input/Output
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument(
        "--image-dir", type=str, default=None, help="Directory of images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Output directory for visualizations",
    )

    # Inference parameters
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.4,
        help="NMS threshold (default: 0.4)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: auto-detect)",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.image is None and args.image_dir is None:
        parser.error("Must specify either --image or --image-dir")

    # Load model
    print("=" * 70)
    print("YOLO v1 Inference")
    print("=" * 70)

    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        freeze_backbone=args.freeze_backbone,
        device=args.device,
    )

    # Run inference
    if args.image:
        # Single image
        output_path = Path(args.output) / f"prediction_{Path(args.image).name}"
        output_path.parent.mkdir(exist_ok=True, parents=True)

        predict_single_image(
            model=model,
            image_path=args.image,
            output_path=str(output_path),
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            device=args.device,
        )

    else:
        # Multiple images
        predict_multiple_images(
            model=model,
            image_dir=args.image_dir,
            output_dir=args.output,
            conf_threshold=args.conf_threshold,
            nms_threshold=args.nms_threshold,
            device=args.device,
        )

    print("\n" + "=" * 70)
    print("Inference completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
