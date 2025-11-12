"""
Visualization utilities for YOLO predictions.
"""

import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import torch

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def draw_detections(
    image: Image.Image,
    detections: list,
    class_names: list[str] | None = None,
    conf_threshold: float = 0.5,
    box_width: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: PIL Image
        detections: List of Detection objects or tuples [(class_id, conf, x, y, w, h), ...]
        class_names: List of class names (optional if Detection objects have class_name)
        conf_threshold: Only draw boxes with confidence above this
        box_width: Width of bounding box lines
        font_size: Size of text labels

    Returns:
        Image with drawn boxes
    """
    from yolo.schemas import Detection

    # Create a copy to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Load font
    font = _load_font(font_size)

    # Image dimensions
    img_width, img_height = image.size

    # Colors for different classes
    colors = [
        "red",
        "green",
        "blue",
        "magenta",
        "cyan",
        "orange",
        "purple",
        "pink",
        "lime",
    ]

    # Draw each detection
    for det in detections:
        # Handle both Detection objects and legacy tuples
        if isinstance(det, Detection):
            class_id = det.class_id
            conf = det.confidence
            class_name = det.class_name

            # Skip low confidence detections
            if conf < conf_threshold:
                continue

            # Get pixel coordinates directly
            x1, y1, x2, y2 = det.bbox.to_pixel_coords(img_width, img_height)
        else:
            # Legacy tuple format
            class_id, conf, x, y, w, h = det

            # Skip low confidence detections
            if conf < conf_threshold:
                continue

            # Convert from normalized coordinates to pixel coordinates
            x_center = x * img_width
            y_center = y * img_height
            box_width_px = w * img_width
            box_height_px = h * img_height

            # Convert to corner coordinates
            x1 = int(x_center - box_width_px / 2)
            y1 = int(y_center - box_height_px / 2)
            x2 = int(x_center + box_width_px / 2)
            y2 = int(y_center + box_height_px / 2)

            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"Class {class_id}"

        # Ensure x1 <= x2 and y1 <= y2 (handle negative widths/heights)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Clamp coordinates to image bounds
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        # Skip if box is too small or invalid
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        # Choose color based on class
        color = colors[class_id % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)

        # Draw label
        label = f"{class_name}: {conf:.2f}"

        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 25), label, fill="white", font=font)

    return img_draw


def _load_font(size: int = 20) -> ImageFont.ImageFont:
    """
    Load a cross-platform font, fall back to default if not available.

    Args:
        size: Font size

    Returns:
        ImageFont object
    """
    try:
        system = platform.system()
        font = None

        if system == "Darwin":
            candidates = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/Library/Fonts/Arial.ttf",
            ]
        elif system == "Windows":
            candidates = [
                r"C:\Windows\Fonts\arial.ttf",
                r"C:\Windows\Fonts\segoeui.ttf",
            ]
        else:  # Linux / other
            candidates = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            ]

        for p in candidates:
            try:
                if Path(p).exists():
                    font = ImageFont.truetype(p, size)
                    break
            except Exception:
                continue

        # Try a generic name (may work if font is in font path)
        if font is None:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", size)
            except Exception:
                font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    return font


def extract_objectness_scores(
    predictions: torch.Tensor,
    S: int = 7,
    B: int = 2,
) -> torch.Tensor:
    """
    Extract maximum objectness/confidence scores from YOLO predictions.

    Args:
        predictions: Tensor of shape (batch, S, S, B*5 + num_classes) or (S, S, B*5 + num_classes)
        S: Grid size (default 7 for YOLOv1)
        B: Number of bounding boxes per cell (default 2)

    Returns:
        Tensor of shape (S, S) containing maximum confidence score for each grid cell
    """
    # Ensure predictions is 3D: [S, S, B*5 + num_classes]
    if predictions.dim() == 4:
        predictions = predictions.squeeze(0)

    # Extract confidence scores for each bounding box
    # Each cell has B boxes, each with format [x, y, w, h, confidence]
    conf_scores = []
    for b in range(B):
        conf_idx = b * 5 + 4  # Confidence is at index 4, 9, 14, etc.
        conf_scores.append(predictions[:, :, conf_idx])

    # Stack and take maximum confidence across all boxes
    conf_tensor = torch.stack(conf_scores, dim=0)  # [B, S, S]
    max_conf = torch.max(conf_tensor, dim=0)[0]  # [S, S]

    return max_conf


def visualize_objectness_grid(
    image: Image.Image,
    predictions: torch.Tensor,
    B: int = 2,
    img_size: int = 448,
) -> torch.Tensor:
    """Visualize the objectness score grid for a YOLO model.

    Args:
        predictions: Tensor of shape (B, S*S, 5+C) containing the model predictions
        grid_size: Size of the grid (S)
        img_size: Size of the input image (width/height)

    Returns:
        None
    """

    # Extract maximum objectness scores
    max_conf = (
        extract_objectness_scores(predictions, S=predictions.shape[1], B=B)
        .cpu()
        .numpy()
    )

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    im = axes[1].imshow(max_conf, cmap="hot", interpolation="nearest")
    axes[1].set_title("Objectness Heatmap")
    axes[1].set_xlabel("Grid X")
    axes[1].set_ylabel("Grid Y")
    plt.colorbar(im, ax=axes[1])

    # Overlay on image
    image_resized = image.resize((img_size, img_size))
    axes[2].imshow(image_resized, alpha=0.6)
    im2 = axes[2].imshow(
        max_conf,
        cmap="hot",
        alpha=0.4,
        extent=[0, img_size, img_size, 0],
        interpolation="bilinear",
    )
    axes[2].set_title("Objectness Overlay")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    return max_conf


def draw_objectness_grid_on_image(
    predictions, image, S=7, B=2, img_size=448
) -> Image.Image:
    """Draw grid with objectness scores as text on the image.

    Args:
        predictions: Tensor of shape (S, S, B*5 + num_classes)
        image: PIL Image
        S: Grid size (default 7 for YOLOv1)
        B: Number of bounding boxes per cell (default 2)
        img_size: Size to resize image for drawing grid (default 448)
    Returns:
        PIL Image with grid and objectness scores drawn on it.
    """
    # Extract maximum objectness scores
    max_conf = extract_objectness_scores(predictions, S=S, B=B).cpu().numpy()

    # Resize image to img_size x img_size
    img_draw = image.resize((img_size, img_size))
    draw = ImageDraw.Draw(img_draw)

    cell_size = img_size // S

    # Draw grid and scores
    for i in range(S):
        for j in range(S):
            x = j * cell_size
            y = i * cell_size

            # Draw grid cell
            draw.rectangle(
                [x, y, x + cell_size, y + cell_size], outline="white", width=1
            )

            # Draw confidence score
            conf = max_conf[i, j]
            text = f"{conf:.2f}"

            # Color code text based on confidence
            color = "red" if conf > 0.5 else "yellow" if conf > 0.2 else "white"
            draw.text((x + 5, y + 5), text, fill=color)

    return img_draw
