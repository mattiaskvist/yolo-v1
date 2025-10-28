"""
Visualization utilities for YOLO predictions.
"""

import platform
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

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

def draw_predictions(
    image: Image.Image,
    detections: list,
    class_names: list[str],
    conf_threshold: float = 0.5,
    box_width: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """
    Draw bounding boxes and labels on image.

    Args:
        image: PIL Image
        detections: List of detections [(class_id, conf, x, y, w, h), ...]
        class_names: List of class names
        conf_threshold: Only draw boxes with confidence above this
        box_width: Width of bounding box lines
        font_size: Size of text labels

    Returns:
        Image with drawn boxes
    """
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
        label = f"{class_names[class_id]}: {conf:.2f}"

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
