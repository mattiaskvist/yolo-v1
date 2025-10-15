"""Utility functions for data processing and visualization."""

from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(
    image: Image.Image,
    bboxes: List[List[float]],
    class_names: List[str],
    confidences: List[float] = None,
    color: str = "red",
    width: int = 2,
) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: PIL Image
        bboxes: List of [x_center, y_center, width, height] normalized to [0, 1]
        class_names: List of class names for each box
        confidences: Optional list of confidence scores
        color: Box color
        width: Line width

    Returns:
        Image with drawn bounding boxes
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    img_width, img_height = image.size

    # Try to use a better font with cross-platform fallbacks
    font = ImageFont.load_default()
    font_names = [
        "DejaVuSans.ttf",  # Linux
        "Arial.ttf",  # Windows
        "Helvetica.ttc",  # macOS (without full path)
    ]
    for font_name in font_names:
        try:
            font = ImageFont.truetype(font_name, 16)
            break
        except (OSError, IOError):
            continue

    for idx, (bbox, class_name) in enumerate(zip(bboxes, class_names)):
        x_center, y_center, box_width, box_height = bbox

        # Convert to pixel coordinates
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        box_width_px = box_width * img_width
        box_height_px = box_height * img_height

        # Calculate box corners
        x1 = x_center_px - box_width_px / 2
        y1 = y_center_px - box_height_px / 2
        x2 = x_center_px + box_width_px / 2
        y2 = y_center_px + box_height_px / 2

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        # Draw label
        if confidences:
            label = f"{class_name}: {confidences[idx]:.2f}"
        else:
            label = class_name

        # Draw label background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1), label, fill="white", font=font)

    return img_copy


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU (Intersection over Union) between two bounding boxes.

    Args:
        box1: [x_center, y_center, width, height]
        box2: [x_center, y_center, width, height]

    Returns:
        IoU score
    """
    # Convert to corner coordinates
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Compute union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def non_max_suppression(
    bboxes: List[List[float]],
    class_ids: List[int],
    confidences: List[float],
    iou_threshold: float = 0.5,
) -> Tuple[List[List[float]], List[int], List[float]]:
    """
    Apply Non-Maximum Suppression to remove overlapping boxes.

    Args:
        bboxes: List of bounding boxes
        class_ids: List of class IDs
        confidences: List of confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered (bboxes, class_ids, confidences)
    """
    if len(bboxes) == 0:
        return [], [], []

    # Sort by confidence (descending)
    indices = sorted(
        range(len(confidences)), key=lambda i: confidences[i], reverse=True
    )

    keep = []
    while indices:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]

        # Remove boxes with high IoU
        new_indices = []
        for idx in indices:
            if class_ids[current] != class_ids[idx]:
                # Different classes, keep both
                new_indices.append(idx)
            else:
                iou = compute_iou(bboxes[current], bboxes[idx])
                if iou <= iou_threshold:
                    new_indices.append(idx)

        indices = new_indices

    # Return kept boxes
    return (
        [bboxes[i] for i in keep],
        [class_ids[i] for i in keep],
        [confidences[i] for i in keep],
    )
