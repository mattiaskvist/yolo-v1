import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from yolo.schemas import Detection, BoundingBox

# Small epsilon value to avoid division by zero in IoU calculation
EPSILON = 1e-6


class YOLOInference:
    def __init__(
        self,
        model: nn.Module,
        device: str = "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_image(self, image_path: str) -> Image.Image:
        """Load an image from the given path."""
        image = Image.open(image_path).convert("RGB")
        return image

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess the image for model input."""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        return img_tensor

    def predict(
        self,
        image_path: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        class_names: list[str] | None = None,
    ) -> list[Detection]:
        """
        Predict objects in an image.

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold (0-1)
            nms_threshold: Non-maximum suppression threshold (0-1)
            class_names: Optional list of class names for labels

        Returns:
            List of Detection objects

        Example:
            >>> detections = inference.predict("image.jpg", class_names=VOC_CLASSES)
            >>> for det in detections:
            ...     print(f"{det.class_name}: {det.confidence:.2f}")
        """
        # Load and preprocess image
        image = self.load_image(image_path)
        img_tensor = self.preprocess_image(image)

        # Forward pass
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Parse predictions
        detections = self.parse_predictions(predictions[0], conf_threshold, class_names)

        # Apply NMS
        detections = self.non_max_suppression(detections, nms_threshold)

        return detections

    def parse_predictions(
        self,
        pred: torch.Tensor,
        conf_threshold: float,
        class_names: list[str] | None = None,
    ) -> list[Detection]:
        """Parse YOLO output tensor into Detection objects.

        Args:
            pred: Prediction tensor of shape (S, S, B*5 + C)
            conf_threshold: Confidence threshold
            class_names: Optional list of class names

        Returns:
            List of Detection objects
        """
        S = self.model.S
        B = self.model.B
        detections: list[Detection] = []

        for i in range(S):
            for j in range(S):
                cell_pred = pred[i, j]
                class_probs = cell_pred[B * 5 :]

                for b in range(B):
                    box_offset = b * 5
                    x, y, w, h, conf = cell_pred[box_offset : box_offset + 5]

                    # Convert to absolute coordinates
                    x = (j + x.item()) / S
                    y = (i + y.item()) / S
                    w = w.item()
                    h = h.item()
                    conf = conf.item()

                    class_id = torch.argmax(class_probs).item()
                    class_prob = class_probs[class_id].item()

                    final_conf = conf * class_prob

                    if final_conf > conf_threshold:
                        class_name = (
                            class_names[class_id]
                            if class_names
                            else f"class_{class_id}"
                        )
                        detections.append(
                            Detection(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=final_conf,
                                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                            )
                        )

        return detections

    def iou(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union between two bounding boxes.

        Args:
            bbox1: First BoundingBox
            bbox2: Second BoundingBox

        Returns:
            IoU value (0-1)
        """
        # Get corner coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1.to_corners()
        x2_min, y2_min, x2_max, y2_max = bbox2.to_corners()

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(
            0, inter_y_max - inter_y_min
        )

        # Calculate union
        box1_area = bbox1.area
        box2_area = bbox2.area

        # Add EPSILON for numerical stability to avoid division by zero
        iou = inter_area / (box1_area + box2_area - inter_area + EPSILON)
        return iou

    def non_max_suppression(
        self, detections: list[Detection], nms_threshold: float = None, iou_threshold: float = None
    ) -> list[Detection]:
        """Apply non-maximum suppression to remove overlapping boxes.

        Args:
            detections: List of Detection objects
            nms_threshold: IoU threshold for suppression (preferred)
            iou_threshold: Deprecated, use nms_threshold instead

        Returns:
            Filtered list of Detection objects
        """
        # Support both nms_threshold and iou_threshold for backward compatibility
        if iou_threshold is not None:
            import warnings
            warnings.warn(
                "Parameter 'iou_threshold' is deprecated, use 'nms_threshold' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            threshold = iou_threshold
        elif nms_threshold is not None:
            threshold = nms_threshold
        else:
            threshold = 0.4  # Default value if not provided

        if len(detections) == 0:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        keep = []

        while detections:
            current = detections.pop(0)
            keep.append(current)

            # Remove boxes that overlap significantly with current box (same class)
            detections = [
                det
                for det in detections
                if det.class_id != current.class_id
                or self.iou(current.bbox, det.bbox) < threshold
            ]

        return keep


# Example usage
if __name__ == "__main__":
    from yolo.models import YOLOv1, ResNetBackbone

    checkpoint_path = "checkpoints/yolo_best.pth"
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = YOLOv1(
        num_classes=20, backbone=ResNetBackbone(pretrained=True, freeze=True)
    )
    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device),
    )

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    # Create inference engine
    inference = YOLOInference(model, device=device)

    # Run prediction on sample image
    detections = inference.predict(
        "notebooks/sample.jpg",
        conf_threshold=0.4,
        nms_threshold=0.4,
    )
    print(f"Found {len(detections)} objects")
    for det in detections:
        print(f"Class: {det[0]}, Confidence: {det[1]:.2f}, Box: {det[2:]}")
