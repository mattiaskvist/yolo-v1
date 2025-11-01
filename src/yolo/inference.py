import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Small epsilon value to avoid division by zero in IoU calculation
EPSILON = 1e-6
Detection = tuple[int, float, float, float, float, float]
Box = tuple[float, float, float, float]


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
        self, image_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> list[Detection]:
        """
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold
            nms_threshold: Non-maximum suppression threshold
        Returns:
            List of detections: [(class_id, confidence, x, y, w, h), ...]
        """
        # Load and preprocess image
        image = self.load_image(image_path)
        img_tensor = self.preprocess_image(image)

        # Forward pass
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Parse predictions
        detections = self.parse_predictions(predictions[0], conf_threshold)

        # Apply NMS
        detections = self.non_max_suppression(detections, nms_threshold)

        return detections

    def parse_predictions(
        self,
        pred: torch.Tensor,
        conf_threshold: float,
    ) -> list[Detection]:
        """Parse YOLO output tensor into bounding boxes
        Args:
            pred: Prediction tensor of shape (S, S, B*5 + C)
            conf_threshold: Confidence threshold
        Returns:
            List of detections: [(class_id, confidence, x, y, w, h), ...]
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
                        detections.append([class_id, final_conf, x, y, w, h])

        return detections

    def iou(
        self,
        box1: Box,
        box2: Box,
    ) -> float:
        """Calculate Intersection over Union
        Args:
            box1: (x, y, w, h) for box 1
            box2: (x, y, w, h) for box 2
        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(
            0, inter_y_max - inter_y_min
        )
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Add EPSILON for numerical stability to avoid division by zero
        iou = inter_area / (box1_area + box2_area - inter_area + EPSILON)
        return iou

    def non_max_suppression(
        self,
        detections: list[Detection],
        nms_threshold: float,
    ) -> list[Detection]:
        """Apply non-maximum suppression (remove overlapping boxes).
        Args:
            detections: List of detections [(class_id, confidence, x, y, w, h), ...]
            nms_threshold: IoU threshold for NMS
        Returns:
            Filtered list of detections after NMS
        """
        if len(detections) == 0:
            return []

        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        keep = []

        while detections:
            current = detections.pop(0)
            keep.append(current)

            detections = [
                det
                for det in detections
                if det[0] != current[0]
                or self.iou(current[2:], det[2:]) < nms_threshold
            ]

        return keep


# Example usage
if __name__ == "__main__":
    from yolo.models import YOLOv1, YOLOv1ResNet  # noqa: F401

    # model = YOLOv1(num_classes=20)
    model = YOLOv1ResNet(num_classes=20, freeze_backbone=True)

    inference = YOLOInference(model)

    # Replace 'path/to/image.jpg' with the actual path to your test image
    detections = inference.predict(
        "path/to/image.jpg",
        conf_threshold=0.5,  # typical confidence threshold
        nms_threshold=0.4,  # standard nms threshold for object detection
    )
    print(f"Found {len(detections)} objects")
    for det in detections:
        print(f"Class: {det[0]}, Confidence: {det[1]:.2f}, Box: {det[2:]}")
