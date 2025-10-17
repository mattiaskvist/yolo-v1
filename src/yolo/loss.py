"""YOLO v1 Loss Function Implementation."""

import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    YOLO v1 Loss Function.

    Implements the multi-part loss from the original YOLO paper:
    - Localization loss (x, y, w, h)
    - Confidence loss (objectness)
    - Classification loss
    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        """
        Args:
            S: Grid size (S x S)
            B: Number of bounding boxes per grid cell
            C: Number of classes
            lambda_coord: Weight for coordinate loss
            lambda_noobj: Weight for no-object confidence loss
        """
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Calculate YOLO loss.

        Args:
            predictions: Model predictions of shape (batch_size, S, S, B*5 + C)
            targets: Ground truth of shape (batch_size, S, S, B*5 + C)

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        batch_size = predictions.shape[0]

        # Split predictions and targets into components
        # Each bounding box has 5 values: x, y, w, h, confidence
        coord_loss = torch.tensor(0.0, device=predictions.device)
        conf_loss_obj = torch.tensor(0.0, device=predictions.device)
        conf_loss_noobj = torch.tensor(0.0, device=predictions.device)
        class_loss = torch.tensor(0.0, device=predictions.device)

        for b in range(batch_size):
            pred = predictions[b]  # (S, S, B*5 + C)
            target = targets[b]  # (S, S, B*5 + C)

            for i in range(self.S):
                for j in range(self.S):
                    # Check if object exists in this cell
                    # Target confidence is at index 4 for first box
                    obj_exists = target[i, j, 4] > 0

                    if obj_exists:
                        # Get target box (use first box in target)
                        target_box = target[i, j, :5]  # x, y, w, h, conf

                        # Find which predicted box is responsible
                        # Compare IoU of both predicted boxes with target
                        best_iou = 0
                        best_box_idx = 0

                        for box_idx in range(self.B):
                            pred_box = pred[i, j, box_idx * 5 : (box_idx + 1) * 5]
                            iou = self._calculate_iou(pred_box[:4], target_box[:4])
                            if iou > best_iou:
                                best_iou = iou
                                best_box_idx = box_idx

                        # Get the responsible box
                        pred_box = pred[i, j, best_box_idx * 5 : (best_box_idx + 1) * 5]

                        # Coordinate loss (x, y)
                        coord_loss += torch.sum((pred_box[:2] - target_box[:2]) ** 2)

                        # Size loss (sqrt of w, h)
                        pred_wh = torch.sqrt(torch.clamp(pred_box[2:4], min=1e-6))
                        target_wh = torch.sqrt(torch.clamp(target_box[2:4], min=1e-6))
                        coord_loss += torch.sum((pred_wh - target_wh) ** 2)

                        # Confidence loss for responsible box
                        conf_loss_obj += (pred_box[4] - target_box[4]) ** 2

                        # Confidence loss for other boxes in this cell
                        for box_idx in range(self.B):
                            if box_idx != best_box_idx:
                                pred_conf = pred[i, j, box_idx * 5 + 4]
                                conf_loss_noobj += pred_conf**2

                        # Classification loss
                        pred_classes = pred[i, j, self.B * 5 :]
                        target_classes = target[i, j, self.B * 5 :]
                        class_loss += torch.sum((pred_classes - target_classes) ** 2)

                    else:
                        # No object in this cell
                        # Penalize all box confidences
                        for box_idx in range(self.B):
                            pred_conf = pred[i, j, box_idx * 5 + 4]
                            conf_loss_noobj += pred_conf**2

        # Average over batch
        coord_loss = coord_loss / batch_size
        conf_loss_obj = conf_loss_obj / batch_size
        conf_loss_noobj = conf_loss_noobj / batch_size
        class_loss = class_loss / batch_size

        # Total loss with weights
        total_loss = (
            self.lambda_coord * coord_loss
            + conf_loss_obj
            + self.lambda_noobj * conf_loss_noobj
            + class_loss
        )

        # Return loss components for logging
        loss_dict = {
            "total": total_loss.item(),
            "coord": coord_loss.item(),
            "conf_obj": conf_loss_obj.item(),
            "conf_noobj": conf_loss_noobj.item(),
            "class": class_loss.item(),
        }

        return total_loss, loss_dict

    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Calculate IoU between two boxes.

        Args:
            box1: (x, y, w, h) in grid cell coordinates
            box2: (x, y, w, h) in grid cell coordinates

        Returns:
            IoU value
        """
        # Convert to corner coordinates
        box1_x1 = box1[0] - box1[2] / 2
        box1_y1 = box1[1] - box1[3] / 2
        box1_x2 = box1[0] + box1[2] / 2
        box1_y2 = box1[1] + box1[3] / 2

        box2_x1 = box2[0] - box2[2] / 2
        box2_y1 = box2[1] - box2[3] / 2
        box2_x2 = box2[0] + box2[2] / 2
        box2_y2 = box2[1] + box2[3] / 2

        # Intersection area
        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )

        # Union area
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area

        # IoU
        iou = inter_area / (union_area + 1e-6)
        return iou.item()
