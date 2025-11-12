"""YOLO v1 Loss Function Implementation."""

import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """YOLO v1 loss function implementation.

    Implements the multi-part loss function from the original YOLO paper:
    1. Localization loss: Penalizes errors in bounding box coordinates (x, y, w, h)
    2. Confidence loss: Penalizes errors in objectness predictions
    3. Classification loss: Penalizes errors in class probability predictions

    The loss uses different weights for different components:
    - lambda_coord: Increases importance of box coordinate predictions
    - lambda_noobj: Decreases importance of confidence predictions for cells without objects

    Attributes:
        S: Grid size (typically 7).
        B: Number of bounding boxes per cell (typically 2).
        C: Number of classes (typically 20 for PASCAL VOC).
        lambda_coord: Weight for coordinate loss (default 5.0).
        lambda_noobj: Weight for no-object confidence loss (default 0.5).

    """

    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    ):
        """Initialize YOLO loss function.

        Args:
            S: Grid size for spatial divisions (default 7 for 7x7 grid).
            B: Number of bounding boxes predicted per grid cell (default 2).
            C: Number of object classes (default 20 for PASCAL VOC).
            lambda_coord: Weight multiplier for coordinate loss to increase
                importance of box localization (default 5.0 from paper).
            lambda_noobj: Weight multiplier for no-object confidence loss to
                decrease importance of background predictions (default 0.5 from paper).

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
        """Computes the YOLO v1 loss.

        Args:
            predictions (torch.Tensor): Predicted tensor of shape (N, S, S, B*5 + C), where
                N is the batch size,
                S is the grid size,
                B is the number of bounding boxes per grid cell,
                C is the number of classes.
                The last dimension contains, for each box, (x, y, w, h, confidence) and class probabilities.
            targets (torch.Tensor): Ground truth tensor of the same shape as predictions.

        Returns:
            tuple: (total_loss, loss_dict)
                total_loss (torch.Tensor): The total loss value (scalar).
                loss_dict (dict): Dictionary with individual loss components:
                    - "total": total loss (float)
                    - "coord": coordinate loss (float)
                    - "conf_obj": confidence loss for cells containing objects (float)
                    - "conf_noobj": confidence loss for cells not containing objects (float)
                    - "class": classification loss (float)

        Methodology:
            The loss consists of:
                - Coordinate loss for bounding box center and size (only for responsible boxes).
                - Confidence loss for object and no-object cells.
                - Classification loss for cells containing objects.
            The responsible bounding box for each object is the one with the highest IoU with the ground truth.

        """
        N = predictions.size(0)
        device = predictions.device

        # Split predictions and targets
        pred_boxes = predictions[..., : self.B * 5].view(N, self.S, self.S, self.B, 5)
        pred_cls = predictions[..., self.B * 5 :]  # (N, S, S, C)

        target_boxes = targets[..., : self.B * 5].view(N, self.S, self.S, self.B, 5)
        target_cls = targets[..., self.B * 5 :]  # (N, S, S, C)

        # Determine which cells contain objects (if any confidence slot > 0)
        target_conf_mask = targets[..., 4::5] > 0  # (N, S, S, B)
        obj_mask = target_conf_mask.any(dim=-1)  # (N, S, S)

        # For each cell, select the target box that actually contains an object
        target_box_idx = target_conf_mask.float().argmax(dim=-1)  # (N, S, S)
        idx = target_box_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 4)
        target_box = target_boxes[..., :4].gather(3, idx).squeeze(3)  # (N, S, S, 4)

        # Compute IoUs between each predicted box and selected target box
        ious = self.compute_iou(
            pred_boxes[..., :4], target_box.unsqueeze(3)
        )  # (N, S, S, B)
        best_box = ious.argmax(dim=3)  # (N, S, S)
        best_ious = ious.gather(3, best_box.unsqueeze(-1)).squeeze(-1)  # (N, S, S)

        # Mask for responsible predicted boxes
        box_mask = torch.zeros_like(ious, dtype=torch.bool, device=device)
        box_mask.scatter_(3, best_box.unsqueeze(-1), True)  # (N,S,S,B)

        # Only select responsible boxes where there are actual objects
        responsible_mask = box_mask & obj_mask.unsqueeze(-1)  # (N,S,S,B)

        # Select responsible predicted boxes and their corresponding target boxes
        responsible_boxes = pred_boxes[responsible_mask]  # (num_obj, 5)
        target_boxes_resp = target_box[obj_mask]  # (num_obj, 4)
        target_conf = best_ious[obj_mask]  # (num_obj,)

        # === Coordinate loss ===
        if responsible_boxes.numel() > 0:  # Check if there are any objects
            xy_loss = torch.sum(
                (responsible_boxes[:, :2] - target_boxes_resp[:, :2]) ** 2
            )
            wh_loss = torch.sum(
                (
                    torch.sqrt(torch.clamp(responsible_boxes[:, 2:4], min=1e-6))
                    - torch.sqrt(torch.clamp(target_boxes_resp[:, 2:4], min=1e-6))
                )
                ** 2
            )
            coord_loss = self.lambda_coord * (xy_loss + wh_loss)
        else:
            coord_loss = torch.tensor(0.0, device=device)

        # === Confidence loss ===
        if responsible_boxes.numel() > 0:  # Check if there are any objects
            pred_conf_obj = responsible_boxes[:, 4]
            conf_loss_obj = torch.sum((pred_conf_obj - target_conf) ** 2)
        else:
            conf_loss_obj = torch.tensor(0.0, device=device)

        # No-object confidence loss: penalize all boxes that are NOT responsible
        # This includes boxes in cells without objects AND non-responsible boxes in cells with objects
        noobj_mask = ~responsible_mask  # All boxes that are NOT responsible
        pred_conf_noobj = pred_boxes[..., 4][noobj_mask]
        conf_loss_noobj = torch.sum(pred_conf_noobj**2)
        conf_loss_noobj = self.lambda_noobj * conf_loss_noobj

        # === Classification loss ===
        if obj_mask.any():  # Check if there are any objects
            class_loss = torch.sum((pred_cls[obj_mask] - target_cls[obj_mask]) ** 2)
        else:
            class_loss = torch.tensor(0.0, device=device)

        # === Total ===
        total_loss = (coord_loss + conf_loss_obj + conf_loss_noobj + class_loss) / N

        loss_dict = {
            "total": total_loss.detach().item(),
            "coord": (coord_loss / N).detach().item(),
            "conf_obj": (conf_loss_obj / N).detach().item(),
            "conf_noobj": (conf_loss_noobj / N).detach().item(),
            "class": (class_loss / N).detach().item(),
        }

        return total_loss, loss_dict

    @staticmethod
    def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute Intersection over Union (IoU) between two sets of boxes.

        Calculates IoU for determining which predicted box is responsible for
        each ground truth object (highest IoU wins).

        Args:
            boxes1: Predicted boxes of shape (N, S, S, B, 4) where last dim is
                (x_center, y_center, width, height) in normalized coordinates.
            boxes2: Ground truth boxes of shape (N, S, S, 1, 4) in same format.

        Returns:
            IoU values of shape (N, S, S, B), one IoU score for each predicted box
            against the ground truth box in that cell.

        """
        box1_x1 = boxes1[..., 0] - boxes1[..., 2] / 2
        box1_y1 = boxes1[..., 1] - boxes1[..., 3] / 2
        box1_x2 = boxes1[..., 0] + boxes1[..., 2] / 2
        box1_y2 = boxes1[..., 1] + boxes1[..., 3] / 2

        box2_x1 = boxes2[..., 0] - boxes2[..., 2] / 2
        box2_y1 = boxes2[..., 1] - boxes2[..., 3] / 2
        box2_x2 = boxes2[..., 0] + boxes2[..., 2] / 2
        box2_y2 = boxes2[..., 1] + boxes2[..., 3] / 2

        inter_x1 = torch.maximum(box1_x1, box2_x1)
        inter_y1 = torch.maximum(box1_y1, box2_y1)
        inter_x2 = torch.minimum(box1_x2, box2_x2)
        inter_y2 = torch.minimum(box1_y2, box2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
            inter_y2 - inter_y1, min=0
        )
        box1_area = boxes1[..., 2] * boxes1[..., 3]
        box2_area = boxes2[..., 2] * boxes2[..., 3]
        union = box1_area + box2_area - inter_area
        return inter_area / (union + 1e-6)
