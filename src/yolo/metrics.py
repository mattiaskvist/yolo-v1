"""
Mean Average Precision (mAP) metrics for object detection evaluation.

This module implements mAP calculation following the PASCAL VOC evaluation protocol.
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

EPSILON = 1e-6  # Small value for numerical stability


class mAPMetric:
    """
    Calculate mean Average Precision (mAP) for object detection.

    Implements the PASCAL VOC evaluation protocol with configurable IoU thresholds.

    Args:
        num_classes: Number of object classes
        iou_threshold: IoU threshold for considering a detection as correct (default: 0.5)
        conf_threshold: Confidence threshold for filtering predictions (default: 0.01)
        nms_threshold: IoU threshold for non-maximum suppression (default: 0.4)

    Example:
        >>> metric = mAPMetric(num_classes=20, iou_threshold=0.5)
        >>> metric.reset()
        >>> # During evaluation loop:
        >>> for images, targets in dataloader:
        ...     predictions = model(images)
        ...     metric.update(predictions, targets)
        >>> results = metric.compute()
        >>> print(f"mAP@0.5: {results['mAP']:.4f}")
    """

    def __init__(
        self,
        num_classes: int,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.01,
        nms_threshold: float = 0.4,
        S: int = 7,
        B: int = 2,
    ):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.S = S
        self.B = B

        # Storage for predictions and ground truths
        self.all_predictions = []  # List of (class_id, confidence, bbox) per image
        self.all_ground_truths = []  # List of (class_id, bbox) per image

    def reset(self):
        """Reset all stored predictions and ground truths."""
        self.all_predictions = []
        self.all_ground_truths = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metric with a batch of predictions and targets.

        Args:
            predictions: Model predictions of shape (batch, S, S, B*5 + C)
            targets: Ground truth targets of shape (batch, S, S, B*5 + C)
        """
        batch_size = predictions.shape[0]

        for i in range(batch_size):
            pred = predictions[i]  # (S, S, B*5 + C)
            target = targets[i]  # (S, S, B*5 + C)

            # Parse predictions and targets
            pred_boxes = self._parse_predictions(pred)
            gt_boxes = self._parse_ground_truth(target)

            # Apply NMS to predictions
            pred_boxes = self._apply_nms(pred_boxes)

            self.all_predictions.append(pred_boxes)
            self.all_ground_truths.append(gt_boxes)

    def compute(self) -> Dict[str, float]:
        """
        Compute mAP and per-class AP.

        Returns:
            Dictionary containing:
                - 'mAP': mean Average Precision across all classes
                - 'AP_class_X': Average Precision for class X
                - 'precision': overall precision
                - 'recall': overall recall
        """
        if len(self.all_predictions) == 0:
            return {"mAP": 0.0, "precision": 0.0, "recall": 0.0}

        # Calculate AP for each class
        aps = []
        results = {}

        for class_id in range(self.num_classes):
            ap, precision, recall = self._calculate_ap_for_class(class_id)
            aps.append(ap)
            results[f"AP_class_{class_id}"] = ap

        # Calculate mAP
        mAP = np.mean(aps)
        results["mAP"] = mAP

        # Calculate overall precision and recall
        overall_precision, overall_recall = self._calculate_overall_metrics()
        results["precision"] = overall_precision
        results["recall"] = overall_recall

        return results

    def _parse_predictions(
        self, pred: torch.Tensor
    ) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
        """
        Parse YOLO predictions into list of detections.

        Args:
            pred: Prediction tensor of shape (S, S, B*5 + C)

        Returns:
            List of (class_id, confidence, (x, y, w, h)) tuples
        """
        detections = []

        for i in range(self.S):
            for j in range(self.S):
                cell_pred = pred[i, j]

                # Extract class probabilities
                class_probs = cell_pred[self.B * 5 :]

                # Check each bounding box in the cell
                for b in range(self.B):
                    box_offset = b * 5
                    x, y, w, h, conf = cell_pred[box_offset : box_offset + 5]

                    # Convert to absolute coordinates
                    x_abs = (j + x.item()) / self.S
                    y_abs = (i + y.item()) / self.S
                    w_abs = w.item()
                    h_abs = h.item()
                    conf_val = conf.item()

                    # Get best class
                    class_id = torch.argmax(class_probs).item()
                    class_prob = class_probs[class_id].item()

                    # Final confidence is objectness * class probability
                    final_conf = conf_val * class_prob

                    if final_conf > self.conf_threshold:
                        detections.append(
                            (class_id, final_conf, (x_abs, y_abs, w_abs, h_abs))
                        )

        return detections

    def _parse_ground_truth(
        self, target: torch.Tensor
    ) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        """
        Parse YOLO target into list of ground truth boxes.

        Args:
            target: Target tensor of shape (S, S, B*5 + C)

        Returns:
            List of (class_id, (x, y, w, h)) tuples
        """
        gt_boxes = []

        for i in range(self.S):
            for j in range(self.S):
                cell_target = target[i, j]

                # Check if this cell contains an object
                if cell_target[4] > 0:  # Confidence for first box
                    # Extract box coordinates
                    x_cell = cell_target[0].item()
                    y_cell = cell_target[1].item()
                    w = cell_target[2].item()
                    h = cell_target[3].item()

                    # Convert to absolute coordinates
                    x_abs = (j + x_cell) / self.S
                    y_abs = (i + y_cell) / self.S

                    # Extract class ID
                    class_probs = cell_target[self.B * 5 :]
                    class_id = torch.argmax(class_probs).item()

                    gt_boxes.append((class_id, (x_abs, y_abs, w, h)))

        return gt_boxes

    def _apply_nms(
        self, detections: List[Tuple[int, float, Tuple[float, float, float, float]]]
    ) -> List[Tuple[int, float, Tuple[float, float, float, float]]]:
        """
        Apply non-maximum suppression to filter overlapping boxes.

        Args:
            detections: List of (class_id, confidence, bbox) tuples

        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)

        # Group by class
        class_detections = defaultdict(list)
        for det in detections:
            class_detections[det[0]].append(det)

        # Apply NMS per class
        keep = []
        for class_id, dets in class_detections.items():
            while len(dets) > 0:
                # Keep the detection with highest confidence
                current = dets.pop(0)
                keep.append(current)

                # Remove overlapping boxes
                dets[:] = [
                    det
                    for det in dets
                    if self._calculate_iou(current[2], det[2]) < self.nms_threshold
                ]

        return keep

    def _calculate_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculate Intersection over Union between two boxes.

        Args:
            box1: (x_center, y_center, width, height)
            box2: (x_center, y_center, width, height)

        Returns:
            IoU value
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to corner coordinates
        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(
            0, inter_y_max - inter_y_min
        )

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _calculate_ap_for_class(self, class_id: int) -> Tuple[float, float, float]:
        """
        Calculate Average Precision for a specific class.

        Args:
            class_id: Class ID to calculate AP for

        Returns:
            Tuple of (AP, precision, recall)
        """
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_ground_truths = []

        for img_idx, (preds, gts) in enumerate(
            zip(self.all_predictions, self.all_ground_truths)
        ):
            # Filter predictions for this class
            for pred in preds:
                if pred[0] == class_id:
                    class_predictions.append(
                        (img_idx, pred[1], pred[2])
                    )  # (img_idx, conf, bbox)

            # Filter ground truths for this class
            for gt in gts:
                if gt[0] == class_id:
                    class_ground_truths.append((img_idx, gt[1]))  # (img_idx, bbox)

        if len(class_ground_truths) == 0:
            return 0.0, 0.0, 0.0

        if len(class_predictions) == 0:
            return 0.0, 0.0, 0.0

        # Sort predictions by confidence (descending)
        class_predictions = sorted(class_predictions, key=lambda x: x[1], reverse=True)

        # Track which ground truths have been matched
        gt_matched = [False] * len(class_ground_truths)

        # Calculate true positives and false positives
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))

        for pred_idx, (img_idx, conf, pred_bbox) in enumerate(class_predictions):
            # Find ground truths for this image
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, (gt_img_idx, gt_bbox) in enumerate(class_ground_truths):
                if gt_img_idx != img_idx:
                    continue

                iou = self._calculate_iou(pred_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            # Check if prediction matches a ground truth
            if best_iou >= self.iou_threshold:
                if not gt_matched[best_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1  # Already matched
            else:
                fp[pred_idx] = 1  # No match

        # Calculate cumulative sums
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        # Calculate precision and recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + EPSILON)
        recalls = tp_cumsum / len(class_ground_truths)

        # Add sentinel values at the beginning
        precisions = np.concatenate(([1.0], precisions))
        recalls = np.concatenate(([0.0], recalls))

        # Compute AP using 11-point interpolation (PASCAL VOC 2007 style).
        # This method samples precision at 11 recall thresholds (0.0, 0.1, ..., 1.0),
        # taking the maximum precision for each threshold. It is the standard for VOC 2007,
        # and differs from later approaches (e.g., COCO) that use all points for integration.
        # The 11-point method provides a simple, interpretable metric but may be less sensitive
        # to small changes in precision-recall curves compared to continuous integration.
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p_interp = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
            ap += p_interp / 11

        # Return AP and final precision/recall
        final_precision = precisions[-1] if len(precisions) > 1 else 0.0
        final_recall = recalls[-1] if len(recalls) > 1 else 0.0

        return ap, final_precision, final_recall

    def _calculate_overall_metrics(self) -> Tuple[float, float]:
        """
        Calculate overall precision and recall across all classes.

        Returns:
            Tuple of (precision, recall)
        """
        total_tp = 0
        total_fp = 0
        total_gt = 0

        for preds, gts in zip(self.all_predictions, self.all_ground_truths):
            # Count ground truths
            total_gt += len(gts)

            # For each prediction, check if it matches any ground truth
            gt_matched = [False] * len(gts)

            for pred in preds:
                pred_class, pred_conf, pred_bbox = pred

                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, (gt_class, gt_bbox) in enumerate(gts):
                    if pred_class != gt_class:
                        continue

                    iou = self._calculate_iou(pred_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # Check if it's a true positive
                if best_iou >= self.iou_threshold and not gt_matched[best_gt_idx]:
                    total_tp += 1
                    gt_matched[best_gt_idx] = True
                else:
                    total_fp += 1

        precision = total_tp / (total_tp + total_fp + EPSILON)
        recall = total_tp / (total_gt + EPSILON)

        return precision, recall


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_classes: int = 20,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.01,
    nms_threshold: float = 0.4,
    S: int = 7,
    B: int = 2,
) -> Dict[str, float]:
    """
    Evaluate a YOLO model on a dataset and compute mAP.

    Args:
        model: YOLO model to evaluate
        dataloader: DataLoader for evaluation dataset
        device: Device to run evaluation on
        num_classes: Number of object classes
        iou_threshold: IoU threshold for mAP calculation
        conf_threshold: Confidence threshold for filtering predictions
        nms_threshold: NMS threshold
        S: Grid size
        B: Number of bounding boxes per cell

    Returns:
        Dictionary of metrics including mAP, per-class AP, precision, and recall

    Example:
        >>> results = evaluate_model(model, val_loader, device='cuda', num_classes=20)
        >>> print(f"mAP: {results['mAP']:.4f}")
        >>> print(f"Precision: {results['precision']:.4f}")
        >>> print(f"Recall: {results['recall']:.4f}")
    """
    model.eval()
    metric = mAPMetric(
        num_classes=num_classes,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        S=S,
        B=B,
    )

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            predictions = model(images)

            # Update metric
            metric.update(predictions, targets)

    # Compute final metrics
    results = metric.compute()

    return results
