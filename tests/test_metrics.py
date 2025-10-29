"""
Tests for mAP metrics implementation.
"""

import torch
import pytest
from src.yolo.metrics import mAPMetric


class TestMapMetric:
    """Test mAPMetric class."""

    def test_initialization(self):
        """Test metric initialization."""
        metric = mAPMetric(num_classes=20, iou_threshold=0.5)
        assert metric.num_classes == 20
        assert metric.iou_threshold == 0.5
        assert metric.S == 7
        assert metric.B == 2

    def test_reset(self):
        """Test metric reset."""
        metric = mAPMetric(num_classes=20)

        # Add some dummy data
        metric.all_predictions.append([(0, 0.9, (0.5, 0.5, 0.2, 0.2))])
        metric.all_ground_truths.append([(0, (0.5, 0.5, 0.2, 0.2))])

        # Reset
        metric.reset()

        assert len(metric.all_predictions) == 0
        assert len(metric.all_ground_truths) == 0

    def test_iou_calculation(self):
        """Test IoU calculation."""
        metric = mAPMetric(num_classes=20)

        # Identical boxes
        box1 = (0.5, 0.5, 0.2, 0.2)
        box2 = (0.5, 0.5, 0.2, 0.2)
        iou = metric._calculate_iou(box1, box2)
        assert iou == pytest.approx(1.0, abs=1e-5)

        # Non-overlapping boxes
        box1 = (0.2, 0.2, 0.1, 0.1)
        box2 = (0.8, 0.8, 0.1, 0.1)
        iou = metric._calculate_iou(box1, box2)
        assert iou == 0.0

        # Partially overlapping boxes
        box1 = (0.5, 0.5, 0.4, 0.4)
        box2 = (0.6, 0.6, 0.4, 0.4)
        iou = metric._calculate_iou(box1, box2)
        assert 0 < iou < 1

    def test_parse_predictions(self):
        """Test parsing of YOLO predictions."""
        metric = mAPMetric(num_classes=20, conf_threshold=0.1, S=7, B=2)

        # Create a dummy prediction tensor
        pred = torch.zeros(7, 7, 30)  # S=7, B=2, C=20: B*5 + C = 30

        # Add a high-confidence prediction at cell (3, 3)
        pred[3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 0.9])  # Box 1
        pred[3, 3, 10] = 1.0  # Class 0 (index 10 = B*5 + 0)

        detections = metric._parse_predictions(pred)

        # Should have at least one detection
        assert len(detections) > 0

        # Check first detection
        class_id, conf, bbox = detections[0]
        assert class_id == 0
        assert conf > 0.1  # Above threshold

    def test_parse_ground_truth(self):
        """Test parsing of ground truth targets."""
        metric = mAPMetric(num_classes=20, S=7, B=2)

        # Create a dummy target tensor
        target = torch.zeros(7, 7, 30)

        # Add a ground truth at cell (3, 3)
        target[3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])
        target[3, 3, 10] = 1.0  # Class 0

        gt_boxes = metric._parse_ground_truth(target)

        # Should have one ground truth
        assert len(gt_boxes) == 1

        # Check ground truth
        class_id, bbox = gt_boxes[0]
        assert class_id == 0

    def test_nms(self):
        """Test non-maximum suppression."""
        metric = mAPMetric(num_classes=20, nms_threshold=0.5)

        # Create overlapping detections
        detections = [
            (0, 0.9, (0.5, 0.5, 0.2, 0.2)),  # High confidence
            (0, 0.8, (0.52, 0.52, 0.2, 0.2)),  # Overlapping, lower confidence
            (1, 0.85, (0.7, 0.7, 0.15, 0.15)),  # Different class
        ]

        filtered = metric._apply_nms(detections)

        # Should keep high confidence box for class 0 and the class 1 box
        assert len(filtered) == 2

        # Check that highest confidence detection is kept
        class_0_dets = [d for d in filtered if d[0] == 0]
        assert len(class_0_dets) == 1
        assert class_0_dets[0][1] == 0.9

    def test_update_and_compute(self):
        """Test update and compute methods."""
        metric = mAPMetric(num_classes=20, S=7, B=2)

        # Create dummy predictions and targets
        batch_size = 2
        predictions = torch.zeros(batch_size, 7, 7, 30)
        targets = torch.zeros(batch_size, 7, 7, 30)

        # Add some detections
        for i in range(batch_size):
            # Prediction
            predictions[i, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 0.8])
            predictions[i, 3, 3, 10] = 1.0  # Class 0

            # Ground truth
            targets[i, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])
            targets[i, 3, 3, 10] = 1.0  # Class 0

        # Update metric
        metric.update(predictions, targets)

        # Compute results
        results = metric.compute()

        # Check results structure
        assert "mAP" in results
        assert "precision" in results
        assert "recall" in results
        assert all(f"AP_class_{i}" in results for i in range(20))
        # Check that AP values are within [0, 1]
        for i in range(20):
            ap_value = results[f"AP_class_{i}"]
            assert 0.0 <= ap_value <= 1.0

    def test_perfect_predictions(self):
        """Test with perfect predictions (should get mAP=1.0)."""
        metric = mAPMetric(num_classes=20, S=7, B=2, iou_threshold=0.5)

        # Create identical predictions and targets
        batch_size = 5
        predictions = torch.zeros(batch_size, 7, 7, 30)
        targets = torch.zeros(batch_size, 7, 7, 30)

        for i in range(batch_size):
            # Add perfect prediction/target pairs
            predictions[i, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])
            predictions[i, 3, 3, 10] = 1.0  # Class 0

            targets[i, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])
            targets[i, 3, 3, 10] = 1.0  # Class 0

        metric.update(predictions, targets)
        results = metric.compute()

        # With perfect predictions, AP for class 0 should be 1.0
        assert results["AP_class_0"] == pytest.approx(1.0, abs=1e-5)
        # Precision and recall should be 1.0
        assert results["precision"] == pytest.approx(1.0, abs=1e-5)
        assert results["recall"] == pytest.approx(1.0, abs=1e-5)

    def test_no_predictions(self):
        """Test with no predictions."""
        metric = mAPMetric(num_classes=20, conf_threshold=0.9)  # High threshold

        # Create targets but predictions below threshold
        batch_size = 2
        predictions = torch.zeros(batch_size, 7, 7, 30)
        targets = torch.zeros(batch_size, 7, 7, 30)

        # Add low confidence predictions
        predictions[:, 3, 3, 4] = 0.1  # Low confidence

        # Add ground truths
        targets[:, 3, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 1.0])
        targets[:, 3, 3, 10] = 1.0

        metric.update(predictions, targets)
        results = metric.compute()

        # Should have low recall (no detections)
        assert results["recall"] == 0.0


def test_iou_edge_cases():
    """Test IoU calculation edge cases."""
    metric = mAPMetric(num_classes=20)

    # Zero-area box
    box1 = (0.5, 0.5, 0.0, 0.0)
    box2 = (0.5, 0.5, 0.2, 0.2)
    iou = metric._calculate_iou(box1, box2)
    assert iou == 0.0

    # Both zero-area boxes
    box1 = (0.5, 0.5, 0.0, 0.0)
    box2 = (0.5, 0.5, 0.0, 0.0)
    iou = metric._calculate_iou(box1, box2)
    assert iou == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
