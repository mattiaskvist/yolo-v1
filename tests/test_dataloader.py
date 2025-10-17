"""Unit tests for YOLO v1 dataloaders."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from yolo.data import BaseYOLODataset
from yolo.data.utils import compute_iou, non_max_suppression


class MockYOLODataset(BaseYOLODataset):
    """Mock dataset for testing base functionality."""

    def __init__(self, num_samples: int = 10, **kwargs) -> None:
        self.num_samples = num_samples
        super().__init__(root_dir="/tmp", **kwargs)

    def _load_samples(self) -> list[dict]:
        """Create mock samples."""
        return [
            {
                "image_path": Path("/tmp/image.jpg"),
                "bbox": [[0.5, 0.5, 0.2, 0.3]],
                "class_id": [0],
            }
            for _ in range(self.num_samples)
        ]

    def _load_class_names(self) -> list[str]:
        """Return mock class names."""
        return ["class_0", "class_1", "class_2"]

    def _parse_annotation(self, sample: dict) -> tuple:
        """Return mock annotations."""
        return sample["bbox"], sample["class_id"]


def test_base_dataset_initialization() -> None:
    """Test that base dataset initializes correctly."""
    dataset = MockYOLODataset(num_samples=5, S=7, B=2, C=3)

    assert len(dataset) == 5
    assert dataset.S == 7
    assert dataset.B == 2
    assert dataset.C == 3
    assert len(dataset.class_names) == 3


def test_target_encoding() -> None:
    """Test that target encoding produces correct shape."""
    dataset = MockYOLODataset(num_samples=1, S=7, B=2, C=20)

    # Mock image loading
    with patch("PIL.Image.open") as mock_open:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_img.size = (224, 224)
        mock_open.return_value = mock_img

        with patch.object(dataset, "transform") as mock_transform:
            mock_transform.return_value = torch.randn(3, 448, 448)

            image, target = dataset[0]

    # Check shapes
    assert image.shape == (3, 448, 448)
    assert target.shape == (7, 7, 5 * 2 + 20)  # S=7, B=2, C=20


def test_compute_iou() -> None:
    """Test IoU computation."""
    # Identical boxes
    box1 = [0.5, 0.5, 0.2, 0.2]
    box2 = [0.5, 0.5, 0.2, 0.2]
    assert compute_iou(box1, box2) == pytest.approx(1.0)

    # Non-overlapping boxes
    box1 = [0.2, 0.2, 0.1, 0.1]
    box2 = [0.8, 0.8, 0.1, 0.1]
    assert compute_iou(box1, box2) == pytest.approx(0.0)

    # Partially overlapping boxes
    box1 = [0.5, 0.5, 0.4, 0.4]
    box2 = [0.6, 0.6, 0.4, 0.4]
    iou = compute_iou(box1, box2)
    assert 0.0 < iou < 1.0


def test_non_max_suppression() -> None:
    """Test NMS removes overlapping boxes."""
    bboxes = [
        [0.5, 0.5, 0.2, 0.2],
        [0.52, 0.51, 0.21, 0.19],  # Overlapping with first
        [0.8, 0.8, 0.1, 0.1],
    ]
    class_ids = [0, 0, 1]
    confidences = [0.9, 0.8, 0.85]

    filtered_boxes, filtered_classes, filtered_confs = non_max_suppression(
        bboxes, class_ids, confidences, iou_threshold=0.5
    )

    # Should keep first box (higher confidence) and third box (different location)
    assert len(filtered_boxes) == 2
    assert filtered_confs[0] == 0.9  # Highest confidence kept
    assert filtered_classes[1] == 1  # Different class kept


def test_grid_cell_assignment() -> None:
    """Test that objects are assigned to correct grid cells."""
    dataset = MockYOLODataset(num_samples=1, S=7, B=2, C=3)

    # Create annotation at specific grid cell
    bboxes = [[0.5, 0.5, 0.1, 0.1]]  # Center of image (normalized coordinates)
    class_ids = [0]

    target = dataset._encode_target(bboxes, class_ids)

    # Object at (0.5, 0.5) should be in grid cell (3, 3)
    i, j = 3, 3

    # Check that this cell has an object
    assert target[i, j, 4] == 1.0  # Confidence
    assert target[i, j, 10] == 1.0  # Class 0 probability (after 5*B=10 values)

    # Check relative coordinates within cell
    assert 0.0 <= target[i, j, 0] <= 1.0  # x relative to cell
    assert 0.0 <= target[i, j, 1] <= 1.0  # y relative to cell


def test_multiple_objects_per_image() -> None:
    """Test handling multiple objects in single image."""
    dataset = MockYOLODataset(num_samples=1, S=7, B=2, C=3)

    # Create multiple objects (normalized coordinates)
    bboxes = [
        [0.2, 0.3, 0.1, 0.1],
        [0.7, 0.8, 0.15, 0.2],
    ]
    class_ids = [0, 1]

    target = dataset._encode_target(bboxes, class_ids)

    # Count cells with objects
    cells_with_objects = (target[:, :, 4] > 0).sum().item()
    assert cells_with_objects == 2  # Two objects in different cells


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
