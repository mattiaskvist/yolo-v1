import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
from typing import Generator

from yolo.inference import YOLOInference
from yolo.models import YOLOv1


@pytest.fixture
def model() -> YOLOv1:
    """Create a YOLO model for testing"""
    return YOLOv1(num_classes=20, S=7, B=2)


@pytest.fixture
def inference_engine(model: YOLOv1) -> YOLOInference:
    """Create an inference engine"""
    return YOLOInference(model, device="cpu")


@pytest.fixture
def sample_image() -> Generator[str, None, None]:
    """Create a temporary sample image"""
    img = Image.new("RGB", (448, 448), color="red")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        yield tmp.name

    os.unlink(tmp.name)


@pytest.fixture
def sample_predictions() -> torch.Tensor:
    """Create sample prediction tensor"""
    # Shape: (S, S, B*5 + C) = (7, 7, 30)
    pred = torch.randn(7, 7, 30)
    # Set some high confidence values
    pred[2, 3, 4] = 0.9  # First box confidence
    pred[2, 3, 9] = 0.8  # Second box confidence
    return pred


class TestYOLOInference:
    def test_initialization(self, model: YOLOv1) -> None:
        """Test inference engine initialization"""
        inference = YOLOInference(model, device="cpu")

        assert inference.model == model
        assert inference.device == "cpu"
        assert hasattr(inference, "transform")
        assert not inference.model.training

    def test_predict_returns_list(
        self, inference_engine: YOLOInference, sample_image: str
    ) -> None:
        """Test that predict returns a list"""
        detections = inference_engine.predict(sample_image)

        assert isinstance(detections, list)

    def test_predict_with_thresholds(
        self, inference_engine: YOLOInference, sample_image: str
    ) -> None:
        """Test predict with different threshold values"""
        detections_low = inference_engine.predict(
            sample_image, conf_threshold=0.1, nms_threshold=0.4
        )
        detections_high = inference_engine.predict(
            sample_image, conf_threshold=0.9, nms_threshold=0.4
        )

        # Higher threshold should return fewer or equal detections
        assert len(detections_high) <= len(detections_low)

    def test_predict_invalid_image_path(self, inference_engine: YOLOInference) -> None:
        """Test predict with invalid image path"""
        with pytest.raises(FileNotFoundError):
            inference_engine.predict("nonexistent_image.jpg")

    def test_parse_predictions_shape(
        self, inference_engine: YOLOInference, sample_predictions: torch.Tensor
    ) -> None:
        """Test _parse_predictions output format"""
        detections = inference_engine._parse_predictions(
            sample_predictions, conf_threshold=0.5
        )

        assert isinstance(detections, list)
        for det in detections:
            assert len(det) == 6  # [class_id, conf, x, y, w, h]
            # Confidence can be > 1 since it's conf * class_prob without sigmoid
            assert det[1] >= 0  # confidence should be non-negative

    def test_parse_predictions_confidence_threshold(
        self, inference_engine: YOLOInference
    ) -> None:
        """Test that confidence threshold filters detections"""
        pred = torch.zeros(7, 7, 30)
        # Set one high confidence box
        pred[0, 0, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 0.9])
        pred[0, 0, 10] = 0.8  # class probability

        detections_low = inference_engine._parse_predictions(pred, conf_threshold=0.5)
        detections_high = inference_engine._parse_predictions(pred, conf_threshold=0.9)

        assert len(detections_low) > 0
        assert len(detections_high) == 0

    def test_iou_identical_boxes(self, inference_engine: YOLOInference) -> None:
        """Test IoU of identical boxes"""
        box = [0.5, 0.5, 0.3, 0.3]
        iou = inference_engine._iou(box, box)

        assert iou == pytest.approx(1.0, abs=1e-4)

    def test_iou_no_overlap(self, inference_engine: YOLOInference) -> None:
        """Test IoU of non-overlapping boxes"""
        box1 = [0.2, 0.2, 0.1, 0.1]
        box2 = [0.8, 0.8, 0.1, 0.1]
        iou = inference_engine._iou(box1, box2)

        assert iou == pytest.approx(0.0, abs=1e-5)

    def test_iou_partial_overlap(self, inference_engine: YOLOInference) -> None:
        """Test IoU of partially overlapping boxes"""
        box1 = [0.5, 0.5, 0.4, 0.4]
        box2 = [0.6, 0.6, 0.4, 0.4]
        iou = inference_engine._iou(box1, box2)

        assert 0 < iou < 1

    def test_iou_symmetry(self, inference_engine: YOLOInference) -> None:
        """Test that IoU is symmetric"""
        box1 = [0.3, 0.3, 0.2, 0.2]
        box2 = [0.4, 0.4, 0.2, 0.2]

        iou1 = inference_engine._iou(box1, box2)
        iou2 = inference_engine._iou(box2, box1)

        assert iou1 == pytest.approx(iou2, abs=1e-5)

    def test_nms_empty_list(self, inference_engine):
        """Test NMS with empty detection list"""
        detections = inference_engine._non_max_suppression([], nms_threshold=0.5)

        assert detections == []

    def test_nms_single_detection(self, inference_engine: YOLOInference) -> None:
        """Test NMS with single detection"""
        detections = [[0, 0.9, 0.5, 0.5, 0.3, 0.3]]
        result = inference_engine._non_max_suppression(detections, nms_threshold=0.5)

        assert len(result) == 1
        assert result[0] == detections[0]

    def test_nms_removes_overlapping_boxes(
        self, inference_engine: YOLOInference
    ) -> None:
        """Test that NMS removes overlapping boxes of same class"""
        detections = [
            [0, 0.9, 0.5, 0.5, 0.3, 0.3],  # High confidence
            [0, 0.7, 0.52, 0.52, 0.3, 0.3],  # Lower confidence, overlapping
        ]
        result = inference_engine._non_max_suppression(detections, nms_threshold=0.3)

        assert len(result) == 1
        assert result[0][1] == 0.9  # Kept the higher confidence box

    def test_nms_keeps_different_classes(self, inference_engine: YOLOInference) -> None:
        """Test that NMS keeps boxes of different classes"""
        detections = [
            [0, 0.9, 0.5, 0.5, 0.3, 0.3],  # Class 0
            [1, 0.8, 0.52, 0.52, 0.3, 0.3],  # Class 1, overlapping
        ]
        result = inference_engine._non_max_suppression(detections, nms_threshold=0.3)

        assert len(result) == 2

    def test_nms_keeps_non_overlapping_boxes(
        self, inference_engine: YOLOInference
    ) -> None:
        """Test that NMS keeps non-overlapping boxes"""
        detections = [
            [0, 0.9, 0.2, 0.2, 0.1, 0.1],  # Box 1
            [0, 0.8, 0.8, 0.8, 0.1, 0.1],  # Box 2, far away
        ]
        result = inference_engine._non_max_suppression(detections, nms_threshold=0.5)

        assert len(result) == 2

    def test_detection_format(
        self, inference_engine: YOLOInference, sample_image: torch.Tensor
    ) -> None:
        """Test that detections have correct format"""
        detections = inference_engine.predict(sample_image, conf_threshold=0.1)

        for det in detections:
            assert len(det) == 6
            class_id, conf, x, y, w, h = det

            assert conf >= 0
            assert isinstance(class_id, (int, np.integer))
            assert 0 <= class_id < 20
            assert 0 <= conf <= 1
            assert 0 <= x <= 1
            assert 0 <= y <= 1
            assert 0 <= w <= 1
            assert 0 <= h <= 1


class TestTransform:
    def test_transform_output_shape(
        self, inference_engine: YOLOInference, sample_image: str
    ) -> None:
        """Test that transform produces correct tensor shape"""
        # Load the image from the path
        img = Image.open(sample_image).convert("RGB")
        tensor = inference_engine.transform(img)

        assert tensor.shape == (3, 448, 448)

    def test_transform_output_range(
        self, inference_engine: YOLOInference, sample_image: str
    ) -> None:
        """Test that transform normalizes values"""
        # Load the image from the path
        img = Image.open(sample_image).convert("RGB")
        tensor = inference_engine.transform(img)

        # After normalization, values should roughly be in [-3, 3]
        assert tensor.min() >= -5
        assert tensor.max() <= 5
