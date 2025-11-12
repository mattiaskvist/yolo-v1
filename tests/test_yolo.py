import pytest
import torch
from PIL import Image
import tempfile
import os
from typing import Generator

from yolo.inference import YOLOInference
from yolo.models import YOLOv1
from yolo.schemas import Detection, BoundingBox


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
        """Test that predict returns a list of detections"""
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

    def test_parse_predictions_shape(self, inference_engine: YOLOInference) -> None:
        """Test parse_predictions output format"""
        # Create a prediction tensor with valid values (0-1 range)
        pred = torch.zeros(7, 7, 30)
        # Set one valid high confidence box with valid coordinates
        pred[2, 3, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 0.9])
        pred[2, 3, 10] = 0.8  # class probability

        detections = inference_engine.parse_predictions(pred, conf_threshold=0.5)

        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            # Accept both Python int and torch scalar types for class_id
            assert isinstance(det.class_id, int) or (
                hasattr(det.class_id, "item") and isinstance(det.class_id.item(), int)
            )
            assert isinstance(det.confidence, float)
            assert isinstance(det.bbox, BoundingBox)
            assert det.confidence >= 0  # confidence should be non-negative
            # Verify bbox values are valid (0-1)
            assert 0 <= det.bbox.x <= 1
            assert 0 <= det.bbox.y <= 1
            assert 0 <= det.bbox.width <= 1
            assert 0 <= det.bbox.height <= 1

    def test_parse_predictions_confidence_threshold(
        self, inference_engine: YOLOInference
    ) -> None:
        """Test that confidence threshold filters detections"""
        pred = torch.zeros(7, 7, 30)
        # Set one high confidence box
        pred[0, 0, 0:5] = torch.tensor([0.5, 0.5, 0.3, 0.3, 0.9])
        pred[0, 0, 10] = 0.8  # class probability

        detections_low = inference_engine.parse_predictions(pred, conf_threshold=0.5)
        detections_high = inference_engine.parse_predictions(pred, conf_threshold=0.9)

        assert len(detections_low) > 0
        assert len(detections_high) == 0

    def test_detection_bbox_validation(self) -> None:
        """Test that BoundingBox validates coordinates"""
        # Valid bbox
        bbox = BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3)
        assert bbox.x == 0.5
        assert bbox.area == pytest.approx(0.09, abs=1e-6)

        # Invalid coordinates should raise error
        with pytest.raises(ValueError):
            BoundingBox(x=1.5, y=0.5, width=0.3, height=0.3)  # x > 1

        with pytest.raises(ValueError):
            BoundingBox(x=0.5, y=0.5, width=-0.1, height=0.3)  # negative width

    def test_detection_coordinate_conversion(self) -> None:
        """Test BoundingBox coordinate conversion methods"""
        bbox = BoundingBox(x=0.5, y=0.5, width=0.4, height=0.6)

        # Test to_corners
        x1, y1, x2, y2 = bbox.to_corners()
        assert x1 == pytest.approx(0.3, abs=1e-6)
        assert y1 == pytest.approx(0.2, abs=1e-6)
        assert x2 == pytest.approx(0.7, abs=1e-6)
        assert y2 == pytest.approx(0.8, abs=1e-6)

        # Test to_pixel_coords
        x1_px, y1_px, x2_px, y2_px = bbox.to_pixel_coords(640, 480)
        assert x1_px == pytest.approx(192, abs=1)
        assert y1_px == pytest.approx(96, abs=1)
        assert x2_px == pytest.approx(448, abs=1)
        assert y2_px == pytest.approx(384, abs=1)

    def test_detections_list(self) -> None:
        """Test working with list of Detection objects"""
        detections = [
            Detection(
                class_id=0,
                class_name="class0",
                confidence=0.9,
                bbox=BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3),
            ),
            Detection(
                class_id=1,
                class_name="class1",
                confidence=0.6,
                bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),
            ),
            Detection(
                class_id=0,
                class_name="class0",
                confidence=0.4,
                bbox=BoundingBox(x=0.7, y=0.7, width=0.2, height=0.2),
            ),
        ]

        # Test basic properties
        assert len(detections) == 3
        assert detections[0].confidence == 0.9

        # Filter detections by confidence > 0.5
        filtered = [d for d in detections if d.confidence > 0.5]
        assert len(filtered) == 2
        assert all(d.confidence > 0.5 for d in filtered)

        # Sort detections by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        assert sorted_dets[0].confidence == 0.9
        assert sorted_dets[1].confidence == 0.6
        assert sorted_dets[2].confidence == 0.4

    def test_iou_identical_boxes(self, inference_engine: YOLOInference) -> None:
        """Test IoU of identical boxes"""
        box = BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3)
        iou = inference_engine.iou(box, box)

        assert iou == pytest.approx(1.0, abs=1e-4)

    def test_iou_no_overlap(self, inference_engine: YOLOInference) -> None:
        """Test IoU of non-overlapping boxes"""
        box1 = BoundingBox(x=0.2, y=0.2, width=0.1, height=0.1)
        box2 = BoundingBox(x=0.8, y=0.8, width=0.1, height=0.1)
        iou = inference_engine.iou(box1, box2)

        assert iou == pytest.approx(0.0, abs=1e-5)

    def test_iou_partial_overlap(self, inference_engine: YOLOInference) -> None:
        """Test IoU of partially overlapping boxes"""
        box1 = BoundingBox(x=0.5, y=0.5, width=0.4, height=0.4)
        box2 = BoundingBox(x=0.6, y=0.6, width=0.4, height=0.4)
        iou = inference_engine.iou(box1, box2)

        assert 0 < iou < 1

    def test_iou_symmetry(self, inference_engine: YOLOInference) -> None:
        """Test that IoU is symmetric"""
        box1 = BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2)
        box2 = BoundingBox(x=0.4, y=0.4, width=0.2, height=0.2)

        iou1 = inference_engine.iou(box1, box2)
        iou2 = inference_engine.iou(box2, box1)

        assert iou1 == pytest.approx(iou2, abs=1e-5)

    def test_nms_empty_list(self, inference_engine):
        """Test NMS with empty detection list"""
        detections = inference_engine.non_max_suppression([], iou_threshold=0.5)

        assert detections == []

    def test_nms_single_detection(self, inference_engine: YOLOInference) -> None:
        """Test NMS with single detection"""
        detections = [
            Detection(
                class_id=0,
                class_name="test",
                confidence=0.9,
                bbox=BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3),
            )
        ]
        result = inference_engine.non_max_suppression(detections, iou_threshold=0.5)

        assert len(result) == 1
        assert result[0] == detections[0]

    def test_nms_removes_overlapping_boxes(
        self, inference_engine: YOLOInference
    ) -> None:
        """Test that NMS removes overlapping boxes of same class"""
        detections = [
            Detection(
                class_id=0,
                class_name="test",
                confidence=0.9,
                bbox=BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3),
            ),  # High confidence
            Detection(
                class_id=0,
                class_name="test",
                confidence=0.7,
                bbox=BoundingBox(x=0.52, y=0.52, width=0.3, height=0.3),
            ),  # Lower confidence, overlapping
        ]
        result = inference_engine.non_max_suppression(detections, iou_threshold=0.3)

        assert len(result) == 1
        assert result[0].confidence == 0.9  # Kept the higher confidence box

    def test_nms_keeps_different_classes(self, inference_engine: YOLOInference) -> None:
        """Test that NMS keeps boxes of different classes"""
        detections = [
            Detection(
                class_id=0,
                class_name="test0",
                confidence=0.9,
                bbox=BoundingBox(x=0.5, y=0.5, width=0.3, height=0.3),
            ),  # Class 0
            Detection(
                class_id=1,
                class_name="test1",
                confidence=0.8,
                bbox=BoundingBox(x=0.52, y=0.52, width=0.3, height=0.3),
            ),  # Class 1, overlapping
        ]
        result = inference_engine.non_max_suppression(detections, iou_threshold=0.3)

        assert len(result) == 2

    def test_nms_keeps_non_overlapping_boxes(
        self, inference_engine: YOLOInference
    ) -> None:
        """Test that NMS keeps non-overlapping boxes"""
        detections = [
            Detection(
                class_id=0,
                class_name="test",
                confidence=0.9,
                bbox=BoundingBox(x=0.2, y=0.2, width=0.1, height=0.1),
            ),  # Box 1
            Detection(
                class_id=0,
                class_name="test",
                confidence=0.8,
                bbox=BoundingBox(x=0.8, y=0.8, width=0.1, height=0.1),
            ),  # Box 2, far away
        ]
        result = inference_engine.non_max_suppression(detections, iou_threshold=0.5)

        assert len(result) == 2

    def test_detection_format(
        self, inference_engine: YOLOInference, sample_image: str
    ) -> None:
        """Test that detections have correct format"""
        detections = inference_engine.predict(sample_image, conf_threshold=0.1)

        for det in detections:
            assert isinstance(det, Detection)
            assert isinstance(det.class_id, int)
            assert isinstance(det.confidence, float)
            assert isinstance(det.bbox, BoundingBox)

            assert det.confidence >= 0
            assert 0 <= det.class_id < 20
            assert 0 <= det.confidence <= 1
            assert 0 <= det.bbox.x <= 1
            assert 0 <= det.bbox.y <= 1
            assert 0 <= det.bbox.width <= 1
            assert 0 <= det.bbox.height <= 1


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
