"""YOLO v1 implementation in PyTorch."""

from .dataset import VOCDetectionYOLO, CombinedVOCDataset, create_voc_datasets
from .loss import YOLOLoss
from .metrics import mAPMetric, evaluate_model
from .models import (
    Backbone,
    DetectionHead,
    ResNetBackbone,
    YOLOv1,
    YOLOv1Backbone,
)

__version__ = "0.1.0"

__all__ = [
    "Backbone",
    "CombinedVOCDataset",
    "DetectionHead",
    "ResNetBackbone",
    "VOCDetectionYOLO",
    "YOLOLoss",
    "YOLOv1",
    "YOLOv1Backbone",
    "create_voc_datasets",
    "evaluate_model",
    "mAPMetric",
]
