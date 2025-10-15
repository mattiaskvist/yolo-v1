"""Configuration file for YOLO v1 training and data loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    """Configuration for data loading."""

    # Dataset settings
    dataset_name: str = "pascal_voc"  # 'pascal_voc' or 'coco'
    root_dir: Path = Path("data/VOCdevkit/VOC2007")
    train_split: str = "train"
    val_split: str = "val"

    # YOLO settings
    S: int = 7  # Grid size
    B: int = 2  # Bounding boxes per cell
    C: int = 20  # Number of classes

    # Image settings
    image_size: Tuple[int, int] = (448, 448)

    # Data loading
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True

    # Augmentation (for future implementation)
    use_augmentation: bool = False
    horizontal_flip: bool = True
    color_jitter: bool = True


@dataclass
class ModelConfig:
    """Configuration for YOLO v1 model."""

    # Architecture
    backbone: str = "darknet"  # Base network
    dropout: float = 0.5

    # Grid settings (inherited from data)
    S: int = 7
    B: int = 2
    C: int = 20

    # Pretrained weights
    pretrained_backbone: bool = False
    weights_path: Path | None = None


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training duration
    epochs: int = 135
    warmup_epochs: int = 0

    # Optimizer
    learning_rate: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # Learning rate schedule
    lr_schedule: str = "step"  # 'step', 'cosine', 'polynomial'
    lr_decay_epochs: list = None  # e.g., [75, 105]
    lr_decay_factor: float = 0.1

    # Loss weights (as per original paper)
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5

    # Gradient clipping
    clip_grad_norm: float = 10.0

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_frequency: int = 5  # Save every N epochs

    # Logging
    log_frequency: int = 10  # Log every N batches

    def __post_init__(self):
        if self.lr_decay_epochs is None:
            self.lr_decay_epochs = [75, 105]


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'

    # Reproducibility
    seed: int = 42

    # Experiment tracking
    experiment_name: str = "yolo_v1"
    output_dir: Path = Path("outputs")

    def __post_init__(self):
        # Ensure model config matches data config
        self.model.S = self.data.S
        self.model.B = self.data.B
        self.model.C = self.data.C

        # Create output directories
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.training.checkpoint_dir.mkdir(exist_ok=True, parents=True)


def get_config() -> Config:
    """Get default configuration."""
    return Config()


def get_pascal_voc_config() -> Config:
    """Get configuration for Pascal VOC dataset."""
    config = Config()
    config.data.dataset_name = "pascal_voc"
    config.data.root_dir = Path("data/VOCdevkit/VOC2007")
    config.data.C = 20
    config.model.C = 20
    return config