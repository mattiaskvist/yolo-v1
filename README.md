# YOLO v1

## Quick Start

### Download Dataset

Datasets auto-download during training. Or manually:

```python
from yolo.dataset import VOCDetectionYOLO
VOCDetectionYOLO.download_from_kaggle(year="2007") # or "2012"
```

### Training

```bash
# Train locally (auto-detects device: mps/cuda/cpu)
uv run modal run src/train.py --epochs 135

# Or specify device explicitly
uv run modal run src/train.py --epochs 135 --device mps  # for Mac M1/M2
uv run modal run src/train.py --epochs 135 --device cpu  # for CPU only

# Resume from latest checkpoint and train remotely on Modal
uv run modal run -d src/train.py --resume true --epochs 135 --remote --device cuda
```

### Using Different Backbones

```python
from yolo import YOLOv1, ResNetBackbone
backbone = ResNetBackbone(pretrained=True, freeze=True)
model = YOLOv1(backbone=backbone)
```
