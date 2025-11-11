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
# Start training locally
uv run modal run src/train.py --epochs 135

# Resume from latest checkpoint and train remotely on Modal, make sure to resume if we get evicted
uv run modal run -d src/train.py --resume true --epochs 135 --remote --device cuda
```

### Using Different Backbones

```python
from yolo import YOLOv1, ResNetBackbone
backbone = ResNetBackbone(pretrained=True, freeze=True)
model = YOLOv1(backbone=backbone)
```

