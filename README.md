# YOLO v1

## Training

```bash
# Start training
uv run modal run src/train.py --epochs 135

# Resume from latest checkpoint
uv run modal run src/train.py --resume true --epochs 135

# Training with TensorBoard and mAP evaluation
uv run modal run src/train.py --epochs 135 --tensorboard --compute-map
```
