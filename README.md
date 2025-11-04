# YOLO v1

## Training

```bash
# Start training locally
uv run modal run src/train.py --epochs 135

# Resume from latest checkpoint and train remotely on Modal, make sure to resume if we get evicted
uv run modal run -detach src/train.py --resume true --epochs 135 --remote
```
