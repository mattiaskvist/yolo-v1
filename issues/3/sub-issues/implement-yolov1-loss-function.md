### Loss Function Implementation
- Localization loss: MSE on (x, y, w, h) with sqrt for w and h terms, as per YOLO v1 paper.
- Confidence loss: Separate terms for object/no-object confidence.
- Classification loss: Cross-entropy or MSE for class probabilities.
- Use weighting: λ_coord=5 for localization, λ_noobj=0.5 for no-object confidence.

#### Notes:
- Integrate with YOLOv1 model output.
- Add comments and documentation for clarity.