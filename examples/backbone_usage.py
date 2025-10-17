"""
Example demonstrating the modular backbone interface for YOLOv1.

This script shows how to create YOLO models with different backbones.
"""

import torch
from yolo.models import (
    YOLOv1,
    YOLOv1Backbone,
    ResNetBackbone,
    DetectionHead,
)


def main():
    """Demonstrate different ways to create YOLO models."""

    # Example 1: Default YOLOv1 (uses YOLOv1Backbone automatically)
    print("=" * 60)
    print("Example 1: Default YOLOv1 model")
    print("=" * 60)
    model1 = YOLOv1(num_classes=20, S=7, B=2)
    print("Model created with default backbone")
    print(f"Backbone type: {type(model1.backbone).__name__}")
    print()

    # Test forward pass
    x = torch.randn(1, 3, 448, 448)
    output = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be (1, 7, 7, 30) for 20 classes
    print()

    # Example 2: YOLOv1 with explicit YOLOv1Backbone
    print("=" * 60)
    print("Example 2: YOLOv1 with explicit YOLOv1Backbone")
    print("=" * 60)
    backbone = YOLOv1Backbone()
    model2 = YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)
    print("Model created with YOLOv1Backbone")
    print(f"Backbone type: {type(model2.backbone).__name__}")
    output2 = model2(x)
    print(f"Output shape: {output2.shape}")
    print()

    # Example 3: YOLOv1 with ResNet50 backbone (pretrained, frozen)
    print("=" * 60)
    print("Example 3: YOLOv1 with ResNet50 backbone (pretrained, frozen)")
    print("=" * 60)
    resnet_backbone = ResNetBackbone(pretrained=True, freeze=True)
    model3 = YOLOv1(backbone=resnet_backbone, num_classes=20, S=7, B=2)
    print("Model created with ResNet50 backbone")
    print(f"Backbone type: {type(model3.backbone).__name__}")
    output3 = model3(x)
    print(f"Output shape: {output3.shape}")

    # Check if backbone parameters are frozen
    frozen_params = sum(1 for p in model3.backbone.parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in model3.backbone.parameters() if p.requires_grad)
    print(f"Frozen parameters in backbone: {frozen_params}")
    print(f"Trainable parameters in backbone: {trainable_params}")
    print()

    # Example 4: YOLOv1 with ResNet50 backbone (pretrained, trainable)
    print("=" * 60)
    print("Example 4: YOLOv1 with ResNet50 backbone (pretrained, trainable)")
    print("=" * 60)
    resnet_backbone_trainable = ResNetBackbone(pretrained=True, freeze=False)
    model4 = YOLOv1(backbone=resnet_backbone_trainable, num_classes=20, S=7, B=2)
    print("Model created with trainable ResNet50 backbone")

    # Check trainable parameters
    frozen_params = sum(1 for p in model4.backbone.parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in model4.backbone.parameters() if p.requires_grad)
    print(f"Frozen parameters in backbone: {frozen_params}")
    print(f"Trainable parameters in backbone: {trainable_params}")
    print()

    # Example 5: Custom backbone + detection head combination
    print("=" * 60)
    print("Example 5: Custom backbone + detection head")
    print("=" * 60)
    custom_backbone = ResNetBackbone(pretrained=True, freeze=True)
    custom_head = DetectionHead(in_channels=2048, num_classes=20, S=7, B=2)
    model5 = YOLOv1(
        backbone=custom_backbone, detection_head=custom_head, num_classes=20, S=7, B=2
    )
    print("Model created with custom backbone and detection head")
    output5 = model5(x)
    print(f"Output shape: {output5.shape}")
    print()

    # Example 6: Compare parameter counts
    print("=" * 60)
    print("Example 6: Parameter comparison")
    print("=" * 60)
    models = [
        ("YOLOv1 (default)", model1),
        ("YOLOv1 + ResNet (frozen)", model3),
        ("YOLOv1 + ResNet (trainable)", model4),
    ]

    for name, model in models:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {total_params - trainable_params:,}")
        print()


if __name__ == "__main__":
    main()
