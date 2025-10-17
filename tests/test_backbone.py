"""
Unit tests for the modular backbone architecture.
"""

import pytest
import torch

from yolo.models import (
    Backbone,
    DetectionHead,
    ResNetBackbone,
    YOLOv1,
    YOLOv1Backbone,
)


class TestBackboneInterface:
    """Tests for the Backbone abstract class."""

    def test_backbone_is_abstract(self):
        """Test that Backbone cannot be instantiated directly."""
        backbone = Backbone()
        x = torch.randn(1, 3, 448, 448)

        with pytest.raises(NotImplementedError):
            backbone(x)


class TestYOLOv1Backbone:
    """Tests for the YOLOv1Backbone implementation."""

    def test_output_shape(self):
        """Test that YOLOv1Backbone produces correct output shape."""
        backbone = YOLOv1Backbone()
        x = torch.randn(2, 3, 448, 448)

        output = backbone(x)

        assert output.shape == (2, 1024, 7, 7), (
            f"Expected (2, 1024, 7, 7), got {output.shape}"
        )

    def test_forward_pass(self):
        """Test that forward pass completes without errors."""
        backbone = YOLOv1Backbone()
        x = torch.randn(1, 3, 448, 448)

        output = backbone(x)

        assert output is not None
        assert not torch.isnan(output).any()


class TestResNetBackbone:
    """Tests for the ResNetBackbone implementation."""

    def test_output_shape(self):
        """Test that ResNetBackbone produces correct output shape."""
        backbone = ResNetBackbone(pretrained=False)
        x = torch.randn(2, 3, 448, 448)

        output = backbone(x)

        assert output.shape == (2, 2048, 14, 14), (
            f"Expected (2, 2048, 14, 14), got {output.shape}"
        )

    def test_frozen_parameters(self):
        """Test that backbone parameters can be frozen."""
        backbone = ResNetBackbone(pretrained=False, freeze=True)

        frozen_params = [p for p in backbone.parameters() if not p.requires_grad]
        trainable_params = [p for p in backbone.parameters() if p.requires_grad]

        assert len(frozen_params) > 0, "Expected some frozen parameters"
        assert len(trainable_params) == 0, (
            "Expected no trainable parameters when frozen"
        )

    def test_trainable_parameters(self):
        """Test that backbone parameters can be trainable."""
        backbone = ResNetBackbone(pretrained=False, freeze=False)

        trainable_params = [p for p in backbone.parameters() if p.requires_grad]

        assert len(trainable_params) > 0, "Expected trainable parameters"


class TestYOLOv1Model:
    """Tests for the modular YOLOv1 model."""

    def test_default_backbone(self):
        """Test YOLOv1 with default backbone."""
        model = YOLOv1(num_classes=20, S=7, B=2)
        x = torch.randn(1, 3, 448, 448)

        output = model(x)

        assert output.shape == (1, 7, 7, 30), (
            f"Expected (1, 7, 7, 30), got {output.shape}"
        )
        assert isinstance(model.backbone, YOLOv1Backbone)

    def test_yolov1_backbone(self):
        """Test YOLOv1 with explicit YOLOv1Backbone."""
        backbone = YOLOv1Backbone()
        model = YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)
        x = torch.randn(2, 3, 448, 448)

        output = model(x)

        assert output.shape == (2, 7, 7, 30), (
            f"Expected (2, 7, 7, 30), got {output.shape}"
        )

    def test_resnet_backbone(self):
        """Test YOLOv1 with ResNetBackbone."""
        backbone = ResNetBackbone(pretrained=False, freeze=True)
        model = YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)
        x = torch.randn(1, 3, 448, 448)

        output = model(x)

        assert output.shape == (1, 7, 7, 30), (
            f"Expected (1, 7, 7, 30), got {output.shape}"
        )

    def test_custom_detection_head(self):
        """Test YOLOv1 with custom backbone and detection head."""
        backbone = ResNetBackbone(pretrained=False, freeze=True)
        head = DetectionHead(in_channels=2048, num_classes=20, S=7, B=2)
        model = YOLOv1(backbone=backbone, detection_head=head, num_classes=20, S=7, B=2)
        x = torch.randn(1, 3, 448, 448)

        output = model(x)

        assert output.shape == (1, 7, 7, 30), (
            f"Expected (1, 7, 7, 30), got {output.shape}"
        )

    def test_different_grid_sizes(self):
        """Test YOLOv1 with different grid sizes."""
        for S in [7, 14]:
            _ = YOLOv1(num_classes=20, S=S, B=2)
            # Note: This test assumes input size is adjusted for different S
            # For S=14, you might need different input dimensions
            # This test just verifies model creation doesn't fail

    def test_different_num_classes(self):
        """Test YOLOv1 with different number of classes."""
        for num_classes in [20, 80, 100]:
            model = YOLOv1(num_classes=num_classes, S=7, B=2)
            x = torch.randn(1, 3, 448, 448)

            output = model(x)
            expected_last_dim = 2 * 5 + num_classes

            assert output.shape[-1] == expected_last_dim, (
                f"Expected last dim {expected_last_dim}, got {output.shape[-1]}"
            )

    def test_different_num_boxes(self):
        """Test YOLOv1 with different number of bounding boxes."""
        for B in [1, 2, 3]:
            model = YOLOv1(num_classes=20, S=7, B=B)
            x = torch.randn(1, 3, 448, 448)

            output = model(x)
            expected_last_dim = B * 5 + 20

            assert output.shape[-1] == expected_last_dim, (
                f"Expected last dim {expected_last_dim}, got {output.shape[-1]}"
            )

    def test_batch_processing(self):
        """Test YOLOv1 with different batch sizes."""
        model = YOLOv1(num_classes=20, S=7, B=2)

        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 3, 448, 448)
            output = model(x)

            assert output.shape[0] == batch_size, (
                f"Expected batch size {batch_size}, got {output.shape[0]}"
            )

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = YOLOv1(num_classes=20, S=7, B=2)
        x = torch.randn(1, 3, 448, 448, requires_grad=True)

        output = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow back to input"


class TestDetectionHead:
    """Tests for the DetectionHead class."""

    def test_output_shape(self):
        """Test that DetectionHead produces correct output shape."""
        head = DetectionHead(in_channels=2048, num_classes=20, S=7, B=2)
        x = torch.randn(2, 2048, 14, 14)

        output = head(x)

        assert output.shape == (2, 7, 7, 30), (
            f"Expected (2, 7, 7, 30), got {output.shape}"
        )

    def test_forward_pass(self):
        """Test that forward pass completes without errors."""
        head = DetectionHead(in_channels=2048, num_classes=20, S=7, B=2)
        x = torch.randn(1, 2048, 14, 14)

        output = head(x)

        assert output is not None
        assert not torch.isnan(output).any()


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with old API."""

    def test_old_api_still_works(self):
        """Test that old-style initialization still works."""
        # Old API: just specify num_classes, S, B
        model = YOLOv1(num_classes=20, S=7, B=2)
        x = torch.randn(1, 3, 448, 448)

        output = model(x)

        assert output.shape == (1, 7, 7, 30), "Old API should still work"
        assert isinstance(model.backbone, YOLOv1Backbone), (
            "Should use YOLOv1Backbone by default"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
