import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Backbone(nn.Module):
    """Abstract feature extractor interface."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement forward method")


class YOLOv1Backbone(Backbone):
    """Original YOLOv1 convolutional backbone."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layers 3-5
            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layers 6-13
            *self._make_conv_block(512, 256, 512, 4),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv Layers 14-20
            *self._make_conv_block(1024, 512, 1024, 2),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            # Conv Layers 21-22
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

    def _make_conv_block(
        self, in_channels: int, mid_channels: int, out_channels: int, num_blocks: int
    ) -> list[nn.Module]:
        layers = []
        for _ in range(num_blocks):
            layers.extend(
                [
                    nn.Conv2d(in_channels, mid_channels, kernel_size=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.1),
                ]
            )
            in_channels = out_channels
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Feature tensor of shape (batch, 1024, 7, 7) for 448x448 input
        """
        return self.features(x)


class ResNetBackbone(Backbone):
    """ResNet50 backbone for transfer learning."""

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """
        Args:
            pretrained: Whether to use pretrained ImageNet weights
            freeze: Whether to freeze backbone parameters
        """
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

        if freeze:
            for param in resnet.parameters():
                param.requires_grad = False

        # Extract layers until the last convolutional layer (before avgpool)
        # This gives us (batch, 2048, 14, 14) for 448x448 input
        self.extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ResNet backbone.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Feature tensor of shape (batch, 2048, 14, 14) for 448x448 input
        """
        return self.extractor(x)


class YOLOv1(nn.Module):
    """YOLOv1 model for object detection with modular backbone support."""

    def __init__(
        self,
        backbone: Backbone | None = None,
        detection_head: nn.Module | None = None,
        num_classes: int = 20,
        S: int = 7,
        B: int = 2,
    ):
        """
        Args:
            backbone: Feature extractor backbone. If None, uses YOLOv1Backbone
            detection_head: Detection head module. If None, creates appropriate head based on backbone
            num_classes: Number of object classes
            S: Grid size (S x S)
            B: Number of bounding boxes per grid cell
        """
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B

        # Use default YOLOv1 backbone if none provided
        if backbone is None:
            backbone = YOLOv1Backbone()

        self.backbone = backbone

        # Create detection head if none provided
        if detection_head is None:
            # Determine input channels based on backbone type
            if isinstance(backbone, YOLOv1Backbone):
                # YOLOv1 backbone outputs 1024 channels at 7x7
                # Use simpler head for original backbone
                detection_head = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(1024 * S * S, 4096),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(0.5),
                    nn.Linear(4096, S * S * (B * 5 + num_classes)),
                )
            elif isinstance(backbone, ResNetBackbone):
                # ResNet backbone outputs 2048 channels at 14x14
                detection_head = DetectionHead(2048, num_classes, S, B)
            else:
                raise ValueError(
                    "Must provide detection_head for custom backbone types"
                )

        self.head = detection_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 3, H, W)

        Returns:
            Predictions of shape (batch, S, S, B*5 + num_classes)
        """
        features = self.backbone(x)
        x = self.head(features)

        # Reshape to (batch_size, S, S, B*5 + num_classes) if needed
        if x.dim() == 2:  # Flattened output from simple head
            x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)

        return x


class DetectionHead(nn.Module):
    """Detection head for YOLO with additional conv layers and FC layers."""

    def __init__(
        self, in_channels: int, num_classes: int = 20, S: int = 7, B: int = 2
    ) -> None:
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B

        # Additional convolutional layers
        # Input: (batch, in_channels, 14, 14) for ResNet50
        # After stride=2 conv: (batch, 1024, 7, 7)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )

        # Fully connected layers
        # After conv_layers: spatial size is S x S (7x7)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return x
