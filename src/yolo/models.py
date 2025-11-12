import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class Backbone(nn.Module):
    """Abstract base class for feature extractor backbones.

    All backbone implementations must inherit from this class and implement
    the forward method to extract features from input images.
    """

    def __init__(self):
        """Initialize the backbone."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (batch, 3, H, W).

        Returns:
            Feature tensor of shape (batch, channels, H', W').

        Raises:
            NotImplementedError: This method must be implemented by subclasses.

        """
        raise NotImplementedError("Subclasses must implement forward method")


class YOLOv1Backbone(Backbone):
    """Original YOLOv1 convolutional backbone from the paper.

    Implements the 24-layer convolutional architecture described in the
    original YOLO paper with alternating 1x1 and 3x3 convolutions,
    LeakyReLU activation, and max pooling layers.

    The architecture progressively reduces spatial dimensions while increasing
    feature channels: 448x448x3 → 224x224x64 → 112x112x192 → ... → 7x7x1024.
    """

    def __init__(self):
        """Initialize the YOLOv1 backbone with convolutional layers."""
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
        """Create a repeating block of 1x1 and 3x3 convolutions.

        This creates the characteristic alternating conv pattern used in YOLOv1,
        where 1x1 convolutions reduce dimensionality before 3x3 convolutions.

        Args:
            in_channels: Number of input channels.
            mid_channels: Number of intermediate channels (after 1x1 conv).
            out_channels: Number of output channels (after 3x3 conv).
            num_blocks: Number of times to repeat the 1x1→3x3 pattern.

        Returns:
            List of nn.Module layers forming the conv block.

        """
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
        """Forward pass through the YOLOv1 backbone.

        Args:
            x: Input tensor of shape (batch, 3, H, W). Expected H=W=448 for YOLO.

        Returns:
            Feature tensor of shape (batch, 1024, 7, 7) for 448x448 input.
            For other input sizes, spatial dimensions scale accordingly.

        """
        return self.features(x)


class ResNetBackbone(Backbone):
    """ResNet50 backbone for transfer learning.

    Uses a pretrained ResNet50 model from torchvision as the feature extractor.
    This typically provides better results and faster convergence than training
    the YOLOv1 backbone from scratch, especially with limited training data.

    The backbone extracts features up to the last convolutional layer (before
    the global average pooling), outputting 2048-channel features at 14x14
    spatial resolution for 448x448 inputs.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        """Initialize ResNet50 backbone.

        Args:
            pretrained: Whether to use pretrained ImageNet weights. Setting to True
                enables transfer learning from ImageNet classification.
            freeze: Whether to freeze backbone parameters during training. When True,
                only the detection head is trained, reducing memory and computation.

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
        """Forward pass through the ResNet50 backbone.

        Args:
            x: Input tensor of shape (batch, 3, H, W). Expected H=W=448 for YOLO.

        Returns:
            Feature tensor of shape (batch, 2048, 14, 14) for 448x448 input.
            Note this has 2048 channels (vs 1024 for YOLOv1Backbone) and 14x14
            spatial size (vs 7x7), which is downsampled by the detection head.

        """
        return self.extractor(x)


class YOLOv1(nn.Module):
    """YOLOv1 object detection model with modular backbone support.

    Combines a feature extraction backbone with a detection head to predict
    bounding boxes and class probabilities in a single forward pass.

    The model divides the input image into an SxS grid and predicts B bounding
    boxes per grid cell, each with 5 values (x, y, w, h, confidence) plus
    C class probabilities per cell.

    Attributes:
        num_classes: Number of object classes (C).
        S: Grid size (typically 7 for 7x7 grid).
        B: Number of bounding boxes per grid cell (typically 2).
        backbone: Feature extractor backbone network.
        head: Detection head that predicts boxes and classes from features.

    """

    def __init__(
        self,
        backbone: Backbone | None = None,
        detection_head: nn.Module | None = None,
        num_classes: int = 20,
        S: int = 7,
        B: int = 2,
    ):
        """Initialize YOLOv1 model.

        Args:
            backbone: Feature extractor backbone. If None, uses YOLOv1Backbone.
                Can be YOLOv1Backbone for the original architecture or ResNetBackbone
                for transfer learning.
            detection_head: Detection head module. If None, creates appropriate head
                based on backbone type. Must be provided for custom backbones.
            num_classes: Number of object classes (default 20 for PASCAL VOC).
            S: Grid size dividing the image (default 7 for 7x7 grid).
            B: Number of bounding boxes predicted per grid cell (default 2).

        Raises:
            ValueError: If detection_head is None with a custom backbone type.

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
        """Forward pass through the YOLO network.

        Args:
            x: Input tensor of shape (batch, 3, H, W). Expected H=W=448.

        Returns:
            Predictions of shape (batch, S, S, B*5 + num_classes).
            Each grid cell contains B bounding box predictions (x, y, w, h, confidence)
            followed by C class probabilities. Coordinates x, y are relative to the cell,
            while w, h are relative to the full image.

        """
        features = self.backbone(x)
        x = self.head(features)

        # Reshape to (batch_size, S, S, B*5 + num_classes) if needed
        if x.dim() == 2:  # Flattened output from simple head
            x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)

        return x


class DetectionHead(nn.Module):
    """Detection head for YOLO with additional convolutional and fully connected layers.

    This head is primarily used with ResNetBackbone to process the 2048-channel
    features and reduce them to the required YOLO output format. It includes
    additional convolutional layers to reduce spatial dimensions from 14x14 to 7x7
    before the fully connected layers.

    Architecture:
        - Conv layers: 2048 → 1024 channels, 14x14 → 7x7 spatial
        - FC layers: Flatten → 4096 → S*S*(B*5 + C)
        - Reshape to: (batch, S, S, B*5 + C)
    """

    def __init__(
        self, in_channels: int, num_classes: int = 20, S: int = 7, B: int = 2
    ) -> None:
        """Initialize the detection head.

        Args:
            in_channels: Number of input channels from backbone (e.g., 2048 for ResNet50).
            num_classes: Number of object classes (default 20 for PASCAL VOC).
            S: Grid size for output predictions (default 7).
            B: Number of bounding boxes per grid cell (default 2).

        """
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
        """Forward pass through the detection head.

        Args:
            x: Feature tensor from backbone of shape (batch, in_channels, H, W).
                Expected (batch, 2048, 14, 14) from ResNet50.

        Returns:
            Detection predictions of shape (batch, S, S, B*5 + num_classes).

        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return x
