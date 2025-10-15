import torch.nn as nn


class YOLOv1(nn.Module):
    """YOLOv1 model for object detection."""

    def __init__(self, num_classes=20, S=7, B=2):
        """
        Args:
            num_classes: Number of object classes
            S: Grid size (S x S)
            B: Number of bounding boxes per grid cell
        """
        super(YOLOv1, self).__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B

        # Convolutional layers
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

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (B * 5 + num_classes)),
        )

    def _make_conv_block(self, in_channels, mid_channels, out_channels, num_blocks):
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # Reshape to (batch_size, S, S, B*5 + num_classes)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.num_classes)
        return x
