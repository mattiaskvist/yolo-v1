"""Wrapper for torchvision's VOCDetection dataset to work with YOLO v1."""

from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms


class VOCDetectionYOLO(Dataset):
    """
    Wrapper around torchvision's VOCDetection that converts annotations to YOLO format.

    This class uses torchvision's built-in VOCDetection dataset and transforms
    the annotations into the YOLO v1 target format (S x S x (5*B + C)).

    Args:
        root: Root directory where the dataset exists or will be downloaded
        year: Year of the VOC dataset ('2007', '2012', '2007-test', '2012-test')
        image_set: Image set to use ('train', 'val', 'trainval', 'test')
        download: If True, downloads the dataset if not found
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        transform: Optional image transformations
        target_size: Target image size (width, height)

    Example:
        >>> dataset = VOCDetectionYOLO(
        ...     root="./data",
        ...     year="2007",
        ...     image_set="train",
        ...     download=True,
        ...     S=7,
        ...     B=2
        ... )
    """

    # Pascal VOC class names (20 classes)
    VOC_CLASSES = [
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    def __init__(
        self,
        root: str = "./data",
        year: str = "2007",
        image_set: str = "train",
        download: bool = False,
        S: int = 7,
        B: int = 2,
        transform: transforms.Compose = None,
        target_size: Tuple[int, int] = (448, 448),
    ):
        self.S = S
        self.B = B
        self.C = len(self.VOC_CLASSES)
        self.target_size = target_size
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.VOC_CLASSES)
        }
        self.class_names = self.VOC_CLASSES

        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transform

        # Create the underlying VOCDetection dataset
        self.voc_dataset = VOCDetection(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=None,  # We'll apply transforms ourselves
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.voc_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, target)
            - image: Preprocessed image tensor of shape (3, H, W)
            - target: Target tensor of shape (S, S, 5*B + C)
        """
        # Get image and annotation from VOCDetection
        image, annotation = self.voc_dataset[idx]

        # Apply transforms to image
        if self.transform:
            image = self.transform(image)

        # Parse annotation and convert to YOLO format
        target = self._parse_voc_annotation(annotation)

        return image, target

    def _parse_voc_annotation(self, annotation: dict) -> torch.Tensor:
        """
        Parse VOC annotation dictionary and convert to YOLO target format.

        Args:
            annotation: VOC annotation dictionary from torchvision

        Returns:
            Target tensor of shape (S, S, 5*B + C)
        """
        # Extract image size
        size = annotation["annotation"]["size"]
        img_width = float(size["width"])
        img_height = float(size["height"])

        # Parse objects
        bboxes = []
        class_ids = []

        objects = annotation["annotation"]["object"]
        # Handle single object case (not a list)
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            # Get class name and ID
            class_name = obj["name"]
            if class_name not in self.class_to_idx:
                continue  # Skip unknown classes

            class_id = self.class_to_idx[class_name]

            # Get bounding box
            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            # Convert to center coordinates and normalize
            x_center = ((xmin + xmax) / 2.0) / img_width
            y_center = ((ymin + ymax) / 2.0) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Ensure values are in valid range
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            bboxes.append([x_center, y_center, width, height])
            class_ids.append(class_id)

        # Encode to YOLO target format
        target = self._encode_target(bboxes, class_ids)

        return target

    def _encode_target(
        self,
        bboxes: list,
        class_ids: list,
    ) -> torch.Tensor:
        """
        Encode bounding boxes and classes into YOLO target format.

        Args:
            bboxes: List of normalized bounding boxes [x_center, y_center, width, height]
            class_ids: List of class IDs corresponding to each bounding box

        Returns:
            Target tensor of shape (S, S, 5*B + C)
        """
        # Initialize target tensor: (S, S, 5*B + C)
        target = torch.zeros((self.S, self.S, 5 * self.B + self.C))

        for bbox, class_id in zip(bboxes, class_ids):
            x_center, y_center, width, height = bbox

            # Determine which grid cell this object belongs to
            i = int(self.S * y_center)  # Row
            j = int(self.S * x_center)  # Column

            # Ensure indices are within bounds
            i = min(i, self.S - 1)
            j = min(j, self.S - 1)

            # Calculate cell-relative coordinates
            x_cell = self.S * x_center - j
            y_cell = self.S * y_center - i

            # Check if this cell already has an object
            if target[i, j, 4] == 0:  # If no object assigned yet
                # Set bounding box coordinates for first box
                target[i, j, 0] = x_cell
                target[i, j, 1] = y_cell
                target[i, j, 2] = width
                target[i, j, 3] = height
                target[i, j, 4] = 1.0  # Confidence

                # Set class probabilities
                target[i, j, 5 * self.B + class_id] = 1.0

        return target

    def visualize_sample(self, idx: int) -> dict:
        """
        Get sample information for visualization.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing image path, bboxes, and class names
        """
        # Get annotation from VOCDetection
        _, annotation = self.voc_dataset[idx]

        # Extract image size
        size = annotation["annotation"]["size"]
        img_width = float(size["width"])
        img_height = float(size["height"])

        # Parse objects
        bboxes = []
        class_ids = []

        objects = annotation["annotation"]["object"]
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            class_name = obj["name"]
            if class_name not in self.class_to_idx:
                continue

            class_id = self.class_to_idx[class_name]

            bbox = obj["bndbox"]
            xmin = float(bbox["xmin"])
            ymin = float(bbox["ymin"])
            xmax = float(bbox["xmax"])
            ymax = float(bbox["ymax"])

            # Convert to center coordinates and normalize
            x_center = ((xmin + xmax) / 2.0) / img_width
            y_center = ((ymin + ymax) / 2.0) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            bboxes.append([x_center, y_center, width, height])
            class_ids.append(class_id)

        return {
            "image_path": self.voc_dataset.images[idx],
            "bboxes": bboxes,
            "class_ids": class_ids,
            "class_names": [self.class_names[cid] for cid in class_ids],
        }
