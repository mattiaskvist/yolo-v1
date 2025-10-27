"""Wrapper for torchvision's VOCDetection dataset to work with YOLO v1."""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


class VOCDetectionYOLO(Dataset):
    """
    Wrapper around torchvision's VOCDetection that converts annotations to YOLO format.

    This class uses torchvision's built-in VOCDetection dataset and transforms
    the annotations into the YOLO v1 target format (S x S x (5*B + C)).

    Args:
        root: Root directory where the dataset exists or will be downloaded
        year: Year of the VOC dataset ('2007', '2012', '2007-test', '2012-test')
        image_set: Image set to use ('train', 'val', 'trainval', 'test')
        download: If True, downloads the dataset from Kaggle using kagglehub
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        transform: Optional image transformations
        target_size: Target image size (width, height)
        augment: Whether to apply data augmentation (only for training)

    Example:
        >>> # Download from Kaggle (recommended - faster)
        >>> root = VOCDetectionYOLO.download_from_kaggle(year="2007")
        >>> dataset = VOCDetectionYOLO(
        ...     root=root,
        ...     year="2007",
        ...     image_set="train",
        ...     S=7,
        ...     B=2
        ... )

        >>> # Or use auto-download
        >>> dataset = VOCDetectionYOLO(
        ...     year="2007",
        ...     image_set="train",
        ...     download=True
        ... )
    """  # Pascal VOC class names (20 classes)

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

    split_paths = {
        "2007": {
            "trainval": "VOCtrainval_06-Nov-2007",
            "test": "VOCtest_06-Nov-2007",
            "train": "VOCtrainval_06-Nov-2007",
            "val": "VOCtrainval_06-Nov-2007",
        },
        "2012": {
            "trainval": "VOCtrainval_11-May-2012",
            "test": "VOCtest_11-May-2012",
            "train": "VOCtrainval_11-May-2012",
            "val": "VOCtrainval_11-May-2012",
        },
    }

    @staticmethod
    def download_from_kaggle(
        year: str = "2007",
        verbose: bool = True,
    ) -> Path | None:
        """
        Download Pascal VOC dataset from Kaggle using kagglehub.

        This is a convenience method that downloads the dataset from Kaggle,
        which is often faster and more reliable than the official source.

        Args:
            year: Year of the VOC dataset ('2007' or '2012')
            verbose: Whether to print progress messages

        Returns:
            Path object to the downloaded dataset root (to use as 'root' parameter), or None if failed

        Raises:
            ImportError: If kagglehub is not installed
            ValueError: If year is not supported

        Example:
            >>> root = VOCDetectionYOLO.download_from_kaggle(year="2007")
            >>> dataset = VOCDetectionYOLO(root=root, year="2007", image_set="train")
        """
        # Map year to Kaggle dataset
        kaggle_datasets = {
            "2007": "zaraks/pascal-voc-2007",
            "2012": "huanghanchina/pascal-voc-2012",
        }
        if year not in kaggle_datasets:
            raise ValueError(
                f"Year '{year}' not supported. Choose from: {list(kaggle_datasets.keys())}"
            )

        try:
            import kagglehub
        except ImportError:
            raise ImportError(
                "kagglehub package is required for Kaggle downloads.\n"
                "Install it with: pip install kagglehub\n"
                "Or update your dependencies: pip install -e ."
            )

        if verbose:
            print("\n" + "=" * 70)
            print(f"Downloading Pascal VOC {year} dataset from Kaggle...")
            print(f"Dataset: {kaggle_datasets[year]}")
            print("=" * 70)

        try:
            # Download the dataset - returns path to the dataset
            download_path = kagglehub.dataset_download(kaggle_datasets[year])
            download_path = Path(download_path)

            if verbose:
                print(f"\n✓ Dataset downloaded to: {download_path}")
            return download_path

        except Exception as e:
            if verbose:
                print(f"\n✗ Error downloading dataset: {e}")
                print("\nAlternatively, you can download manually from:")
                print(
                    f"  Kaggle: https://www.kaggle.com/datasets/{kaggle_datasets[year]}"
                )
                print("=" * 70 + "\n")
            return None

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
        augment: bool = True,
    ):
        """
        Initialize VOCDetectionYOLO dataset.

        Args:
            root: Root directory where the dataset exists or will be downloaded
            year: Year of the VOC dataset ('2007', '2012', '2007-test', '2012-test')
            image_set: Image set to use ('train', 'val', 'trainval', 'test')
            download: If True, downloads the dataset from Kaggle using kagglehub
            S: Grid size (default: 7)
            B: Number of bounding boxes per grid cell (default: 2)
            transform: Optional image transformations
            target_size: Target image size (width, height)
            augment: Whether to apply data augmentation (only for training)
        """
        self.S = S
        self.B = B
        self.C = len(self.VOC_CLASSES)
        self.target_size = target_size
        self.augment = augment and image_set == "train"  # Only augment training set
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.VOC_CLASSES)
        }
        self.class_names = self.VOC_CLASSES

        # Extract base year (remove '-test' suffix if present)
        base_year = year.split("-")[0]

        # Handle Kaggle download if requested
        if download:
            try:
                kaggle_root = self.download_from_kaggle(year=base_year, verbose=True)
                if kaggle_root:
                    # Use the Kaggle download path directly
                    root = kaggle_root
                    download = False  # Don't use torchvision download
                else:
                    raise RuntimeError(
                        f"Failed to download from Kaggle for year {base_year}"
                    )
            except ImportError as e:
                raise ImportError(
                    f"Kaggle download failed: {e}\n"
                    "Install kagglehub with: pip install kagglehub"
                )

        # Set up transforms
        if transform is None:
            # Use v2 transforms for proper bbox handling during augmentation
            if self.augment:
                self.transform = self._get_augmentation_transforms()
            else:
                # Simple transforms for validation/test
                self.transform = v2.Compose(
                    [
                        v2.Resize(target_size, antialias=True),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transform = transform

        # Convert root to Path if it's a string
        root = Path(root)
        # Create the underlying VOCDetection dataset
        self.voc_dataset = VOCDetection(
            root=root / self.split_paths[base_year][image_set],
            year=year,
            image_set=image_set,
            download=download,  # This will be False if Kaggle download succeeded
            transform=None,  # We'll apply transforms ourselves
        )

    def _get_augmentation_transforms(self):
        """
        Get augmentation transforms as specified in YOLO v1 paper.

        Implements:
        - Random scaling and translation up to 20% of original image size
        - Random exposure and saturation adjustment by up to 1.5x in HSV color space
        """
        return v2.Compose(
            [
                # Random scaling and translation (up to 20%)
                # RandomResizedCrop does scaling + translation
                v2.RandomResizedCrop(
                    size=self.target_size,
                    scale=(0.8, 1.2),  # Scale between 80% and 120% (20% range)
                    ratio=(0.8, 1.2),  # Allow some aspect ratio variation
                    antialias=True,
                ),
                # HSV color space adjustments
                # Brightness: exposure adjustment (affects V channel)
                # Hue: color tone adjustment
                # Saturation: up to 1.5x
                v2.ColorJitter(
                    brightness=0.5,  # Exposure adjustment (±50% = 1.5x range)
                    saturation=0.5,  # Saturation adjustment (0.5x to 1.5x range)
                    hue=0.1,  # Small hue variation
                ),
                # Convert to tensor and normalize
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
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

        # If augmentation is enabled, we need to handle bboxes during transforms
        if self.augment:
            # Parse bboxes and classes first
            bboxes_list, class_ids = self._extract_bboxes_from_annotation(annotation)

            if len(bboxes_list) > 0:
                # Convert to format expected by v2 transforms
                # Get original image size
                orig_width, orig_height = image.size

                # Convert normalized coords back to pixel coords for v2 transforms
                boxes_pixels = []
                for bbox in bboxes_list:
                    x_center, y_center, width, height = bbox
                    xmin = (x_center - width / 2) * orig_width
                    ymin = (y_center - height / 2) * orig_height
                    xmax = (x_center + width / 2) * orig_width
                    ymax = (y_center + height / 2) * orig_height
                    boxes_pixels.append([xmin, ymin, xmax, ymax])

                # Create BoundingBoxes object
                boxes = tv_tensors.BoundingBoxes(
                    boxes_pixels, format="XYXY", canvas_size=(orig_height, orig_width)
                )

                # Apply transforms (will transform both image and boxes)
                transformed = self.transform(
                    {"image": image, "boxes": boxes, "labels": torch.tensor(class_ids)}
                )
                image = transformed["image"]
                boxes = transformed["boxes"]
                class_ids = transformed["labels"].tolist()

                # Convert boxes back to normalized center format
                _, h, w = image.shape
                bboxes_normalized = []
                for box in boxes:
                    xmin, ymin, xmax, ymax = box.tolist()
                    x_center = ((xmin + xmax) / 2) / w
                    y_center = ((ymin + ymax) / 2) / h
                    width = (xmax - xmin) / w
                    height = (ymax - ymin) / h

                    # Clamp values
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))

                    bboxes_normalized.append([x_center, y_center, width, height])

                # Encode to YOLO format
                target = self._encode_target(bboxes_normalized, class_ids)
            else:
                # No objects, just transform image (pass empty boxes/labels)
                orig_width, orig_height = image.size
                empty_boxes = tv_tensors.BoundingBoxes(
                    [], format="XYXY", canvas_size=(orig_height, orig_width)
                )
                transformed = self.transform(
                    {"image": image, "boxes": empty_boxes, "labels": torch.tensor([])}
                )
                image = transformed["image"]
                target = torch.zeros((self.S, self.S, 5 * self.B + self.C))
        else:
            # No augmentation - use old path
            image = self.transform(image)
            target = self._parse_voc_annotation(annotation)

        return image, target

    def _extract_bboxes_from_annotation(self, annotation: dict) -> Tuple[list, list]:
        """
        Extract bounding boxes and class IDs from VOC annotation.

        Args:
            annotation: VOC annotation dictionary

        Returns:
            Tuple of (bboxes, class_ids)
            - bboxes: List of normalized bounding boxes [x_center, y_center, width, height]
            - class_ids: List of class IDs
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

        return bboxes, class_ids

    def _parse_voc_annotation(self, annotation: dict) -> torch.Tensor:
        """
        Parse VOC annotation dictionary and convert to YOLO target format.

        Args:
            annotation: VOC annotation dictionary from torchvision

        Returns:
            Target tensor of shape (S, S, 5*B + C)
        """
        # Extract bboxes and class IDs
        bboxes, class_ids = self._extract_bboxes_from_annotation(annotation)

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
