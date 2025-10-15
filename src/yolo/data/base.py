"""Base dataset class for YOLO v1 object detection."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BaseYOLODataset(Dataset, ABC):
    """
    Abstract base class for YOLO v1 datasets.
    
    This class defines the interface that all YOLO dataset implementations
    must follow. Subclasses should implement dataset-specific loading and
    parsing logic.
    
    Args:
        root_dir: Root directory containing the dataset
        split: Dataset split ('train', 'val', 'test')
        S: Grid size (default: 7 for YOLO v1)
        B: Number of bounding boxes per grid cell (default: 2)
        C: Number of classes
        transform: Optional image transformations
        target_size: Target image size (width, height)
    """
    
    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        S: int = 7,
        B: int = 2,
        C: int = 20,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (448, 448),
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per cell
        self.C = C  # Number of classes
        self.target_size = target_size
        
        # Set up default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load dataset-specific data
        self.samples = self._load_samples()
        self.class_names = self._load_class_names()
    
    @abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load and return list of samples.
        
        Each sample should be a dictionary containing at minimum:
        - 'image_path': Path to the image file
        - 'annotations': List of annotations (bounding boxes and classes)
        
        Returns:
            List of sample dictionaries
        """
        pass
    
    @abstractmethod
    def _load_class_names(self) -> List[str]:
        """
        Load and return list of class names.
        
        Returns:
            List of class name strings
        """
        pass
    
    @abstractmethod
    def _parse_annotation(self, sample: Dict[str, Any]) -> Tuple[List[List[float]], List[int]]:
        """
        Parse annotation for a single sample.
        
        Args:
            sample: Sample dictionary containing annotation information
            
        Returns:
            Tuple of (bounding_boxes, class_ids)
            - bounding_boxes: List of [x_center, y_center, width, height] normalized to [0, 1]
            - class_ids: List of class IDs corresponding to each box
        """
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
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
        sample = self.samples[idx]
        
        # Load image
        image_path = sample['image_path']
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Parse annotations
        bboxes, class_ids = self._parse_annotation(sample)
        
        # Convert to YOLO target format
        target = self._encode_target(bboxes, class_ids, original_width, original_height)
        
        return image, target
    
    def _encode_target(
        self,
        bboxes: List[List[float]],
        class_ids: List[int],
        original_width: int,
        original_height: int
    ) -> torch.Tensor:
        """
        Encode bounding boxes and classes into YOLO target format.
        
        Args:
            bboxes: List of normalized bounding boxes [x_center, y_center, width, height]
            class_ids: List of class IDs
            original_width: Original image width
            original_height: Original image height
            
        Returns:
            Target tensor of shape (S, S, 5*B + C)
            Each grid cell contains: [x, y, w, h, confidence] * B + [class_probs] * C
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
    
    def visualize_sample(self, idx: int) -> Dict[str, Any]:
        """
        Get sample information for visualization.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing image path, bboxes, and class names
        """
        sample = self.samples[idx]
        bboxes, class_ids = self._parse_annotation(sample)
        
        return {
            'image_path': sample['image_path'],
            'bboxes': bboxes,
            'class_ids': class_ids,
            'class_names': [self.class_names[cid] for cid in class_ids]
        }
