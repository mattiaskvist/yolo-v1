"""Pascal VOC dataset implementation for YOLO v1."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torchvision import transforms

from .base import BaseYOLODataset


class PascalVOCDataset(BaseYOLODataset):
    """
    Pascal VOC dataset for YOLO v1 object detection.
    
    Expected directory structure:
        root_dir/
            Annotations/
                *.xml
            JPEGImages/
                *.jpg
            ImageSets/
                Main/
                    train.txt
                    val.txt
                    trainval.txt
                    test.txt
    
    Args:
        root_dir: Root directory of Pascal VOC dataset
        year: Dataset year ('2007' or '2012')
        split: Dataset split ('train', 'val', 'trainval', 'test')
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        transform: Optional image transformations
        target_size: Target image size (width, height)
    """
    
    # Pascal VOC class names (20 classes)
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        root_dir: str | Path,
        year: str = "2007",
        split: str = "train",
        S: int = 7,
        B: int = 2,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (448, 448),
    ):
        self.year = year
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.VOC_CLASSES)}
        
        super().__init__(
            root_dir=root_dir,
            split=split,
            S=S,
            B=B,
            C=len(self.VOC_CLASSES),
            transform=transform,
            target_size=target_size,
        )
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load Pascal VOC samples from the split file.
        
        Returns:
            List of sample dictionaries
        """
        # Path to split file
        split_file = self.root_dir / f"ImageSets/Main/{self.split}.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Make sure the Pascal VOC dataset is properly structured."
            )
        
        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                image_id = line.strip()
                if not image_id:
                    continue
                
                image_path = self.root_dir / f"JPEGImages/{image_id}.jpg"
                annotation_path = self.root_dir / f"Annotations/{image_id}.xml"
                
                if image_path.exists() and annotation_path.exists():
                    samples.append({
                        'image_id': image_id,
                        'image_path': image_path,
                        'annotation_path': annotation_path,
                    })
        
        if len(samples) == 0:
            raise ValueError(
                f"No valid samples found for split '{self.split}' in {self.root_dir}"
            )
        
        return samples
    
    def _load_class_names(self) -> List[str]:
        """
        Return Pascal VOC class names.
        
        Returns:
            List of class name strings
        """
        return self.VOC_CLASSES
    
    def _parse_annotation(self, sample: Dict[str, Any]) -> Tuple[List[List[float]], List[int]]:
        """
        Parse Pascal VOC XML annotation file.
        
        Args:
            sample: Sample dictionary containing annotation_path
            
        Returns:
            Tuple of (bounding_boxes, class_ids)
            - bounding_boxes: List of [x_center, y_center, width, height] normalized to [0, 1]
            - class_ids: List of class IDs
        """
        tree = ET.parse(sample['annotation_path'])
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)
        
        bboxes = []
        class_ids = []
        
        # Parse each object
        for obj in root.findall('object'):
            # Get class name
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue  # Skip unknown classes
            
            class_id = self.class_to_idx[class_name]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
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
