"""Pydantic schemas for YOLO predictions and results."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in normalized coordinates (0-1)."""

    x: float = Field(
        ..., ge=0.0, le=1.0, description="Center x coordinate (normalized)"
    )
    y: float = Field(
        ..., ge=0.0, le=1.0, description="Center y coordinate (normalized)"
    )
    width: float = Field(..., ge=0.0, le=1.0, description="Box width (normalized)")
    height: float = Field(..., ge=0.0, le=1.0, description="Box height (normalized)")

    def to_corners(self) -> tuple[float, float, float, float]:
        """Convert center format to corner format.

        Returns:
            Tuple of (x1, y1, x2, y2) in normalized coordinates

        """
        x1 = self.x - self.width / 2
        y1 = self.y - self.height / 2
        x2 = self.x + self.width / 2
        y2 = self.y + self.height / 2
        return (x1, y1, x2, y2)

    def to_pixel_coords(
        self, img_width: int, img_height: int
    ) -> tuple[int, int, int, int]:
        """Convert to pixel coordinates.

        Args:
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            Tuple of (x1, y1, x2, y2) in pixel coordinates

        """
        x1, y1, x2, y2 = self.to_corners()
        return (
            int(x1 * img_width),
            int(y1 * img_height),
            int(x2 * img_width),
            int(y2 * img_height),
        )

    @property
    def area(self) -> float:
        """Calculate normalized area of the bounding box."""
        return self.width * self.height

    @classmethod
    def from_corners(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        """Create BoundingBox from corner coordinates.

        Args:
            x1, y1: Top-left corner (normalized)
            x2, y2: Bottom-right corner (normalized)

        Returns:
            BoundingBox instance

        """
        width = x2 - x1
        height = y2 - y1
        x = x1 + width / 2
        y = y1 + height / 2
        return cls(x=x, y=y, width=width, height=height)
    
    # custom str method for easier reading
    def __str__(self) -> str:
        x1, y1, x2, y2 = self.to_corners()
        return f"({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})"


class Detection(BaseModel):
    """Single object detection with class and bounding box."""

    class_id: int = Field(..., ge=0, description="Predicted class ID")
    class_name: str | None = Field(
        None, description="Class name (e.g., 'person', 'car')"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
