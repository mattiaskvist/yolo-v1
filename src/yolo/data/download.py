"""Utilities for setting up datasets."""

from pathlib import Path


def setup_pascal_voc(
    root_dir: Path = Path("data"),
    year: str = "2007",
    split: str = "trainval",
) -> Path:
    """
    Setup Pascal VOC dataset path.

    Args:
        root_dir: Root directory for datasets
        year: Dataset year ('2007' or '2012')
        split: Which split to use (for validation only)

    Returns:
        Path to VOC dataset directory

    Raises:
        ValueError: If year is not '2007' or '2012'
        FileNotFoundError: If dataset not found at expected location
    """
    # Validate year
    if year not in ["2007", "2012"]:
        raise ValueError(f"Unsupported year '{year}'. Choose '2007' or '2012'")

    # Validate split
    valid_splits = {"trainval", "train", "val", "test"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Choose from {valid_splits}")

    voc_dir = root_dir / "VOCdevkit" / f"VOC{year}"

    # Check if dataset exists
    if not voc_dir.exists() or not (voc_dir / "JPEGImages").exists():
        kaggle_url = (
            "https://www.kaggle.com/datasets/zaraks/pascal-voc-2007"
            if year == "2007"
            else "https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012"
        )
        raise FileNotFoundError(
            f"Pascal VOC {year} not found at {voc_dir}\n\n"
            f"Please download the dataset:\n"
            f"  Recommended (Kaggle): {kaggle_url}\n"
            f"  Official: http://host.robots.ox.ac.uk/pascal/VOC/voc{year}/\n\n"
            f"Then extract to: {root_dir}\n"
        )

    return voc_dir
