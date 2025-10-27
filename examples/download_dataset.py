#!/usr/bin/env python3
"""Example script demonstrating dataset download from Kaggle."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yolo.dataset import VOCDetectionYOLO


def download_voc_2007():
    """Download Pascal VOC 2007 dataset from Kaggle."""
    print("=" * 70)
    print("Pascal VOC 2007 Dataset Download Example")
    print("=" * 70)

    # Method 1: Use static method (recommended for just downloading)
    print("\nMethod 1: Using static method")
    print("-" * 70)
    download_path = VOCDetectionYOLO.download_from_kaggle(year="2007", verbose=True)

    if download_path:
        print("\n✓ Download successful!")

        # Now you can create datasets without download=True
        print("\nCreating dataset...")
        dataset = VOCDetectionYOLO(
            root=download_path,
            year="2007",
            image_set="train",
            download=False,  # Already downloaded
        )
        print(f"✓ Dataset created with {len(dataset)} images")
    else:
        print("\n✗ Download failed. See error messages above.")
        return False

    return True


def download_with_auto_download():
    """Download using auto-download."""
    print("\n" + "=" * 70)
    print("Method 2: Auto-download with dataset creation")
    print("=" * 70)

    try:
        dataset = VOCDetectionYOLO(
            year="2007",
            image_set="train",
            download=True,
        )
        print(f"\n✓ Dataset ready with {len(dataset)} images")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


if __name__ == "__main__":
    print("\nThis script demonstrates two ways to download Pascal VOC datasets:\n")
    print("1. Static method: VOCDetectionYOLO.download_from_kaggle()")
    print("2. Auto-download: download=True")
    print("\nNote: You need Kaggle API credentials set up for Kaggle downloads.")
    print("See docs/KAGGLE_DOWNLOAD.md for setup instructions.\n")

    # Try method 1
    if download_voc_2007():
        print("\n" + "=" * 70)
        print("All methods completed successfully!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("Some methods failed. Check error messages above.")
        print("=" * 70)
        sys.exit(1)
