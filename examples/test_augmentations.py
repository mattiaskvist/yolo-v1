"""
Test script to verify data augmentations are working correctly.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from yolo.dataset import VOCDetectionYOLO


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def draw_bounding_boxes(ax, target, S=7, w=448, h=448, color="red"):
    """
    Draw bounding boxes from YOLO target tensor on a matplotlib axis.

    Args:
        ax: Matplotlib axis to draw on
        target: YOLO target tensor of shape (S, S, 5*B + C)
        S: Grid size
        w: Image width
        h: Image height
        color: Color of the bounding boxes
    """
    for i in range(S):
        for j in range(S):
            if target[i, j, 4] > 0:  # If cell has object
                x_cell = target[i, j, 0]
                y_cell = target[i, j, 1]
                width = target[i, j, 2]
                height = target[i, j, 3]

                # Convert to image coordinates
                x_center = (j + x_cell) / S * w
                y_center = (i + y_cell) / S * h
                box_w = width * w
                box_h = height * h

                # Draw box
                rect = patches.Rectangle(
                    (x_center - box_w / 2, y_center - box_h / 2),
                    box_w,
                    box_h,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)


def visualize_augmentation(idx=0, num_samples=4):
    """
    Visualize the same image with different augmentations.

    Args:
        idx: Index of the image to visualize
        num_samples: Number of augmented versions to show
    """
    # Create dataset WITH augmentation
    dataset_augmented = VOCDetectionYOLO(
        root="./data",
        year="2007",
        image_set="train",
        download=False,
        S=7,
        B=2,
        augment=True,
    )

    # Create dataset WITHOUT augmentation for comparison
    dataset_original = VOCDetectionYOLO(
        root="./data",
        year="2007",
        image_set="train",
        download=False,
        S=7,
        B=2,
        augment=False,
    )

    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    # Get original image (no augmentation)
    orig_image, orig_target = dataset_original[idx]
    orig_image = denormalize(orig_image)

    # Show original in first column
    axes[0, 0].imshow(orig_image.permute(1, 2, 0))
    axes[0, 0].set_title("Original (No Augmentation)")
    axes[0, 0].axis("off")

    # Draw bounding boxes on original
    axes[1, 0].imshow(orig_image.permute(1, 2, 0))
    axes[1, 0].set_title("Original with Boxes")
    axes[1, 0].axis("off")

    # Draw boxes using helper function
    S = 7
    h, w = 448, 448
    draw_bounding_boxes(axes[1, 0], orig_target, S, w, h, color="red")

    # Get augmented versions
    for col in range(1, num_samples):
        aug_image, aug_target = dataset_augmented[idx]
        aug_image = denormalize(aug_image)

        # Show augmented image
        axes[0, col].imshow(aug_image.permute(1, 2, 0))
        axes[0, col].set_title(f"Augmented {col}")
        axes[0, col].axis("off")

        # Show augmented image with boxes
        axes[1, col].imshow(aug_image.permute(1, 2, 0))
        axes[1, col].set_title(f"Augmented {col} with Boxes")
        axes[1, col].axis("off")

        # Draw boxes using helper function
        draw_bounding_boxes(axes[1, col], aug_target, S, w, h, color="red")

    plt.tight_layout()
    plt.savefig("augmentation_test.png", dpi=150, bbox_inches="tight")
    print("Visualization saved to augmentation_test.png")
    plt.show()


if __name__ == "__main__":
    print("Testing data augmentations...")
    print("\nAugmentations implemented:")
    print("- Random scaling and translation (up to 20%)")
    print("- HSV color space adjustments:")
    print("  - Brightness/Exposure adjustment (up to 1.5x)")
    print("  - Saturation adjustment (up to 1.5x)")
    print("\nGenerating visualization...\n")

    visualize_augmentation(idx=4, num_samples=4)

    print("\nYou should see:")
    print("- Different crops/scales of the same image")
    print("- Different color/brightness variations")
    print("- Bounding boxes properly transformed with the images")
