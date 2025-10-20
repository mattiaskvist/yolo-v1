"""
Quick test to verify inference setup is working correctly.

This script tests:
1. Model loading from checkpoint
2. Inference on a sample image
3. Output generation

Run this to verify everything is set up correctly before running full inference.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from yolo import YOLOv1, ResNetBackbone
from yolo.inference import YOLOInference
from PIL import Image
import tempfile


@pytest.fixture
def model():
    """Fixture that creates a model for testing."""
    backbone = ResNetBackbone(pretrained=False, freeze=False)
    return YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)


@pytest.fixture
def inference(model):
    """Fixture that creates an inference engine."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return YOLOInference(model, device=device)


@pytest.fixture
def device():
    """Fixture that returns the device to use."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_model_creation():
    """Test that we can create a model."""
    backbone = ResNetBackbone(pretrained=False, freeze=False)
    model = YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)
    assert model is not None
    assert isinstance(model, YOLOv1)


def test_inference_engine(model, device):
    """Test that inference engine works."""
    inference = YOLOInference(model, device=device)
    assert inference is not None
    assert isinstance(inference, YOLOInference)
    assert inference.device == device


def test_dummy_inference(inference, device):
    """Test inference on a dummy image."""
    # Create a dummy image
    dummy_img = Image.new("RGB", (448, 448), color=(128, 128, 128))

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = f.name
        dummy_img.save(temp_path)

    try:
        # Run inference
        detections = inference.predict(temp_path, conf_threshold=0.5, nms_threshold=0.4)

        # Verify detections is a list
        assert isinstance(detections, list)

        # Note: Random model may not detect anything - this is expected
    finally:
        # Clean up
        Path(temp_path).unlink()


def test_checkpoint_loading():
    """Test loading from checkpoint if available."""
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        pytest.skip("No checkpoints directory")

    checkpoints = list(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        pytest.skip("No checkpoint files found")

    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

    from predict import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(
        checkpoint_path=str(latest_checkpoint), num_classes=20, device=device
    )
    assert model is not None
    assert isinstance(model, YOLOv1)


def main():
    """Original main function for running as a script."""
    print("=" * 70)
    print("YOLO v1 Inference Setup Test")
    print("=" * 70)
    print()

    all_tests_passed = True

    # Test 1: Model creation
    print("Testing model creation...")
    try:
        backbone = ResNetBackbone(pretrained=False, freeze=False)
        model = YOLOv1(backbone=backbone, num_classes=20, S=7, B=2)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        model = None
        all_tests_passed = False

    # Test 2: Inference engine
    if model is not None:
        print("\nTesting inference engine...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inference = YOLOInference(model, device=device)
            print(f"✅ Inference engine created (device: {device})")
        except Exception as e:
            print(f"❌ Failed to create inference engine: {e}")
            inference = None
            all_tests_passed = False
    else:
        inference = None

    # Test 3: Dummy inference
    if inference is not None:
        print("\nTesting inference on dummy image...")
        try:
            dummy_img = Image.new("RGB", (448, 448), color=(128, 128, 128))
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                temp_path = f.name
                dummy_img.save(temp_path)

            detections = inference.predict(
                temp_path, conf_threshold=0.5, nms_threshold=0.4
            )
            Path(temp_path).unlink()

            print(f"✅ Inference ran successfully (found {len(detections)} detections)")
            print("   Note: Random model may not detect anything - this is expected")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback

            traceback.print_exc()
            all_tests_passed = False

    # Test 4: Checkpoint loading
    print("\nTesting checkpoint loading...")
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("⚠️  No checkpoints directory - skipping checkpoint test")
        checkpoint_model = None
    else:
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            print("⚠️  No checkpoint files found - skipping checkpoint test")
            checkpoint_model = None
        else:
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            try:
                from predict import load_model

                device = "cuda" if torch.cuda.is_available() else "cpu"
                checkpoint_model = load_model(
                    checkpoint_path=str(latest_checkpoint),
                    num_classes=20,
                    device=device,
                )
                print(f"✅ Successfully loaded checkpoint: {latest_checkpoint.name}")
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
                import traceback

                traceback.print_exc()
                checkpoint_model = None

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    if all_tests_passed:
        print("✅ All basic tests passed!")
        print("\nYour inference setup is working correctly.")

        if checkpoint_model is not None:
            print("\nYou can now run inference:")
            print(
                "  python predict.py --checkpoint checkpoints/your_model.pth --image test.jpg"
            )
            print("  python examples/inference_demo.py")
        else:
            print("\nTo use inference, you need to train a model first:")
            print("  python train.py --freeze-backbone --epochs 50")
    else:
        print("❌ Some tests failed")
        print("\nPlease check the error messages above and ensure:")
        print("  1. All dependencies are installed: pip install -e .")
        print("  2. PyTorch is properly installed with CUDA support (if using GPU)")
        print("  3. The yolo package can be imported")

    print()


if __name__ == "__main__":
    main()
