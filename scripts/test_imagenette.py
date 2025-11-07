"""Test script for Imagenette DataModule."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modern_vit.data.imagenette_datamodule import ImagenetteDataModule


def test_imagenette_datamodule():
    """Test Imagenette DataModule initialization and data loading."""
    print("Testing Imagenette DataModule...")

    # Initialize datamodule
    datamodule = ImagenetteDataModule(
        data_dir="data/imagenette",
        image_size=224,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )

    print("✓ DataModule initialized")

    # Prepare data (download if needed)
    print("Preparing data (this may take a while on first run)...")
    datamodule.prepare_data()
    print("✓ Data prepared")

    # Setup datasets
    print("Setting up datasets...")
    datamodule.setup(stage="fit")
    print("✓ Datasets setup")

    # Check dataset sizes
    print("\nDataset sizes:")
    print(f"  Train: {len(datamodule.train_dataset)} samples")
    print(f"  Validation: {len(datamodule.val_dataset)} samples")

    # Get a batch
    print("\nTesting data loading...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch

    print("✓ Batch loaded successfully")
    print(f"  Batch shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  Unique labels in batch: {labels.unique().tolist()}")

    # Validate image shape
    assert images.shape == (4, 3, 224, 224), f"Unexpected image shape: {images.shape}"
    assert labels.shape == (4,), f"Unexpected labels shape: {labels.shape}"

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_imagenette_datamodule()
