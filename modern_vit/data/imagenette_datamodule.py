"""Imagenette DataModule for PyTorch Lightning using HuggingFace datasets."""

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms


class ImagenetteDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for HuggingFace Imagenette dataset.

    Args:
        hf_dataset: HuggingFace dataset
        transform: Image transforms to apply
    """

    def __init__(self, hf_dataset: Dataset, transform: transforms.Compose | None = None) -> None:
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, label)
        """
        sample = self.hf_dataset[idx]
        image = sample["image"]
        label = sample["label"]

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        return image, label


class ImagenetteDataModule(LightningDataModule):
    """DataModule for Imagenette dataset using HuggingFace datasets.

    Imagenette is a subset of ImageNet with 10 classes, designed for
    faster experimentation. This DataModule uses the HuggingFace datasets
    library to load the frgfm/imagenette dataset.

    Args:
        data_dir: Path to cache directory for datasets (optional)
        image_size: Target image size (default: 224)
        batch_size: Batch size for data loaders (default: 32)
        num_workers: Number of worker processes for data loading (default: 4)
        pin_memory: Whether to pin memory in data loaders (default: True)
        train_transforms: List of transform names for training (default: None)
        val_transforms: List of transform names for validation (default: None)
        mean: Mean values for normalization (default: ImageNet stats)
        std: Std values for normalization (default: ImageNet stats)
        dataset_config: Dataset configuration name ('160px', '320px', or 'full_size') (default: '320px')
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_transforms: list[str] | None = None,
        val_transforms: list[str] | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        dataset_config: str = "320px",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir) if data_dir else None
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_config = dataset_config

        # Default normalization values (ImageNet stats)
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        # Default transforms
        self.train_transform_names = train_transforms or [
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "ColorJitter",
            "ToTensor",
            "Normalize",
        ]
        self.val_transform_names = val_transforms or [
            "Resize",
            "CenterCrop",
            "ToTensor",
            "Normalize",
        ]

        # Build transforms
        self.train_transform = self._build_transforms(self.train_transform_names)
        self.val_transform = self._build_transforms(self.val_transform_names)

        # Datasets (initialized in setup)
        self.train_dataset: ImagenetteDataset | None = None
        self.val_dataset: ImagenetteDataset | None = None

    def _build_transforms(self, transform_names: list[str]) -> transforms.Compose:
        """Build transforms from a list of transform names.

        Args:
            transform_names: List of transform names

        Returns:
            Composed transforms
        """
        transform_list: list[Any] = []

        for name in transform_names:
            if name == "RandomResizedCrop":
                transform_list.append(
                    transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0))
                )
            elif name == "RandomHorizontalFlip":
                transform_list.append(transforms.RandomHorizontalFlip())
            elif name == "ColorJitter":
                transform_list.append(
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
                )
            elif name == "Resize":
                # Resize to slightly larger than target size for center crop
                transform_list.append(transforms.Resize(int(self.image_size * 1.15)))
            elif name == "CenterCrop":
                transform_list.append(transforms.CenterCrop(self.image_size))
            elif name == "ToTensor":
                transform_list.append(transforms.ToTensor())
            elif name == "Normalize":
                transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
            else:
                raise ValueError(f"Unknown transform: {name}")

        return transforms.Compose(transform_list)

    def prepare_data(self) -> None:
        """Download and prepare data if needed.

        This method is called only from a single process.
        """
        # Load dataset to trigger download if needed
        cache_dir = str(self.data_dir) if self.data_dir else None
        load_dataset(
            "frgfm/imagenette",
            name=self.dataset_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training and validation.

        Args:
            stage: Stage name ('fit', 'validate', 'test', or 'predict')
        """
        cache_dir = str(self.data_dir) if self.data_dir else None

        if stage in ("fit", None):
            dataset = load_dataset(
            "frgfm/imagenette",
            name=self.dataset_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
            )

            # Create PyTorch datasets
            self.train_dataset = ImagenetteDataset(
            hf_dataset=dataset["train"],
            transform=self.train_transform,
            )

            self.val_dataset = ImagenetteDataset(
            hf_dataset=dataset["validation"],
            transform=self.val_transform,
            )
        elif stage in ("validate", "test"):
            dataset = load_dataset(
            "frgfm/imagenette",
            name=self.dataset_config,
            cache_dir=cache_dir,
            trust_remote_code=True,
            )
            self.val_dataset = ImagenetteDataset(
            hf_dataset=dataset["validation"],
            transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Create training data loader.

        Returns:
            Training data loader
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader.

        Returns:
            Validation data loader
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader.

        Returns:
            Test data loader (uses validation dataset)
        """
        return self.val_dataloader()
