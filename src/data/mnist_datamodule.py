from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(LightningDataModule):
    """
    LightningDataModule for the MNIST dataset.

    Args:
        data_dir (str): Directory to save/load the data.
        train_val_test_split (list[int]): Split sizes for training, validation, and test sets.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory in data loaders.
    """
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: list[int] = [55000, 5000, 10000],
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download the MNIST dataset if not already downloaded."""
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            stage (Optional[str]): Stage to set up (fit, test, etc.).
        """
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        """Return the training data loader."""
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation data loader."""
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test data loader."""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return the prediction data loader."""
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

