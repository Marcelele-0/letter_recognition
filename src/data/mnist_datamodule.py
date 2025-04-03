from typing import Optional, Tuple
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class MNISTDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the MNIST dataset.

    Args:
        data_dir (str): Directory to save/load the data.
        train_val_test_split (Tuple[int, int, int]): Sizes for training, validation, and test splits.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory in data loaders.
    """
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # Save hyperparameters for logging and checkpoints
        self.save_hyperparameters(logger=False)

        # For Pylance type checking
        self.data_dir = data_dir
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transforms = transforms.Compose([
            transforms.RandomRotation(10),  
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download the MNIST dataset if not already downloaded."""
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split the dataset into training, validation, and test sets.

        Args:
            stage (Optional[str]): Stage to set up ('fit', 'test', etc.). Defaults to None.
        """
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.data_dir, train=True, transform=self.train_transforms)
            testset = MNIST(self.data_dir, train=False, transform=self.test_transforms)

            # Ensure the split sizes match the dataset size
            total_train_samples = len(trainset)
            train_val_split = self.train_val_test_split[:2]
            if sum(train_val_split) != total_train_samples:
                train_val_split = (int(total_train_samples * 0.9), int(total_train_samples * 0.1))

            # Split trainset into training and validation sets
            self.data_train, self.data_val = random_split(
                dataset=trainset,
                lengths=train_val_split,
                generator=torch.Generator().manual_seed(42),
            )

            # Use testset directly for testing
            self.data_test = testset

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        if self.data_train is None:
            raise ValueError("data_train is None. Did you run MnistDataModule.setup()?")
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        if self.data_val is None:
            raise ValueError("data_val is None. Did you run MnistDataModule.setup()?")
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        if self.data_test is None:
            raise ValueError("data_test is None. Did you run MnistDataModule.setup()?")
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )

