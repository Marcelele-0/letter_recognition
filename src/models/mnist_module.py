import lightning as pl
import torch
import torch.nn as nn
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from typing import Optional, Tuple



class MNISTModule(pl.LightningModule):
    """
    LightningModule for MNIST classification

    A 'LightningModule' is a subclass of 'nn.Module' that provides additional
    functionality to streamline the training process. It includes the following
    methods:

    - setup: defines the data loaders
    - forward: defines the forward pass of the model
    - training_step: defines the training step
    - validation_step: defines the validation step
    - test_step: defines the test step
    """

    def __init__(
            self,
            model,
            optimizer = torch.optim.Adam,
            lr: float = 0.001,
            loss_fn = nn.CrossEntropyLoss,
            compile: bool = True
    ) -> None:
        """
        Initializes the MNISTModule.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use for training. Defaults to Adam.
            lr (float): The learning rate for the optimizer. Defaults to 0.001.
            compile (bool): Whether to compile the model. Defaults to True.
        """
        super().__init__()

        # Save the hyperparameters
        self.save_hyperparameters(logger=False)

        # Save the model and optimizer
        self.model = model

        # Save the loss function
        self.loss_fn = loss_fn

        # Metrics to track
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # Loss to track
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Optimizer
        self.val_acc_max = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input data

        Returns:
            torch.Tensor: Model output
        """
        return self.model(x)
    

    def model_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a single model step for a batch of data.

        params:
         - batch: A tuple containing the input data and labels

        returns:
         - Tuple of predictions, loss, and labels in that order
        """
        # Unpack the batch
        x, y = batch

        # Forward pass
        logits = self.model(x)

        # Compute the loss
        loss = self.loss_fn(logits, y)

        # Compute the predictions
        preds = torch.argmax(logits, dim=1)

        return preds,loss, y

    def training_step(
            self, 
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> torch.Tensor:
        """
        Defines a single training step on a batch of data.

        params:
         - batch: A tuple containing the input data and labels
         - batch_idx: The index of the batch

        returns:
         - The loss for the batch - Tensor 
        """
        
        preds, loss, targets = self.model_step(batch)

        # Log the metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)

        self.log("train_loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> None:
        """
        Defines a single validation step on a batch of data.

        params:
         - batch: A tuple containing the input data and labels
         - batch_idx: The index of the batch
        """

        preds, loss, targets = self.model_step(batch)

        # Log the metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        
        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
        ) -> None:
        """
        Defines a single test step on a batch of data.

        params:
         - batch: A tuple containing the input data and labels
         - batch_idx: The index of the batch
        """

        preds, loss, targets = self.model_step(batch)

        # Log the metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)

        self.log("test_loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    def setup(self, stage) -> None:
        """
        Set up the data loaders for the model.

        params:
         - stage: The stage of training (fit, validate, test, predict)
        """
        if stage == "fit" and compile:
            self.model = torch.compile(self.model)

    def configure_optimizers(self):
        # TODO: Add learning rate scheduler
        pass

    # TODO: add lightning hooks on stages end