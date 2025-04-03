from typing import Tuple

import torch 
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, MeanMetric, MaxMetric


class MNISTLitModule(pl.LightningModule):
    """PyTorch Lightning Module for MNIST classification."""

    def __init__(
        self,
        model: torch.nn.Module,
        compile_model: bool = False,
    ) -> None:
        """
        Initializes the module.

        Args:
            model: Neural network model.
            compile_model: Whether to use `torch.compile()`.
        """
        super().__init__()

        # Save the hyperparameters
        self.save_hyperparameters(logger=False)

        # Save the model and optimizer
        self.model = model
        self.compile_model = compile_model

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics to track
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        self.train_precision = Precision(task="multiclass", num_classes=10, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=10, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=10, average="macro")

        self.train_recall = Recall(task="multiclass", num_classes=10, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=10, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=10, average="macro")

        # Loss to track
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

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
        loss = self.criterion(logits, y)

        # Compute the predictions
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Single training step."""
        loss, preds, targets = self.model_step(batch)

        self.train_loss.update(loss)
        self.train_acc.update(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", self.train_precision(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", self.train_recall(preds, targets), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Single validation step."""
        loss, preds, targets = self.model_step(batch)

        # Log the metrics
        self.val_loss.update(loss)
        self.val_acc.update(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall(preds, targets), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Single test step."""
        loss, preds, targets = self.model_step(batch)

        # Log the metrics
        self.test_loss.update(loss)
        self.test_acc.update(preds, targets)

        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision(preds, targets), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall(preds, targets), on_step=False, on_epoch=True, prog_bar=True)

    def on_train_start(self) -> None:
        """Resets validation and training metrics at the start of training."""
        self.train_loss.reset()
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def on_validation_epoch_end(self) -> None:
        """Updates the best validation accuracy at the end of each epoch."""
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """Compiles the model if required."""
        if self.compile_model and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.Adam( 
            self.model.parameters(), # type: ignore[call-arg]
            lr=1e-3,
            weight_decay=1e-4 ) 

        # Define the scheduler (StepLR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Return optimizer and scheduler
        return [optimizer], [scheduler]
