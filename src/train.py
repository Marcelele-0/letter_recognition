import os
import pytorch_lightning as pl
from models.lenet import LeNet5
from data.mnist_datamodule import MNISTDataModule
from models.mnist_module import MNISTLitModule
from utils.split_data import calculate_split_sizes
from omegaconf import DictConfig
from omegaconf import OmegaConf
import hydra
import importlib
from pytorch_lightning.loggers import WandbLogger 
import wandb

@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):
    # Hyperparameters and configurations for data module
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    pin_memory = cfg.data.pin_memory
    
    # Split ratios
    split_ratios = cfg.data.train_val_test_split
    train_val_test_split = calculate_split_sizes(70_000, split_ratios)

    # Initialize the data module
    data_module = MNISTDataModule(
        train_val_test_split=train_val_test_split,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Dynamically import the activation function
    activation_module, activation_class = cfg.models.model.activation.rsplit('.', 1)
    activation = getattr(importlib.import_module(activation_module), activation_class)

    # Initialize the model
    model = LeNet5(
        input_channels=cfg.models.model.input_channels,
        num_classes=cfg.models.model.num_classes,
        activation=activation,
        conv1_out_channels=cfg.models.model.conv1_out_channels,
        conv2_out_channels=cfg.models.model.conv2_out_channels,
        fc1_units=cfg.models.model.fc1_units,
        fc2_units=cfg.models.model.fc2_units
    )

    # Initialize W&B logger
    wandb_logger = WandbLogger(        
        project=cfg.logger.project,
        name=cfg.logger.run_name,
        save_dir=cfg.logger.save_dir,
        log_model=cfg.logger.log_model
    )

    # Initialize the Lightning module
    lightning_model = MNISTLitModule(model, compile_model=cfg.models.compile)

    # Define the trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=cfg.trainer.max_epochs,
        logger=wandb_logger,
    )

    # logging hyperparameters to W&B
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict")
    )  

    # Train the model
    trainer.fit(lightning_model, datamodule=data_module)

    # Test the model
    trainer.test(lightning_model, datamodule=data_module)

    # Zamknięcie sesji W&B
    wandb.finish()


if __name__ == "__main__":
    main()
