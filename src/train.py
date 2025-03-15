import pytorch_lightning as pl
from models.lenet import LeNet5
from data.mnist_datamodule import MNISTDataModule
from models.mnist_module import MNISTLitModule

# TODO: Add Hydra for configuration management
def main():
    # Hyperparameters and configurations
    data_dir = "data/"
    batch_size = 64
    num_workers = 4
    pin_memory = True

    # Initialize the data module
    data_module = MNISTDataModule(
        data_dir=data_dir,
        train_val_test_split=(55_000, 5_000, 10_000),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Initialize the model
    model = LeNet5(input_channels=1, num_classes=10)

    # Initialize the Lightning module
    lightning_model = MNISTLitModule(model)

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=10,
        devices=1, 
        logger=False,  

    )

    # Train the model
    trainer.fit(lightning_model, datamodule=data_module)

    # Test the model
    trainer.test(lightning_model, datamodule=data_module)

if __name__ == "__main__":
    main()
