import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

class CustomDataLoader:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size

    def load_data(self):
        # Get the directory that this script is in
        src_lr_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct absolute paths to the data files
        X_train_path = os.path.join(src_lr_dir, '../data/train/X_train_weight.npy')
        y_train_path = os.path.join(src_lr_dir, '../data/train/y_train_weight.npy')
        X_val_path = os.path.join(src_lr_dir, '../data/val/X_val.npy')
        y_val_path = os.path.join(src_lr_dir, '../data/val/y_val.npy')

        # Load data from .npy files
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        X_val = np.load(X_val_path)
        y_val = np.load(y_val_path)

        print(f'X_train shape: {X_train.shape}')

        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # add an channel dimension to the images
        X_train_tensor = X_train_tensor.unsqueeze(1)
        X_val_tensor = X_val_tensor.unsqueeze(1)
        
        

        # Create TensorDataset for training and validation data
        self.train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        self.val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    def get_data_loader(self):
        if not hasattr(self, 'train_dataset') or not hasattr(self, 'val_dataset'):
            self.load_data()

        # Create DataLoader for training and validation data
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        return train_loader, val_loader
    



