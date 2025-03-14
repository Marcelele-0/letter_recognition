import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """
    A simple LeNet implementation for MNIST number classification.
    """

    def __init__(
            self,
            num_classes: int = 10,
            activation: type = nn.ReLU,  
    ) -> None:
        """
        Initialize the LeNet model.

        Args:
            num_classes (int): Number of classes in the dataset. Defaults to 10.
            activation (type): Activation function class to use (e.g., nn.ReLU). Defaults to nn.ReLU.
        """
        
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),  # Padding to maintain dimensions
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Placeholder for dynamically computing the input to FC layers
        self._fc_input_dim = None

        # Fully connected layers (initialized later)
        self.fc_layers = None

        # Number of classes in the model
        self.num_classes = num_classes
        self.activation = activation  # Save the activation class for later use

    def _initialize_fc_layers(self, x: torch.Tensor) -> None:
        """
        Initializes the fully connected layers based on the actual input size.

        Args:
            x (torch.Tensor): Input sample to the convolutional layers.
        """
        x = self.conv_layers(x)
        self._fc_input_dim = x.numel() // x.shape[0]  # Size of the flattened input
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self._fc_input_dim, 120),
            self.activation(),
            nn.Linear(120, 84),
            self.activation(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LeNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)

        if self.fc_layers is None:
            self._initialize_fc_layers(x)  # Initialize FC layers if they haven't been created yet

        x = self.fc_layers(x)
        return x
