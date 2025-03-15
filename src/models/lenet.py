import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        activation: type = nn.ReLU,
        conv1_out_channels: int = 6,
        conv2_out_channels: int = 16,
        fc1_units: int = 120,
        fc2_units: int = 84
    ) -> None:
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, conv1_out_channels, kernel_size=5, padding=2),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=5),
            activation(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers will be initialized dynamically
        self.fc_layers = None
        self.num_classes = num_classes
        self.activation = activation

    def _initialize_fc_layers(self, x: torch.Tensor) -> None:
        """
        Initialize fully connected layers based on the actual input size.
        """
        # No need to call the conv layers again here, we already processed them in forward
        self._fc_input_dim = x.numel() // x.shape[0]

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self._fc_input_dim, 120),
            self.activation(),
            nn.Linear(120, 84),
            self.activation(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        """
        # Convolutional layers
        x = self.conv_layers(x)

        # Flatten the tensor to pass it to fully connected layers
        x = torch.flatten(x, 1)

        # Initialize fully connected layers if not already created
        if self.fc_layers is None:
            self._initialize_fc_layers(x)

        x = self.fc_layers(x) # type: ignore
        return x

