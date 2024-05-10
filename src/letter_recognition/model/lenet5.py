import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

import io
from contextlib import redirect_stdout

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 1x28x28 -> 6x32x32
        self.padding = nn.ZeroPad2d(padding=(2, 2, 2, 2))

        # 6x32x32 -> 6x16x16 
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)

        # 6x16x16 -> 16x5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # 16x5x5 -> 120
        self.fc1 = nn.Linear(16*5*5, 120)

        # 120 -> 84
        self.fc2 = nn.Linear(120, 84)

        # 84 -> 26
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = F.relu(self.conv1(self.padding(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    # Static method to get the model summary
    @staticmethod
    def get_summary():
        model = LeNet5()
        f = io.StringIO()
        with redirect_stdout(f):
            summary(model, input_size=(1, 1, 28, 28), col_width=20,
                        col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'], verbose=2)
        return f.getvalue()


