from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.evaluation_metrics import calculate_loss_and_accuracy
from letter_recognition.utils.custom_data_loader import CustomDataLoader
from letter_recognition.model.lenet5 import LeNet5

import torch

print ("Imported all the required modules")


# Load the data
data = CustomDataLoader(batch_size=64)
train_loader, val_loader = data.get_data_loader()
print ("Loaded the data")

# Calculate class weights
class_weights = calculate_class_weights(data.train_dataset.tensors[1])
print ("Calculated class weights")

# Initialize the model
model = LeNet5()
print ("Initialized the model")

# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print ("Defined the loss function and optimizer")

# Train the model


