from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.evaluation_metrics import calculate_loss_and_accuracy
from letter_recognition.utils.custom_data_loader import CustomDataLoader
from letter_recognition.model.lenet5 import LeNet5

import torch

print ("Imported all the required modules")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("Set the device to: ", device)

# Load the data
data = CustomDataLoader(batch_size=64)
train_loader, val_loader = data.get_data_loader()
print ("Loaded the data")

# Calculate class weights
class_weights = calculate_class_weights(data.train_dataset.tensors[1])
class_weights = class_weights.to(device)
print("Calculated class weights") 

# Initialize the model
model = LeNet5()
model = model.to(device)
print ("Initialized the model")

# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print ("Defined the loss function and optimizer")

num_epochs = 10


# Train the model
for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        
        # Obliczanie funkcji straty z uwzględnieniem wag klas
        loss = torch.nn.functional.cross_entropy(outputs, labels, weight=class_weights)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
