from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.evaluation_metrics import calculate_loss_and_accuracy
from letter_recognition.utils.custom_data_loader import CustomDataLoader
from letter_recognition.model.lenet5 import LeNet5
import os

import torch
import torch.nn as nn

print("Imported all the required modules")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Set the device to: ", device)

# Load the data
data = CustomDataLoader(batch_size=64)
train_loader, val_loader = data.get_data_loader()
print("Loaded the data")

# Calculate class weights
class_weights = calculate_class_weights(data.train_dataset.tensors[1])
class_weights = class_weights.to(device)

# Initialize the model
model = LeNet5()
model = model.to(device)
print("Initialized the model")

# Define the loss function with class weights and L2 regularization
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)  # Dodanie regularyzacji L2
print("Defined the loss function and optimizer")

num_epochs = 15  # Zwiększenie liczby epok

# Train the model
for epoch in range(num_epochs):
    # Training loop
    train_loss, train_acc = 0.0, 0.0
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss and accuracy on the training set
        loss = criterion(outputs, labels)
        accuracy = calculate_loss_and_accuracy(outputs, labels)[1]

        # Add L2 regularization to the loss
        l2_reg = sum(torch.norm(param) for param in model.parameters())  # Suma norm kwadratów wag
        loss += 0.001 * l2_reg  # Współczynnik lambda dla regularyzacji L2
        # Sum the loss and accuracy
        train_loss += loss.item()
        train_acc += accuracy

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Calculate average loss and accuracy on the training set
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    # Validation loop
    val_loss, val_acc = 0.0, 0.0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation during validation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss and accuracy on the validation set
            loss = criterion(outputs, labels)
            accuracy = calculate_loss_and_accuracy(outputs, labels)[1]

            # Sum the loss and accuracy
            val_loss += loss.item()
            val_acc += accuracy

    # Calculate average loss and accuracy on the validation set
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    # Display the results
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}"
    )

print("Trained the model")

# Save the model
model_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(model_dir, "../model/lenet5_weight.pth")

torch.save(model.state_dict(), model_save_path)
print("Saved the model to: ", model_save_path)
