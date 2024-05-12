from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.evaluation_metrics import calculate_loss_and_accuracy
from letter_recognition.utils.custom_data_loader import CustomDataLoader
from letter_recognition.model.lenet5 import LeNet5
import os

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
print(class_weights)


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
    # Training loop
    train_loss, train_acc = 0.0, 0.0
    model.train()  # Ustawiamy model w tryb trenowania
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Obliczanie funkcji straty i dokładności na zbiorze treningowym
        loss, accuracy = calculate_loss_and_accuracy(outputs, labels)
        
        # Sumowanie straty i dokładności
        train_loss += loss.item()
        train_acc += accuracy
        
        # Backward pass i optymalizacja
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Obliczanie średniej straty i dokładności na zbiorze treningowym
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    # Validation loop
    val_loss, val_acc = 0.0, 0.0
    model.eval()  # Ustawiamy model w tryb ewaluacji
    with torch.no_grad():  # Wyłączamy obliczanie gradientów podczas walidacji
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Obliczanie funkcji straty i dokładności na zbiorze walidacyjnym
            loss, accuracy = calculate_loss_and_accuracy(outputs, labels)
            
            # Sumowanie straty i dokładności
            val_loss += loss.item()
            val_acc += accuracy
    
    # Obliczanie średniej straty i dokładności na zbiorze walidacyjnym
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    # Wyświetlenie wyników
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.5f}, Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.5f}')


print ("Trained the model")

# Save the model
model_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(model_dir, '../model/lenet5_weight.pth')

torch.save(model.state_dict(), model_save_path)
print ("Saved the model to: ", model_save_path)
