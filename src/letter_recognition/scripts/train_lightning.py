from letter_recognition.utils.evaluation_metrics import calculate_class_weights
from letter_recognition.utils.custom_data_loader import CustomDataLoader
from letter_recognition.model.lenet5_lightning import LeNet5  # Upewnij się, że ten plik jest poprawnie zaimportowany
import os
import torch
import pytorch_lightning as pl

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
model = LeNet5(class_weights=class_weights)
print("Initialized the model")

# Define the trainer
trainer = pl.Trainer(
    max_epochs=15,
    gpus=1 if torch.cuda.is_available() else 0,  # Użyj GPU jeśli dostępne
    log_every_n_steps=10
)

# Train the model
trainer.fit(model, train_loader, val_loader)

# Save the model
model_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(model_dir, "../model/lenet5_weight.pth")
torch.save(model.state_dict(), model_save_path)
print("Saved the model to: ", model_save_path)
