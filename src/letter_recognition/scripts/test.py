import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import string
from letter_recognition.model.lenet5 import LeNet5  # Importing LeNet5 model

# Function to load images from a numpy file
def load_image(image_path):
    image = np.load(image_path)
    return image

# Function to map indices to letters of the alphabet
def index_to_letter(index):
    alphabet = string.ascii_uppercase  # Changed to uppercase as LeNet5 likely trained with uppercase letters
    return alphabet[index]

# Function to display the prediction of the model for a given image
def display_prediction(model, image):
    # Transform the image to a tensor and normalize it
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Make predictions using the model
    with torch.no_grad():
        model.eval()
        output = model(image)
        predicted_label = torch.argmax(output, dim=1).item()
    
    # Display the image and predicted label
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f'Predicted Label: {predicted_label} ({index_to_letter(predicted_label)})')
    plt.show()

# Path to the trained model
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, '../model/lenet5_weight.pth')

# Load the trained model
model = LeNet5()
model.load_state_dict(torch.load(model_path))
model.eval()

# Path to the directory containing test data
test_data_path = os.path.join(model_dir, '../data/test/X_test.npy')

# Load the test data
test_data = np.load(test_data_path)

#   predictions for couple random different images from the test set

for i in range(5):
    random_index = np.random.randint(0, len(test_data))
    image = test_data[random_index]
    display_prediction(model, image)