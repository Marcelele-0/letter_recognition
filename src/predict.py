import torch
from torchvision import transforms
from PIL import Image, ImageOps
import os
import random
import matplotlib.pyplot as plt
from models.mnist_module import MNISTLitModule
from models.lenet import LeNet5
import torch.serialization
from xai.gradcam import generate_gradcam, show_gradcam
from torch.nn import Sequential, Conv2d, Linear, ReLU, Flatten, BatchNorm2d, MaxPool2d, Dropout

# Fix for OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CHECKPOINT_PATH = "wandb_logs/MNIST-Training/h57gb7f6/checkpoints/epoch=9-step=8440.ckpt"
DATA_DIR = "data/my_data/"

def load_model(checkpoint_path):
    """Load the trained model from a checkpoint."""
    torch.serialization.add_safe_globals([LeNet5, Sequential, Conv2d, Linear, ReLU, Flatten, BatchNorm2d, MaxPool2d, Dropout])
    model = MNISTLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False
    )
    model.eval()
    return model

def get_transform():
    """Define the image transformations."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(lambda img: ImageOps.autocontrast(img)),
        transforms.Resize((28, 28)),
        transforms.Lambda(lambda img: ImageOps.invert(img)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def load_images(data_dir, num_samples=10):
    """Load and randomly select a specified number of images from the dataset."""
    all_images = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                all_images.append((os.path.join(label_path, img_name), int(label)))
    random.shuffle(all_images)
    return all_images[:num_samples]

def predict_and_visualize(model, transform, images, all_images):
    """Make predictions on all images, calculate accuracy, and visualize a random sample."""
    correct = 0
    total = 0

    # Calculate accuracy on the entire dataset
    for img_path, true_label in all_images:
        image = Image.open(img_path).convert("L")
        augmented_image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(augmented_image)
            predicted_label = output.argmax(dim=1).item()

        # Update accuracy counters
        total += 1
        if predicted_label == true_label:
            correct += 1

    # Calculate accuracy
    accuracy = correct / total * 100

    # Display predictions for a random sample of 10 images
    fig, axes = plt.subplots(5, 2, figsize=(8, 12))
    for idx, (img_path, true_label) in enumerate(images):
        image = Image.open(img_path).convert("L")
        augmented_image = transform(image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(augmented_image)
            predicted_label = output.argmax(dim=1).item()

        # Display the image and prediction
        ax = axes[idx // 2, idx % 2]
        ax.imshow(augmented_image.squeeze(0).squeeze(0), cmap="gray")
        ax.set_title(f"True: {true_label}, Pred: {predicted_label}")
        ax.axis("off")

    fig.suptitle(f"Accuracy on entire dataset: {accuracy:.2f}%", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def visualize_gradcam_samples(model, transform, all_images, num_samples=6):
    """Generate Grad-CAM visualizations for a few random samples."""
    selected_samples = random.sample(all_images, num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for idx, (img_path, true_label) in enumerate(selected_samples):
        image = Image.open(img_path).convert("L")
        augmented = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(augmented)
            pred_label = output.argmax(dim=1).item()

        heatmap = generate_gradcam(model, augmented, target_class=None)

        # Left: original
        axes[idx][0].imshow(augmented.squeeze().cpu(), cmap="gray")
        axes[idx][0].set_title(f"True: {true_label}, Pred: {pred_label}")
        axes[idx][0].axis("off")

        # Right: Grad-CAM overlay
        axes[idx][1].imshow(augmented.squeeze().cpu(), cmap="gray")
        axes[idx][1].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[idx][1].set_title("Grad-CAM")
        axes[idx][1].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    """Main function to load the model, process images, and make predictions."""
    model = load_model(CHECKPOINT_PATH)
    transform = get_transform()
    all_images = load_images(DATA_DIR, num_samples=None)  # Load all images
    sample_images = random.sample(all_images, 10)  # Select a random sample of 10 images

    predict_and_visualize(model, transform, sample_images, all_images)

    # Now show multiple Grad-CAM visualizations
    visualize_gradcam_samples(model, transform, all_images, num_samples=6)

if __name__ == "__main__":
    main()
