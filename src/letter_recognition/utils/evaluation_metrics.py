import torch
import numpy as np
import torch.nn.functional as F

def calculate_class_weights(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().detach().numpy()

    elif not isinstance(labels, (np.ndarray, list, tuple)):
        raise ValueError("Unsupported data type for labels. Supported types: numpy array, list, tuple.")

    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    if labels.size == 0:
        raise ValueError("Empty input labels array.")
    
    if labels.dtype == object:
        try:
            labels = labels.astype(np.float)
        except ValueError:
            raise ValueError("Labels array contains non-numeric values that cannot be converted to float.")

    class_counts = np.bincount(labels)

    if len(class_counts) == 1:
        # Single class present in the labels
        return torch.tensor([1.0])  # Return as torch tensor

    total_samples = np.sum(class_counts)

    epsilon = 1e-7  # Small epsilon value to avoid division by zero
    class_weights = total_samples / (class_counts + epsilon)
    class_weights = class_weights / np.sum(class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float32)

def calculate_loss_and_accuracy(outputs, labels):
    # Obliczanie straty za pomocą funkcji cross-entropy
    loss = F.cross_entropy(outputs, torch.argmax(labels, dim=1))

    # Obliczanie dokładności
    _, predicted = torch.max(outputs, 1)
    correct_predictions = (predicted == torch.argmax(labels, dim=1)).sum().item()
    accuracy = correct_predictions / labels.size(0)

    return loss, accuracy