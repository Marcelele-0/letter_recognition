import torch
import numpy as np
import torch.nn.functional as F

def calculate_class_weights(labels_onehot):
    if isinstance(labels_onehot, np.ndarray):
        labels_onehot = torch.tensor(labels_onehot, dtype=torch.float32)
    elif not isinstance(labels_onehot, torch.Tensor):
        raise ValueError("Input should be a numpy array or a PyTorch tensor")
    
    if len(labels_onehot.shape) != 2:
        raise ValueError("Input should have 2 dimensions")
    
    if not ((labels_onehot == 0) | (labels_onehot == 1)).all():
        raise ValueError("Input should contain only 0s and 1s")

    if torch.sum(labels_onehot, dim=1).min() == 0:
        raise ValueError("Input should not contain any zero rows")

    labels = torch.argmax(labels_onehot, dim=1)

    class_counts = np.bincount(labels)

    total_samples = np.sum(class_counts)

    epsilon = 1e-9  # Small epsilon value to avoid division by zero
    class_weights = total_samples / (class_counts + epsilon)
    class_weights = class_weights / np.sum(class_weights)

    print("Class imbalance information:")
    for idx, count in enumerate(class_counts):
        class_weight = class_weights[idx]
        class_percent = count / total_samples * 100
        print(f"Class {idx}: {class_percent:.2f}% samples, weight: {class_weight:.6f}")

    return torch.tensor(class_weights, dtype=torch.float32)


def calculate_loss_and_accuracy(outputs, labels):
    # Obliczanie straty za pomocą funkcji cross-entropy
    loss = F.cross_entropy(outputs, torch.argmax(labels, dim=1))

    # Obliczanie dokładności
    _, predicted = torch.max(outputs, 1)
    correct_predictions = (predicted == torch.argmax(labels, dim=1)).sum().item()
    accuracy = correct_predictions / labels.size(0)

    return loss, accuracy
