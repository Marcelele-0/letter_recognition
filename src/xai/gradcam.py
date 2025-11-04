import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def generate_gradcam(model, image_tensor, target_class=None):
    """
    Generate Grad-CAM heatmap for a given image tensor and model.

    Args:
        model: The pre-trained model.
        image_tensor: The input image tensor (C, H, W).
        target_class: The target class index for which to generate the Grad-CAM. If None, use the predicted class.

    Returns:
        heatmap: The Grad-CAM heatmap.
    """
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError("No Conv2d layer found.")
    
    f_hook = last_conv.register_forward_hook(forward_hook)
    b_hook = last_conv.register_backward_hook(backward_hook)

    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)

    weights = grad.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * act, dim=0)
    cam = F.relu(cam)

    cam = cam.detach().cpu()
    cam = cam.unsqueeze(0).unsqueeze(0)  # 1x1xH xW for interpolation
    cam = F.interpolate(cam, size=(28, 28), mode='bilinear', align_corners=False)
    cam = cam.squeeze().numpy()
    cam -= cam.min()
    cam /= cam.max()

    # Usuwamy hooki
    f_hook.remove()
    b_hook.remove()
    return cam

def show_gradcam(image_tensor, heatmap, title="Grad-CAM"):
    img = image_tensor.squeeze().cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.show()
