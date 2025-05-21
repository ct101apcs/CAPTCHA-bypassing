import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

def pil_to_tensor(pil_image):
    """Convert PIL Image to PyTorch tensor (CHW, float in [0, 1])"""
    return TF.to_tensor(pil_image)

def tensor_to_pil(tensor):
    """Convert PyTorch tensor (CHW, float in [0, 1]) to PIL Image"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.clamp(0, 1)
        return TF.to_pil_image(tensor)
    return tensor 