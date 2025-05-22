import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

def pil_to_tensor(pil_img):
    """Convert PIL Image to PyTorch tensor (CHW, float in [0, 1])"""
    if isinstance(pil_img, Image.Image):
        return TF.to_tensor(pil_img)
    return pil_img

def tensor_to_pil(tensor):
    """Convert PyTorch tensor (CHW, float in [0, 1]) to PIL Image"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.clamp(0, 1)
        return TF.to_pil_image(tensor)
    return tensor

def apply_transformation(img, transform_func, **params):
    """Apply transformation to image, handling PIL Image to tensor conversion"""
    # Convert PIL Image to tensor if needed
    tensor = pil_to_tensor(img)
    
    # Apply transformation
    transformed = transform_func(tensor, **params)
    
    # Convert back to PIL Image
    return tensor_to_pil(transformed) 