import torch
import numpy as np
from PIL import Image, ImageFilter
from ..utils import pil_to_tensor, tensor_to_pil

def gaussianNoise(img, stddev=0.1):
    # Convert PIL Image to tensor if needed
    if not isinstance(img, torch.Tensor):
        img = pil_to_tensor(img)
    
    # Convert to numpy and ensure proper shape
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Generate Gaussian noise
    noise = np.random.normal(0, stddev, img_np.shape)
    
    # Add noise and clip to valid range
    noisy_img = np.clip(img_np + noise, 0, 1)
    
    # Convert back to tensor
    noisy_tensor = torch.from_numpy(noisy_img).permute(2, 0, 1).float()
    
    # Convert back to PIL Image
    return tensor_to_pil(noisy_tensor) 