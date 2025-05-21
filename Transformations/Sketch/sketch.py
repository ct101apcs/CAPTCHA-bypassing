import numpy as np
import cv2
import torch
from ..utils import pil_to_tensor, tensor_to_pil


def sketch(img):
    # Convert PIL Image to tensor
    if not isinstance(img, torch.Tensor):
        img = pil_to_tensor(img)

    img = img.permute(1, 2, 0).cpu().numpy()

    # Ensure image is in 8-bit format
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert grayscale image
    inv_gray = 255 - gray_img

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)

    # Invert the blurred image
    inv_blurred = 255 - blurred

    # Blend the grayscale image with the inverted blurred image to create sketch
    sketch = cv2.divide(gray_img, inv_blurred, scale=256.0)

    # Convert back to RGB
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    # Convert to tensor and normalize
    final_img = torch.from_numpy(sketch_rgb).permute(2, 0, 1).float() / 255.0

    # Clamp to [0, 1] just in case
    final_img = final_img.clamp(0, 1)

    # Convert back to PIL Image
    return tensor_to_pil(final_img) 