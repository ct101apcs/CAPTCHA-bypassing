import numpy as np
import torch
import random
import cv2

def rotation(img, angle=30):
    # Convert torch tensor [C, H, W] to [H, W, C] numpy
    img = img.permute(1, 2, 0).numpy()

    # Assure que les valeurs sont entre 0 et 255 et de type uint8
    img = (img * 255).astype(np.uint8)

    # Compute center and rotation matrix
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply rotation using warpAffine
    rotated_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Convert back to torch tensor [C, H, W] and normalize to [0,1]
    rotated_img = torch.from_numpy(rotated_img).permute(2, 0, 1).float() / 255.0

    return rotated_img
