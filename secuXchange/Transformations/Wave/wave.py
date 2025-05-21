import torch
import cv2
import numpy as np

import cv2
import numpy as np
import torch


def wave(img, intensity=10, frequency=20.0):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.shape[0] == 3:  # RGB
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

    img = (img * 255).astype(np.uint8)

    rows, cols = img.shape[:2]
    map_y, map_x = np.indices((rows, cols), dtype=np.float32)
    map_x = map_x + intensity * np.sin(map_y / frequency)

    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    distorted = distorted.astype(np.float32) / 255.0
    distorted = np.transpose(distorted, (2, 0, 1))  # (3, H, W)
    return torch.from_numpy(distorted)
