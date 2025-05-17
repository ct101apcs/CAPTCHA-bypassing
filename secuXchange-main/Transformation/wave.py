import torch
import cv2
import numpy as np

def wave(img):
    img = img.numpy()
    rows, cols = img.shape[:2]

    map_y, map_x = np.indices((rows, cols), dtype=np.float32)
    map_x = map_x + 5.0 * np.sin(map_y / 20.0)

    # Apply modification
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(distorted)
