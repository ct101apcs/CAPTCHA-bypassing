import torch
import cv2
import numpy as np
import random

def shuffle(img):
    n_parts = 7
    img = img.numpy()
    height, width = img.shape[:2]

    part_width = width // n_parts

    parts = [img[:, i*part_width : (i+1)*part_width] for i in range(n_parts)]

    if width % n_parts != 0:
        parts.append(img[:, n_parts*part_width :])

    random.shuffle(parts)

    shuffled_img = np.hstack(parts)
    return torch.from_numpy(shuffled_img)
