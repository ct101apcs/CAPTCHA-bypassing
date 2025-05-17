import cv2
import numpy as np
import torch

def gaussianNoise(img):
    img = img.numpy()
    image_float = img.astype(np.float32)

    # Define noise parameters
    mean = 0
    stddev = 0.5

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image_float.shape)

    # Add noise and clip the result to valid range [0,1]
    noisy_image = np.clip(image_float + noise, 0, 1)

    return torch.from_numpy(noisy_image).float()

# in main, use:
# ax.imshow(gaussianNoise(dataloader.dataset[list_img[i]][0].permute(1, 2, 0)))