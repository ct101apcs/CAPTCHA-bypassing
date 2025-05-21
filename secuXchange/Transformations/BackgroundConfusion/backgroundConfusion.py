import torch
import numpy as np
import cv2


def backgroundConfusion(img, block_size=15, noise_intensity=0.6):
    # Convert to numpy and permute to HWC
    img = img.permute(1, 2, 0).cpu().numpy()

    # Ensure range is [0, 255] and dtype is uint8
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    rows, cols = img.shape[:2]

    # Generate blocky noise
    noise = np.random.randint(
        0,
        256,
        (max(1, rows // block_size), max(1, cols // block_size), 3),
        dtype=np.uint8,
    )
    noise = cv2.resize(noise, (cols, rows), interpolation=cv2.INTER_NEAREST)

    # Blend image and noise
    confused_img = cv2.addWeighted(
        img, 1.0 - noise_intensity, noise, noise_intensity, 0
    )

    # Convert back to tensor (CHW, float in [0, 1])
    confused_img = torch.from_numpy(confused_img).permute(2, 0, 1).float() / 255.0

    return confused_img
