import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_local_masks(h, w, num_patches, patch_size, intensity, is_light=False):
    mask = np.ones((h, w), dtype=np.float32)

    for _ in range(num_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)

        local = np.zeros((patch_size, patch_size), dtype=np.uint8)

        points = np.array(
            [
                [np.random.randint(0, patch_size), np.random.randint(0, patch_size)]
                for _ in range(np.random.randint(5, 15))
            ],
            np.int32,
        )
        cv2.fillPoly(local, [points], 255)
        local = cv2.GaussianBlur(local, (21, 21), 0)

        local = local.astype(np.float32) / 255.0

        if is_light:
            local = 1 + local * intensity
        else:
            local = 1 - local * intensity

        mask[y : y + patch_size, x : x + patch_size] *= local

    return mask


def shadows(img_tensor, num_patches=8, patch_size=50, intensity=0.6):
    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
    h, w = img.shape[:2]

    light_mask = generate_local_masks(
        h, w, num_patches, patch_size, intensity, is_light=True
    )
    shadow_mask = generate_local_masks(
        h, w, num_patches, patch_size, intensity, is_light=False
    )

    combined_mask = shadow_mask * light_mask
    combined_mask_3c = np.stack([combined_mask] * 3, axis=-1)  # (H, W, 3)

    result = img.astype(np.float32) * combined_mask_3c
    result = np.clip(result, 0, 1.0)

    return torch.from_numpy(result).permute(2, 0, 1).float()  # [C, H, W]
