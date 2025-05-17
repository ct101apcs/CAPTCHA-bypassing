import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_local_masks(h, w, num_patches, patch_size, intensity, is_light=False):
    # create a mask with the same dimension as the image
    mask = np.ones((h, w), dtype=np.float32)

    # compute the patches
    for _ in range(num_patches):
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)

        local = np.zeros((patch_size, patch_size), dtype=np.uint8)

        # random polynom
        points = np.array([
            [np.random.randint(0, patch_size), np.random.randint(0, patch_size)]
            for _ in range(np.random.randint(6, 30))
        ], np.int32)
        cv2.fillPoly(local, [points], 255)

        local = cv2.GaussianBlur(local, (31, 31), 0)
        local = local.astype(np.float32)

        if is_light:
            # Brighten the areas of the patch
            local = 1 + local * intensity
        else:
            # Darken the areas of the patch
            local = 1 - local * intensity  

        # Apply the perturbation on the polygon
        mask[y:y+patch_size, x:x+patch_size] *= local

    return mask


def shadows(img, num_patches=12):
    # img is a torch tensor with shape [3, 224, 224]
    img = img.permute(1, 2, 0).numpy()  # ➜ [224, 224, 3]
    h, w = img.shape[:2]

    patch_size = 50
    intensity = 1
    light_mask = generate_local_masks(
        h, w, num_patches, patch_size, intensity, is_light=True
    )
    shadow_mask = generate_local_masks(
        h, w, num_patches, patch_size, intensity, is_light=False
    )

    combined_mask = shadow_mask * light_mask
    combined_mask_3c = cv2.merge([combined_mask] * 3)

    # apply the mask
    result = img.astype(np.float32) * combined_mask_3c
    result = np.clip(result, 0, 255)  # still float
    result = (
        torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
    )  # ➜ [3, 224, 224] in [0,1]

    return result


# img = cv2.imread("Datasets/archive/animals/animals/bison/0cd71800f3.jpg")
# output = shadows(img)
# img = torch.from_numpy(output)
# print(type(img))
# print(img.shape)

# plt.figure()
# plt.imshow(img)
# plt.show()
