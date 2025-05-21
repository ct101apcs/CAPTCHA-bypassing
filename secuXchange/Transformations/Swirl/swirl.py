import cv2
import numpy as np
import torch


def swirl(img, strength=1.5, radius=50):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    if img.shape[0] == 3:  # (3, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)

    img = (img * 255).astype(np.uint8)

    rows, cols = img.shape[:2]
    center_x, center_y = cols // 2, rows // 2

    map_y, map_x = np.indices((rows, cols), dtype=np.float32)
    x = map_x - center_x
    y = map_y - center_y
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) + strength * np.exp(-r / radius)

    map_x = center_x + r * np.cos(theta)
    map_y = center_y + r * np.sin(theta)

    swirled = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    swirled = swirled.astype(np.float32) / 255.0
    swirled = np.transpose(swirled, (2, 0, 1))  # (3, H, W)
    return torch.from_numpy(swirled)


# img = cv2.imread("../Datasets/img_1570.jpg")
# output = swirl(img)
# cv2.imwrite("../Edited/edited_cat.jpg", output)
