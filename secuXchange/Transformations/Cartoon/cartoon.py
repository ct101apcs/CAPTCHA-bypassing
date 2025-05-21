import numpy as np
import cv2
import torch


def cartoon(img):
    # Convert to HWC and ensure range is [0, 255] and type is uint8
    img = img.permute(1, 2, 0).cpu().numpy()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    num_down = 2
    num_bilateral = 7

    # Downsample image using Gaussian pyramid
    img_color = img.copy()
    for _ in range(num_down):
        img_color = cv2.pyrDown(img_color)

    # Apply bilateral filter multiple times
    for _ in range(num_bilateral):
        img_color = cv2.bilateralFilter(img_color, d=9, sigmaColor=9, sigmaSpace=7)

    # Upsample back to original size
    for _ in range(num_down):
        img_color = cv2.pyrUp(img_color)
    img_color = cv2.resize(img_color, (img.shape[1], img.shape[0]))

    # Convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.medianBlur(img_gray, 7)

    # Detect and enhance edges
    img_edge = cv2.adaptiveThreshold(
        img_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2,
    )

    # Convert back to color and combine with edge mask
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    img_cartoon = cv2.bitwise_and(img_color, img_edge)

    # Back to torch tensor (CHW, float in [0, 1])
    img_cartoon = torch.from_numpy(img_cartoon).permute(2, 0, 1).float() / 255.0
    return img_cartoon
