import cv2
import numpy as np
import random
import torch

def colorRotation(img):
    img = img.permute(1, 2, 0).numpy()  # Convert [C, H, W] to [H, W, C]
    img = (img * 255).astype(np.uint8)  # Ensure values are between 0 and 255

    # Convert image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Apply noticeable shifts (increase values to ensure effect)
    hue_shift = 20                    # Shift hue by 20 degrees
    sat_scale = 1.5                   # Increase saturation by 50%
    val_scale = 1.3                   # Increase brightness by 30%

    # Apply the shifts
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180  # Hue is circular
    hsv[..., 1] *= sat_scale
    hsv[..., 2] *= val_scale

    # Clip HSV values to valid ranges
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

    # Convert back to BGR
    image_hsv_adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # --- 2. ROTATION USING getRotationMatrix2D ---

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-30, 30)   # Random rotation between -30 and 30 degrees
    scale = 1.0

    # Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Rotate the image with border reflection (to avoid black corners)
    rotated_image = cv2.warpAffine(image_hsv_adjusted, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    final_img = torch.from_numpy(rotated_image).permute(2, 0, 1).float() / 255.0  # Convert back to [C, H, W] and normalize

    return final_img