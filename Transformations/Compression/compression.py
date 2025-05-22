import cv2
import numpy as np
import torch

def compression(img, compression_quality=10):
    # img: torch.Tensor [C, H, W]
    img = img.permute(1, 2, 0).numpy()  # Convert to [H, W, C]
    img = (img * 255).astype(np.uint8)  # Ensure values are between 0 and 255

    # Encode as JPEG (simulate compression)
    result, encoded_img = cv2.imencode(
        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, compression_quality]
    )
    if not result:
        raise RuntimeError("Compression JPEG a échoué")

    # Decode back to image
    img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    # Convert back to tensor [C, H, W] and normalize to [0, 1]
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    return img
