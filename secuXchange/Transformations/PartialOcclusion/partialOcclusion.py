import cv2
import numpy as np
import torch
import random

def partialOcclusion(img, num_boxes=5, box_min_size=10, box_max_size=40):        
    # Load the image
    image = img.permute(1, 2, 0).numpy()  # Convert to [H, W, C]
    image = (image * 255).astype(np.uint8)  # Ensure values are between 0 and 255


    if image is None:
        print("Error: image not found or unable to load.")
        exit()

    # Copy the image to avoid modifying the original
    occluded_image = image.copy()

    # Draw random boxes
    for _ in range(num_boxes):
        h, w = image.shape[:2]
        
        # Random box size
        box_w = random.randint(box_min_size, box_max_size)
        box_h = random.randint(box_min_size, box_max_size)

        # Random position
        top_left_x = random.randint(0, w - box_w)
        top_left_y = random.randint(0, h - box_h)

        # Optional: randomize color 
        color = (0, 0, 0)  # Black box (BGR format)

        # Draw rectangle
        cv2.rectangle(occluded_image, (top_left_x, top_left_y), 
                    (top_left_x + box_w, top_left_y + box_h), color, thickness=-1)

    final_img = torch.from_numpy(occluded_image).permute(2, 0, 1).float() / 255.0
    return final_img
