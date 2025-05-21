import torch
from PIL import Image
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print(f"Using device: {device}")

# Load the model
model = YOLO()
model = model.to(device)
model.eval()

def yolo_prediction(image):
    # Prediction
    results = model.predict(image)

    # Process the results
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
