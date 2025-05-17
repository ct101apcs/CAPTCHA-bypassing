import torch
from torchvision import models

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
# print(f"Using device: {device}")

# Load the pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Modify the fc layer to match the number of output features in your trained model
num_classes = 90
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load the state dictionary from your trained model
model.load_state_dict(
    torch.load("Models/ResNet/resnet18_animals.pth", map_location=device)
)
model = model.to(device)
model.eval()


def resnet_prediction(image, classes=None):
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move the image to the same device as the model

    with torch.no_grad():
        outputs = model(image)
        _, predicted_label = torch.max(outputs, 1)
        predicted_index = predicted_label.item()
        predicted_label = classes[predicted_index]

    print("\nResNet18 Prediction:")
    print(f"Predicted index: {predicted_index}")
    print(f"Predicted label: {predicted_label}")
    return predicted_label
