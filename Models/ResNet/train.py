import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
from Datasets.animal_dataset import AnimalDataset
import tqdm
from torch.utils.data import random_split

# Configuration
batch_size = 32
num_epochs = 20
learning_rate = 0.001

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using device: {device}")

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
    ]
)

# Load the dataset
dataset = AnimalDataset(
    root_dir="Datasets/archive/animals/animals", transform=transform
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Model setup
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, 90)
model.to(device)

# Loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm.tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%"
    )

# Save the model
torch.save(model.state_dict(), "Models/ResNet/resnet18_animals.pth")