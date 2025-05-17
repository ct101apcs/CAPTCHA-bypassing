import random
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Transformation.partialOcclusion import partialOcclusion
from Models.ResNet.resnet import resnet_prediction
from Datasets.animal_dataset import AnimalDataset

dataset = AnimalDataset(
    root_dir="Datasets/archive/animals/animals",
    transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    ),
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

nb = random.randint(0, len(dataloader.dataset) - 1)

list_img = [random.randint(0, len(dataloader.dataset) - 1) for _ in range(12)]

selected_img = [0 for _ in range(12)]

# Displaying 12 random images
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, ax in enumerate(axes.flat):
    img = partialOcclusion(dataloader.dataset[list_img[i]][0])
    label = dataset.classes[dataloader.dataset[list_img[i]][1]]
    prediction = resnet_prediction(
    image=img,
    classes=dataset.classes
)
    selected_img[i] = 1 if label == prediction else 0

    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(
        f"Label: {label}\n Predicted: {prediction}\n Correct: {selected_img[i]}"
    )

    ax.legend()
    ax.axis("off")
plt.tight_layout()
plt.show()
