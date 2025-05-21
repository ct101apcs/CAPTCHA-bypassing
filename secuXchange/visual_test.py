import json
import json
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys

from Models.ResNet.resnet import resnet_prediction
from Datasets.animal_dataset import AnimalDataset

import argparse

from Transformations.BackgroundConfusion.backgroundConfusion import backgroundConfusion
from Transformations.Cartoon.cartoon import cartoon
from Transformations.ColorRotation.colorRotation import colorRotation
from Transformations.Compression.compression import compression
from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from Transformations.PartialOcclusion.partialOcclusion import partialOcclusion
from Transformations.Rotation.rotation import rotation
from Transformations.Shuffle.shuffle import shuffle
from Transformations.Shadows.shadows import shadows
from Transformations.Sketch.sketch import sketch
from Transformations.Swirl.swirl import swirl
from Transformations.Wave.wave import wave

# Define valid transformation names
valid_transformations = [
    "backgroundConfusion", "cartoon", "colorRotation", "compression", "gaussianNoise", "partialOcclusion", "rotation", "shadows", "shuffle", "sketch", "swirl", "wave",
]

parser = argparse.ArgumentParser(description="Run transformations.")
parser.add_argument(
    "-t", "--transforms",
    nargs="+",
    choices=valid_transformations,
    help="Choose one or more transformations",
)

args = parser.parse_args()

if not args.transforms:
    print(
        "No transformation selected.\nUse -t followed by one or more of the following transformations:"
    )
    print(", ".join(valid_transformations))
    sys.exit(1)

print("Selected transformations:", args.transforms)


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

with open("transformations_config.json", "r") as f:
    config = json.load(f)


with open("transformations_config.json", "r") as f:
    config = json.load(f)


# Displaying 12 random images
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, ax in enumerate(axes.flat):
    img = dataloader.dataset[list_img[i]][0]
    params = []
    for t in args.transforms:
        if config[t]["enabled"]:
            func = globals()[t]
            params = config[t]["parameters"]
            img = func(img, **params)
        else:
            break

    label = dataset.classes[dataloader.dataset[list_img[i]][1]]

    prediction = resnet_prediction( image=img, label = label, classes=dataset.classes)

    selected_img[i] = 1 if label == prediction else 0

    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(
        f"Label: {label}\n Predicted: {prediction}\n Correct: {selected_img[i]}"
    )
    ax.axis("off")

plt.tight_layout()
plt.show()

# plt.savefig("demo.png")

# good combos
# sketch(colorRotation(img)) 10v1
# sketch(swirl(img)) 10v1
# backgroundConfusion(wave(img)) 8v0
# cartoon(colorRotation(img)) 9v1
# cartoon(compression(img)) 10v1
# cartoon(partialOcclusion(img)) 10v2
# swirl(backgroundConfusion(img, **params)) 10v1

# bad combos
# sketch cartoon hard
# sketch(wave(img)) 8v4
# sketch(backgroundConfusion(img)) 6v1
# cartoon(wave(img)) 6v0
# sketch(compression(img)) 6v0
