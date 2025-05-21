import argparse

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import tqdm

from Datasets.animal_dataset import AnimalDataset
from Models.ResNet.resnet import resnet_prediction

import os
import json
import importlib

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
    required=False
)

args = parser.parse_args()

print("Selected transformations:", args.transforms)

# Dataset and dataloader
dataset = AnimalDataset(
    root_dir="Datasets/archive/animals/animals",
    transform=None,
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Image transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Load transformation settings from JSON
with open("transformations_config.json", "r") as f:
    config = json.load(f)

def apply_combo(combo, img):
    for func, params in combo.items():
        img = func(img, **params)
    return img

# Dynamically load enabled transformations
if args.transforms:
    combos = [args.transforms]
else:
    combos = [
        ["sketch", "colorRotation"],
        #["sketch", "swirl"],
        # ["backgroundConfusion", "wave"],
        # ["cartoon", "colorRotation"],
        # ["cartoon", "partialOcclusion"],
        # ["cartoon", "compression"],
        # ["backgroundConfusion", "swirl"],
        # ["gaussianNoise", "cartoon", "compression"]
    ]

for t_combo in combos:
    print(f"\n=== Evaluating combo: {t_combo} ===")
    correct = 0
    true_labels = []
    predicted_labels = []
    results = []

    # Make a readable folder name, e.g., "Sketch_ColorRotation"
    combo_name = "_".join([t.capitalize() for t in t_combo])

    # Prepare output directory and file path
    out_dir = os.path.join("Transformations/Combinations", combo_name)
    os.makedirs(out_dir, exist_ok=True)
    prediction_file = os.path.join(out_dir, "resnet_predictions.txt")
    metrics_file = os.path.join(out_dir, "resnet_metrics.txt")
    combo = {}
    for t in t_combo:
        func = globals()[t]
        params = config[t]["parameters"]
        combo[func] = params

    for i in tqdm.tqdm(range(len(dataloader.dataset))):
        img, label_idx = dataloader.dataset[i]
        label = dataset.classes[label_idx]
        img_name = dataset.samples[i][0] if hasattr(dataset, "samples") else f"img_{i}"
        img = apply_combo(combo, transform(img))

        prediction = resnet_prediction(image=img, label=label, classes=dataset.classes)

        results.append(
            f"{os.path.basename(img_name)}\tActual: {label}\tPrediction: {prediction}"
        )

        if label == prediction:
            correct += 1

        true_labels.append(label)
        predicted_labels.append(prediction)

    accuracy = correct / len(dataloader.dataset) * 100
    f1 = f1_score(
        true_labels, predicted_labels, average="macro", labels=dataset.classes
    )

    with open(prediction_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    with open(metrics_file, "w") as f:
        for t, params in combo.items():
            f.write(f"{t.__name__}\t{params}\n")
        f.write(f"\nAccuracy: {accuracy:.2f}%\n")
        f.write(f"F1 Score (macro): {f1:.4f}\n")