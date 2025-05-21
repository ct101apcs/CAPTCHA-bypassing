import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as tf
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import tqdm

from Datasets.animal_dataset import AnimalDataset
from Models.ResNet.resnet import resnet_prediction

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

import os
import json
import importlib

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

# Dynamically load enabled transformations
AVAILABLE_TRANSFORMATIONS = {}
for name, entry in config.items():
    if entry.get("enabled", False):
        try:
            module_path = f"Transformations.{name[0].upper() + name[1:]}.{name}"
            mod = importlib.import_module(module_path)
            AVAILABLE_TRANSFORMATIONS[name] = (
                getattr(mod, name),
                entry.get("parameters", {}),
            )
        except (ImportError, AttributeError) as e:
            print(f"Could not import {name}: {e}")

# Evaluate transformations
for t_name, (t_func, t_params) in AVAILABLE_TRANSFORMATIONS.items():
    print(t_params)
    print(f"\n=== Evaluating transformation: {t_name} ===")
    correct = 0
    true_labels = []
    predicted_labels = []
    results = []

    # Prepare output directory and file path
    out_dir = os.path.join("Transformations", t_name[0].upper() + t_name[1:])
    os.makedirs(out_dir, exist_ok=True)
    prediction_file = os.path.join(out_dir, "resnet_predictions.txt")
    metrics_file = os.path.join(out_dir, "resnet_metrics.txt")

    for i in tqdm.tqdm(range(len(dataloader.dataset))):
        img, label_idx = dataloader.dataset[i]
        label = dataset.classes[label_idx]
        img_name = dataset.samples[i][0] if hasattr(dataset, "samples") else f"img_{i}"
        img = t_func(transform(img), **t_params)

        prediction = resnet_prediction(image=img, label=label, classes=dataset.classes)

        results.append(
            f"{os.path.basename(img_name)}\tActual: {label}\tPrediction: {prediction}"
        )

        if label == prediction:
            correct += 1

        true_labels.append(label)
        predicted_labels.append(prediction)

    img_tensor = transform(dataloader.dataset[0][0])  # [C, H, W]
    img_tensor = t_func(img_tensor, **t_params)

    img = img_tensor.detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    accuracy = correct / len(dataloader.dataset) * 100
    f1 = f1_score(
        true_labels, predicted_labels, average="macro", labels=dataset.classes
    )

    with open(prediction_file, "w") as f:
        for line in results:
            f.write(line + "\n")

    with open(metrics_file, "w") as f:
        f.write(f"{t_params}")
        f.write(f"\nAccuracy: {accuracy:.2f}%\n")
        f.write(f"F1 Score (macro): {f1:.4f}\n")
