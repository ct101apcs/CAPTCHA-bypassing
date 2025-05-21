import torch
import torchvision.transforms as transforms
from PIL import Image
from Models.ResNet.resnet import resnet_prediction
import random

from Datasets.animal_dataset import AnimalDataset

def predict_with_model(selected_model_key, pil_image, target_category_name):
    """
    Model prediction function that uses actual ResNet implementation and mock implementations for others.
    Args:
        selected_model_key (str): The key identifying which model to use
        pil_image (PIL.Image): The input image
        target_category_name (str): The target category name
    Returns:
        tuple: (is_predicted_target_bool, confidence_float)
    """
    if selected_model_key == 'resnet18':
        # Convert PIL image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(pil_image)

        dataset = AnimalDataset(root_dir='Datasets/archive/animals/animals', transform=transform)
        
        # Get prediction from actual ResNet model
        predicted_label = resnet_prediction(img_tensor, classes=dataset.classes)
        is_target_prediction = (predicted_label.lower() == target_category_name.lower())
        
        # For now, we'll use a fixed high confidence when the prediction matches
        confidence = 0.95 if is_target_prediction else 0.1
        return is_target_prediction, confidence
        
    # Mock implementations for other models
    is_target_prediction = False
    confidence = round(random.uniform(0.1, 0.95), 2)
    
    base_success_rate = 0.1 
    target_bonus = 0      
    confidence_floor = 0.1
    confidence_ceiling = 0.95

    if selected_model_key == 'yolov12': 
        base_success_rate = 0.3
        target_bonus = 0.3
        confidence_floor = 0.4
        confidence_ceiling = 0.98
    elif selected_model_key == 'yolov8': 
        base_success_rate = 0.25
        target_bonus = 0.2
        confidence_floor = 0.35
        confidence_ceiling = 0.92

    if random.random() < (base_success_rate + target_bonus):
        is_target_prediction = True
        confidence = round(random.uniform(confidence_floor, confidence_ceiling), 2)
    else:
        confidence = round(random.uniform(0.05, confidence_floor - 0.05 if confidence_floor > 0.1 else 0.3), 2)
        
    return is_target_prediction, confidence 