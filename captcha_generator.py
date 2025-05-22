from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os
from flask import session, redirect, url_for

with open("./Datasets/name of the animals.txt", "r") as file:
    IMAGE_CATEGORIES = [line.strip() for line in file if line.strip()]

DATASET_PATH = "./Datasets/archive/animals/animals" 

# Initialize CATEGORY_IMAGES dictionary
CATEGORY_IMAGES = {}
for category in IMAGE_CATEGORIES:
    category_path = os.path.join(DATASET_PATH, category)
    if os.path.exists(category_path) and os.path.isdir(category_path):
        image_files = [os.path.join(category_path, f) for f in os.listdir(category_path) 
                      if os.path.isfile(os.path.join(category_path, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            CATEGORY_IMAGES[category] = image_files

def get_random_image_paths(category, count=1, exclude_paths=None):
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(category_path) or not os.path.isdir(category_path):
        print(f"Warning: Category path {category_path} does not exist or is not a directory.")
        return []
    
    image_files = [f for f in os.listdir(category_path) 
                   if os.path.isfile(os.path.join(category_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Warning: No suitable image files found in category {category_path}.")
        return []
        
    image_files = [os.path.join(category_path, f) for f in image_files]
    
    if exclude_paths:
        image_files = [f for f in image_files if f not in exclude_paths]
    
    if len(image_files) < count:
        print(f"Warning: Not enough unique images in category {category}. Requested {count}, found {len(image_files)}.")
        return image_files
        
    return random.sample(image_files, count)

def get_random_image_path(category):
    paths = get_random_image_paths(category, count=1)
    return paths[0] if paths else None

def no_transform(image):
    return image

def generate_3x3_image_captcha(transformation_func=no_transform, grid_size=3):
    """
    Generate a grid of images with one target category and other random categories.
    The grid will be grid_size x grid_size.
    """
    # Get all available categories
    categories = list(CATEGORY_IMAGES.keys())
    if not categories:
        return None, None, None

    # Calculate total images needed
    total_grid_size = grid_size * grid_size
    num_target_images = grid_size
    num_other_images = total_grid_size - num_target_images

    print(f"Generating {grid_size}x{grid_size} grid. Need {num_target_images} target images and {num_other_images} other images.")

    # Find a category with enough images
    target_category = None
    for category in categories:
        if len(CATEGORY_IMAGES[category]) >= num_target_images:
            target_category = category
            break
    
    if not target_category:
        # If no category has enough images, use the one with the most images
        target_category = max(categories, key=lambda x: len(CATEGORY_IMAGES[x]))
        # If we still don't have enough target images, we'll need to reuse some
        target_images = CATEGORY_IMAGES[target_category]
        while len(target_images) < num_target_images:
            target_images.extend(random.sample(target_images, min(len(target_images), num_target_images - len(target_images))))
    else:
        target_images = CATEGORY_IMAGES[target_category]

    # Randomly select target images
    selected_target_images = random.sample(target_images, num_target_images)
    
    # Get images from other categories
    other_categories = [cat for cat in categories if cat != target_category]
    other_images = []
    
    # First try to get unique images from each category
    for cat in other_categories:
        cat_images = CATEGORY_IMAGES[cat]
        if cat_images:
            other_images.extend(cat_images)
    
    # If we don't have enough other images, reuse some from other categories
    if len(other_images) < num_other_images:
        # Add more images by reusing some from other categories
        while len(other_images) < num_other_images:
            cat = random.choice(other_categories)
            cat_images = CATEGORY_IMAGES[cat]
            if cat_images:
                other_images.extend(random.sample(cat_images, min(len(cat_images), num_other_images - len(other_images))))
    
    # If we still don't have enough images, reuse some from the target category
    if len(other_images) < num_other_images:
        remaining_needed = num_other_images - len(other_images)
        other_images.extend(random.sample(target_images, min(len(target_images), remaining_needed)))
    
    # Randomly select other images
    selected_other_images = random.sample(other_images, num_other_images)
    
    # Combine all selected images
    all_selected_images = selected_target_images + selected_other_images
    random.shuffle(all_selected_images)
    
    print(f"Selected {len(all_selected_images)} total images: {len(selected_target_images)} target images and {len(selected_other_images)} other images")
    
    # Apply transformation to all images
    transformed_images = []
    for img_path in all_selected_images:
        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            transformed_img = transformation_func(img)
            transformed_images.append(transformed_img)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Create a placeholder image if there's an error
            placeholder = Image.new('RGB', (224, 224), color='gray')
            draw = ImageDraw.Draw(placeholder)
            draw.text((10, 10), "Error", fill="red", font=ImageFont.load_default())
            transformed_images.append(placeholder)
    
    # Get indices of target images in the final grid
    solution_indices = []
    for i, img_path in enumerate(all_selected_images):
        if img_path in selected_target_images:
            solution_indices.append(i)
    solution_indices.sort()
    
    print(f"Generated grid with {len(transformed_images)} images and {len(solution_indices)} target indices")
    
    return transformed_images, target_category, solution_indices
    