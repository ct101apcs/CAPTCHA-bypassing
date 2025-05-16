from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import os

IMAGE_CATEGORIES = ["cat", "dog"] 
DATASET_PATH = "./Datasets/archive/animals/animals" 

def get_random_image_paths(category, count=1, exclude_paths=None):
    """
    Get multiple unique random image paths from a category.
    
    Args:
        category (str): The category to get images from
        count (int): Number of unique images to get
        exclude_paths (set): Set of paths to exclude
        
    Returns:
        list: List of unique image paths
    """
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(category_path) or not os.path.isdir(category_path):
        print(f"Warning: Category path {category_path} does not exist or is not a directory.")
        return []
    
    image_files = [f for f in os.listdir(category_path) 
                   if os.path.isfile(os.path.join(category_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Warning: No suitable image files found in category {category_path}.")
        return []
        
    # Convert paths to full paths
    image_files = [os.path.join(category_path, f) for f in image_files]
    
    # Remove excluded paths
    if exclude_paths:
        image_files = [f for f in image_files if f not in exclude_paths]
    
    if len(image_files) < count:
        print(f"Warning: Not enough unique images in category {category}. Requested {count}, found {len(image_files)}.")
        return image_files
        
    return random.sample(image_files, count)

def get_random_image_path(category):
    """Get a single random image path."""
    paths = get_random_image_paths(category, count=1)
    return paths[0] if paths else None

def generate_3x3_image_captcha(num_targets_min=2, num_targets_max=4, image_size=(80,80), transformation_func=None):
    """
    Generates a 3x3 grid of images for the CAPTCHA.

    Args:
        num_targets_min (int): Min number of target images in the grid.
        num_targets_max (int): Max number of target images in the grid.
        image_size (tuple): Target size (width, height) for each image in the grid.
        transformation_func (function, optional): A function that takes a PIL.Image 
                                                 and returns a transformed PIL.Image.

    Returns:
        tuple: (list_of_pil_images, target_category_str, list_of_correct_indices)
               Returns (None, None, None) if categories or images are insufficient.
    """
    if not IMAGE_CATEGORIES:
        print("Error: IMAGE_CATEGORIES is empty. Please define categories.")
        return None, None, None
        
    target_category = random.choice(IMAGE_CATEGORIES)
    distractor_categories = [c for c in IMAGE_CATEGORIES if c != target_category]

    if not distractor_categories and len(IMAGE_CATEGORIES) > 0: 
        print("Warning: Only one image category available. Using it for both target and distractors.")
        distractor_categories = [target_category] 
    elif not distractor_categories and len(IMAGE_CATEGORIES) == 0:
        print("Error: No image categories defined for distractors.")
        return None, None, None

    num_targets = random.randint(num_targets_min, min(num_targets_max, 9))
    
    grid_pil_images = []
    correct_indices = []
    used_image_paths = set()

    # First, get all target images
    target_images = get_random_image_paths(target_category, count=num_targets, exclude_paths=used_image_paths)
    if len(target_images) < num_targets:
        print("Error: Could not get enough unique target images.")
        return None, None, None
    
    # Add paths to used set
    used_image_paths.update(target_images)
    
    # Get positions for target images
    target_positions = random.sample(range(9), num_targets)
    for pos in range(9):
        if pos in target_positions:
            # Use a target image
            img_path = target_images.pop()
            correct_indices.append(pos)
        else:
            # Get a distractor image
            distractor_category = random.choice(distractor_categories)
            img_path = get_random_image_paths(distractor_category, count=1, exclude_paths=used_image_paths)[0]
            if not img_path:
                print(f"Error: Could not get unique distractor image for position {pos}")
                return None, None, None
            used_image_paths.add(img_path)
            
        try:
            img = Image.open(img_path).convert("RGB").resize(image_size)
        except Exception as e:
            print(f"Error loading/resizing image {img_path}: {e}")
            img = Image.new('RGB', image_size, color='gray')
            draw = ImageDraw.Draw(img)
            draw.text((10,10), "Err", fill="red", font=ImageFont.load_default())
            
        if transformation_func:
            try:
                img = transformation_func(img.copy())
            except Exception as e:
                print(f"Error applying transformation to image: {e}")
                
        grid_pil_images.append(img)

    return grid_pil_images, target_category, sorted(correct_indices)

def no_transform(image):
    return image

def simple_blur_transform(image):
    return image.filter(ImageFilter.GaussianBlur(radius=1.2))

def best_transform_placeholder(image):
    # Example: Add some noise and a bit of swirl
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(50): 
        nx, ny = random.randint(0, width-1), random.randint(0, height-1)
        draw.point((nx,ny), fill=(random.randint(0,50), random.randint(0,50), random.randint(0,50)))
    
    # A very simple "swirl" like effect (pixel displacement)
    # This is a placeholder for a more sophisticated geometric transform
    # For a real swirl, you'd use more complex pixel mapping or libraries like OpenCV
    # image = image.rotate(random.uniform(-5, 5), expand=False, fillcolor='white')
    # image = image.transform(image.size, Image.AFFINE, (1, random.uniform(-0.1,0.1), 0, random.uniform(-0.1,0.1), 1, 0))
    return image.filter(ImageFilter.SMOOTH)

# if __name__ == '__main__':
#     if not os.path.exists(DATASET_PATH):
#         os.makedirs(os.path.join(DATASET_PATH, "cat"))
#         os.makedirs(os.path.join(DATASET_PATH, "dog"))
#         print(f"Created dummy '{DATASET_PATH}' and subfolders. Please add images to them.")
    
#     print("Testing CAPTCHA generation...")
#     images, target, solution = generate_3x3_image_captcha(transformation_func=simple_blur_transform)
    
#     if images:
#         print(f"Generated CAPTCHA for target: '{target}', solution indices: {solution}")
#         composite_w = images[0].width * 3
#         composite_h = images[0].height * 3
#         composite_img = Image.new('RGB', (composite_w, composite_h))
#         for idx, img in enumerate(images):
#             row, col = divmod(idx, 3)
#             composite_img.paste(img, (col * img.width, row * img.height))
#         composite_img.save("test_captcha_grid.png")
#         print("Saved test_captcha_grid.png")
#     else:
#         print("Failed to generate CAPTCHA. Check Datasets setup and categories.")