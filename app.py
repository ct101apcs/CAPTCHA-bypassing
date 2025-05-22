from flask import Flask, render_template, request, session, Response, url_for, redirect
import io
import random 
import os 
import uuid 
import json
from datetime import datetime
from pathlib import Path
from werkzeug.middleware.proxy_fix import ProxyFix

from captcha_generator import (
    generate_3x3_image_captcha,
    no_transform
)
from Transformations.Cartoon.cartoon import cartoon
from Transformations.BackgroundConfusion.backgroundConfusion import backgroundConfusion
from Transformations.Sketch.sketch import sketch
from Transformations.GaussianNoise.gaussianNoise import gaussianNoise
from Transformations.Compression.compression import compression
from Transformations.PartialOcclusion.partialOcclusion import partialOcclusion
from Transformations.Swirl.swirl import swirl
from Transformations.Combinations import (
    cartoon_partialOcclusion,
    gaussianNoise_cartoon,
    partialOcclusion_cartoon,
    sketch_compression,
    sketch_compression_gaussianNoise,
    sketch_swirl_gaussianNoise,
    sketch_gaussianNoise,
    sketch_partialOcclusion_gaussianNoise
)
from Transformations.utils import apply_transformation
from PIL import Image, ImageDraw, ImageFont
from model_predictions import predict_with_model

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_demo_only_v3'

# Add ProxyFix middleware to handle X-Forwarded-For headers
# x_for=1 means trust the first IP in X-Forwarded-For
# x_proto=1 means trust X-Forwarded-Proto
# x_host=1 means trust X-Forwarded-Host
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

TEMP_IMAGE_STORE = {}

# Create a directory to store access logs if it doesn't exist
ACCESS_LOGS_DIR = Path("access_logs")
ACCESS_LOGS_DIR.mkdir(exist_ok=True)

# Load transformation settings from JSON
with open("transformations_config.json", "r") as f:
    TRANSFORMATIONS_CONFIG = json.load(f)

def get_client_ip():
    """
    Get the real client IP address by checking various headers.
    
    With ProxyFix middleware and ngrok:
    1. ngrok sets X-Forwarded-For header
    2. ProxyFix middleware processes this header
    3. request.remote_addr will contain the actual client IP
    """
    # First check if we're behind ngrok
    ngrok_headers = {
        'X-Forwarded-For': request.headers.get('X-Forwarded-For'),
        'X-Real-IP': request.headers.get('X-Real-IP'),
        'CF-Connecting-IP': request.headers.get('CF-Connecting-IP'),
        'True-Client-IP': request.headers.get('True-Client-IP')
    }
    
    # If we have X-Forwarded-For from ngrok, use the first IP
    if ngrok_headers['X-Forwarded-For']:
        return ngrok_headers['X-Forwarded-For'].split(',')[0].strip()
    
    # Fallback to other headers
    for header, value in ngrok_headers.items():
        if value:
            return value.strip()
    
    # If no headers found, use remote_addr (which should be correct due to ProxyFix)
    return request.remote_addr or 'Unknown'

def collect_accessor_info():
    """Collect and store information about the accessor"""
    client_ip = get_client_ip()
    
    # Get all relevant headers for debugging
    ip_headers = {
        'X-Forwarded-For': request.headers.get('X-Forwarded-For'),
        'X-Real-IP': request.headers.get('X-Real-IP'),
        'CF-Connecting-IP': request.headers.get('CF-Connecting-IP'),
        'True-Client-IP': request.headers.get('True-Client-IP'),
        'Remote-Addr': request.remote_addr
    }
    
    # Filter out None values
    ip_headers = {k: v for k, v in ip_headers.items() if v is not None}
    
    accessor_info = {
        'timestamp': datetime.now().isoformat(),
        'ip': {
            'detected_client_ip': client_ip,
            'remote_addr': request.remote_addr,
            'all_ip_headers': ip_headers,
            'is_ngrok': 'ngrok' in str(request.headers.get('User-Agent', '')).lower()
        },
        'user_agent': request.headers.get('User-Agent'),
        'session_id': session.get('session_image_store_key')
    }
    
    # Store in a single JSON file that gets updated
    log_file = ACCESS_LOGS_DIR / "access_logs.json"
    
    try:
        # Ensure the directory exists
        ACCESS_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        logs = []
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, start fresh
                logs = []
        
        logs.append(accessor_info)
        
        # Write to a temporary file first
        temp_file = log_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        # If write was successful, replace the original file
        temp_file.replace(log_file)
            
    except Exception as e:
        print(f"Error writing to log file: {e}")
    
    return accessor_info

# Define available transformations with their parameters and accuracy metrics
AVAILABLE_TRANSFORMATIONS = {
    key: {
        "name": details["name"],
        "func": globals()[key.replace('-', '_')] if key != "none" else no_transform,
        "params": details["parameters"],
        "accuracy": details["accuracy"]
    }
    for key, details in TRANSFORMATIONS_CONFIG["transformations"].items()
    if details["enabled"] and details.get("is_demo", False)
}

AVAILABLE_ATTACKER_MODELS = {
    'yolov12': 'YOLOv-12',  
    'resnet18': 'ResNet-18',
    'yolov8': 'YOLOv-8'
}

@app.before_request
def ensure_session_image_key():
    # Collect accessor info before processing the request
    collect_accessor_info()
    
    if 'session_image_store_key' not in session:
        session['session_image_store_key'] = str(uuid.uuid4())
        TEMP_IMAGE_STORE[session['session_image_store_key']] = {}
        session['needs_new_captcha'] = True

@app.route('/', methods=['GET', 'POST'])
def index_visual_attack():
    ai_message = ""
    ai_predictions_visual = []
    ai_solved_correctly = None
    target_category = "N/A"  # Initialize with default value
    target_category_display = "N/A"
    image_urls_for_user = [None]*9 

    session_img_key = session.get('session_image_store_key')
    if not session_img_key or session_img_key not in TEMP_IMAGE_STORE:
        session['session_image_store_key'] = str(uuid.uuid4())
        session_img_key = session['session_image_store_key']
        TEMP_IMAGE_STORE[session_img_key] = {}
        session['needs_new_captcha'] = True

    # Handle refresh button click
    if request.form.get('refresh_captcha'):
        session['needs_new_captcha'] = True

    # Get Attacker Model
    default_attacker_key = list(AVAILABLE_ATTACKER_MODELS.keys())[0]
    selected_attacker_model_key = request.form.get('attacker_model', session.get('current_attacker_key', default_attacker_key))
    # Ensure the selected model is valid
    if selected_attacker_model_key not in AVAILABLE_ATTACKER_MODELS:
        selected_attacker_model_key = default_attacker_key
    session['current_attacker_key'] = selected_attacker_model_key

    # Get transformation count
    selected_transformation_count = int(request.form.get('transformation_count', session.get('current_transformation_count', 1)))
    session['current_transformation_count'] = selected_transformation_count

    # Get transformation type (only used when count is 1)
    default_transform_key = list(AVAILABLE_TRANSFORMATIONS.keys())[0]
    selected_transformation_key = request.form.get('transformation_type', session.get('current_transform_key', default_transform_key))
    session['current_transform_key'] = selected_transformation_key

    # Get current transformation details
    current_transform_details = AVAILABLE_TRANSFORMATIONS.get(selected_transformation_key, AVAILABLE_TRANSFORMATIONS[default_transform_key])
    current_transform_name = current_transform_details["name"]
    current_transform_accuracy = current_transform_details["accuracy"][selected_attacker_model_key]

    # Sort transformations by accuracy for the selected model
    sorted_transformations = dict(sorted(
        AVAILABLE_TRANSFORMATIONS.items(),
        key=lambda x: x[1]["accuracy"][selected_attacker_model_key],
        reverse=True
    ))

    # Generate new CAPTCHA
    if session.get('needs_new_captcha', True):
        # First generate with no transformation
        grid_pil_images, target_category, solution_indices = generate_3x3_image_captcha(
            transformation_func=no_transform
        )
        
        if grid_pil_images is None:
            ai_message = "Error: Could not generate CAPTCHA. Please check image_dataset setup."
            session.pop('captcha_target_category', None)
            session.pop('captcha_solution_indices', None)
            session.pop('ai_predictions_visual', None)
            session.pop('ai_solved_correctly', None)
            if session_img_key and session_img_key in TEMP_IMAGE_STORE:
                TEMP_IMAGE_STORE[session_img_key].pop('captcha_images_data_for_user', None)
        else:
            # Store original images without transformation
            original_images = []
            for img in grid_pil_images:
                img_copy = img.copy()
                original_images.append(img_copy)
                
            session['captcha_target_category'] = target_category
            session['captcha_solution_indices'] = solution_indices
            
            # Store original images in session store
            if session_img_key not in TEMP_IMAGE_STORE:
                TEMP_IMAGE_STORE[session_img_key] = {}
            TEMP_IMAGE_STORE[session_img_key]['original_images'] = original_images
            session['needs_new_captcha'] = False

    # Apply transformations to the original images
    if session_img_key in TEMP_IMAGE_STORE and 'original_images' in TEMP_IMAGE_STORE[session_img_key]:
        original_images = TEMP_IMAGE_STORE[session_img_key]['original_images']
        grid_pil_images = []
        
        if selected_transformation_count == 1:
            # Apply single selected transformation to all images
            transform_details = AVAILABLE_TRANSFORMATIONS.get(selected_transformation_key, AVAILABLE_TRANSFORMATIONS[default_transform_key])
            for img in original_images:
                if selected_transformation_key in ['cartoon_partialOcclusion', 'gaussianNoise_cartoon', 'partialOcclusion_cartoon', 
                                                 'sketch_compression', 'sketch_compression_gaussianNoise', 'sketch_swirl_gaussianNoise',
                                                 'sketch_gaussianNoise', 'sketch_partialOcclusion_gaussianNoise']:
                    # For combinations, pass the parameters from config
                    transformed_img = transform_details["func"](img.copy(), config_params=transform_details["params"])
                else:
                    # For single transformations, use the parameters directly
                    transformed_img = apply_transformation(img.copy(), transform_details["func"], **transform_details["params"])
                grid_pil_images.append(transformed_img)
        else:
            # Get available demo transformations (excluding 'none')
            available_transforms = [(key, details) for key, details in AVAILABLE_TRANSFORMATIONS.items() 
                                 if key != 'none' and TRANSFORMATIONS_CONFIG["transformations"][key].get("is_demo", False)]
            
            # Randomly select transformations for each image
            for img in original_images:
                # Randomly select one transformation for this image
                transform_key, transform_details = random.choice(available_transforms)
                if transform_key in ['cartoon_partialOcclusion', 'gaussianNoise_cartoon', 'partialOcclusion_cartoon', 
                                   'sketch_compression', 'sketch_compression_gaussianNoise', 'sketch_swirl_gaussianNoise',
                                   'sketch_gaussianNoise', 'sketch_partialOcclusion_gaussianNoise']:
                    # For combinations, pass the parameters from config
                    transformed_img = transform_details["func"](img.copy(), config_params=transform_details["params"])
                else:
                    # For single transformations, use the parameters directly
                    transformed_img = apply_transformation(img.copy(), transform_details["func"], **transform_details["params"])
                grid_pil_images.append(transformed_img)
        
        target_category = session.get('captcha_target_category', 'N/A')  # Get target category from session
    else:
        session['needs_new_captcha'] = True
        return redirect(url_for('index_visual_attack'))

    target_category_display = target_category  # Use the already retrieved target_category
    
    # Process AI predictions
    ai_selected_indices = []
    current_ai_predictions_visual = []

    for i, pil_img in enumerate(grid_pil_images):
        is_predicted_target, confidence = predict_with_model(selected_attacker_model_key, pil_img.copy(), target_category)
        current_ai_predictions_visual.append({'is_selected': is_predicted_target, 'confidence': confidence})
        if is_predicted_target:
            ai_selected_indices.append(i)
    
    ai_selected_indices.sort()
    solution_indices = session.get('captcha_solution_indices', [])
    if ai_selected_indices == solution_indices:
        ai_message = f"AI ({AVAILABLE_ATTACKER_MODELS[selected_attacker_model_key]}): CORRECT! (Selected: {ai_selected_indices})"
        ai_solved_correctly = True
    else:
        ai_message = f"AI ({AVAILABLE_ATTACKER_MODELS[selected_attacker_model_key]}): INCORRECT. (AI: {ai_selected_indices}, Correct: {solution_indices})"
        ai_solved_correctly = False
    
    session['ai_predictions_visual'] = current_ai_predictions_visual 
    ai_predictions_visual = current_ai_predictions_visual

    # Save transformed images for display
    images_data_for_current_session = []
    for pil_img_item in grid_pil_images:
        img_io_bytes = io.BytesIO()
        pil_img_item.save(img_io_bytes, 'PNG')
        img_io_bytes.seek(0)
        images_data_for_current_session.append(img_io_bytes.read())
    
    TEMP_IMAGE_STORE[session_img_key]['captcha_images_data_for_user'] = images_data_for_current_session
    image_urls_for_user = [url_for('captcha_image_for_user_grid', grid_index=i) for i in range(9)]

    if not ai_predictions_visual or len(ai_predictions_visual) != 9:
        ai_predictions_visual = [{'is_selected': False, 'confidence': 0.0}] * 9
        if grid_pil_images is None and not ("Error" in ai_message):
             ai_message = "AI Attacker: Waiting for CAPTCHA..."

    cache_buster = random.randint(100000, 999999)

    return render_template('index_visual_attack.html',
                           ai_message=ai_message,
                           ai_predictions_visual=ai_predictions_visual,
                           ai_solved_correctly=ai_solved_correctly,
                           target_category=target_category_display,
                           image_urls=image_urls_for_user,
                           transformations_options=sorted_transformations,
                           selected_transformation_key=selected_transformation_key,
                           current_transform_name=current_transform_name,
                           current_transform_accuracy=current_transform_accuracy,
                           attacker_models_options=AVAILABLE_ATTACKER_MODELS, 
                           selected_attacker_model=selected_attacker_model_key,
                           selected_transformation_count=selected_transformation_count,
                           num_images=9,
                           cache_buster_value=cache_buster)

@app.route('/captcha_image_user_grid/<int:grid_index>')
def captcha_image_for_user_grid(grid_index):
    session_img_key = session.get('session_image_store_key')
    if not session_img_key:
        return "Session error", 400

    session_specific_store = TEMP_IMAGE_STORE.get(session_img_key, {})
    images_data_list = session_specific_store.get('captcha_images_data_for_user', [])
    
    if images_data_list and 0 <= grid_index < len(images_data_list) and images_data_list[grid_index]:
        return Response(images_data_list[grid_index], mimetype='image/png')
    
    placeholder_img = Image.new('RGB', (224, 224), color='grey')
    draw = ImageDraw.Draw(placeholder_img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 30), "No Img", fill="white", font=font)
    img_io_bytes = io.BytesIO()
    placeholder_img.save(img_io_bytes, 'PNG')
    img_io_bytes.seek(0)
    return Response(img_io_bytes.read(), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)