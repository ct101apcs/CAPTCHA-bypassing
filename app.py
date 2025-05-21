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
        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append(accessor_info)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        print(f"Error writing to log file: {e}")
    
    return accessor_info

def mock_predict_with_model(selected_model_key, pil_image, target_category_name):
    """
    MOCK FUNCTION for model prediction.
    Behavior changes based on selected_model_key.
    Returns: (is_predicted_target_bool, confidence_float)
    """
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
    elif selected_model_key == 'resnet18': 
        base_success_rate = 0.2
        target_bonus = 0.25
        confidence_floor = 0.3
        confidence_ceiling = 0.9
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

# Define available transformations with their parameters
AVAILABLE_TRANSFORMATIONS = {
    "none": {"name": "No Transformation (Baseline)", "func": no_transform, "params": {}},
    "gaussian_noise": {"name": "Gaussian Noise", "func": gaussianNoise, "params": {"stddev": 0.1}},
    "cartoon": {"name": "Cartoon Effect", "func": cartoon, "params": TRANSFORMATIONS_CONFIG["cartoon"]["parameters"]},
    "background_confusion": {"name": "Background Confusion", "func": backgroundConfusion, "params": TRANSFORMATIONS_CONFIG["backgroundConfusion"]["parameters"]},
    "sketch": {"name": "Sketch Effect", "func": sketch, "params": TRANSFORMATIONS_CONFIG["sketch"]["parameters"]}
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

    default_transform_key = list(AVAILABLE_TRANSFORMATIONS.keys())[0]
    selected_transformation_key = request.form.get('transformation_type', session.get('current_transform_key', default_transform_key))
    session['current_transform_key'] = selected_transformation_key
    transformation_details = AVAILABLE_TRANSFORMATIONS.get(selected_transformation_key, AVAILABLE_TRANSFORMATIONS[default_transform_key])
    transform_function_to_apply = transformation_details["func"]
    transform_params = transformation_details["params"]
    current_transform_name = transformation_details["name"]

    # Get Attacker Model
    default_attacker_key = list(AVAILABLE_ATTACKER_MODELS.keys())[0]
    selected_attacker_model_key = request.form.get('attacker_model', session.get('current_attacker_key', default_attacker_key))
    # Ensure the selected model is valid
    if selected_attacker_model_key not in AVAILABLE_ATTACKER_MODELS:
        selected_attacker_model_key = default_attacker_key
    session['current_attacker_key'] = selected_attacker_model_key

    # Only generate new CAPTCHA if needed
    if session.get('needs_new_captcha', True):
        grid_pil_images, target_category, solution_indices = generate_3x3_image_captcha(
            transformation_func=lambda img: transform_function_to_apply(img, **transform_params)
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
    else:
        # Retrieve stored original images and apply current transformation
        original_images = TEMP_IMAGE_STORE[session_img_key].get('original_images', [])
        if original_images:
            grid_pil_images = [transform_function_to_apply(img.copy(), **transform_params) for img in original_images]
            target_category = session.get('captcha_target_category')
            solution_indices = session.get('captcha_solution_indices')
        else:
            session['needs_new_captcha'] = True
            return redirect(url_for('index_visual_attack'))

    target_category_display = session.get('captcha_target_category', 'N/A')
    
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
                           transformations_options=AVAILABLE_TRANSFORMATIONS,
                           selected_transformation_key=selected_transformation_key,
                           current_transform_name=current_transform_name,
                           attacker_models_options=AVAILABLE_ATTACKER_MODELS, 
                           selected_attacker_model=selected_attacker_model_key, 
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