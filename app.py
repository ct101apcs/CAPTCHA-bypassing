from flask import Flask, render_template, request, session, Response, url_for
import io
import random 
import os 
import uuid 

from captcha_generator import (
    generate_3x3_image_captcha,
    no_transform,
    simple_blur_transform,
    best_transform_placeholder
)
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_demo_only_v3' 

TEMP_IMAGE_STORE = {}

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
        if target_category_name.lower() in ["car", "dog"]: 
            target_bonus = 0.3
        confidence_floor = 0.4
        confidence_ceiling = 0.98
    elif selected_model_key == 'resnet18': 
        base_success_rate = 0.2
        if target_category_name.lower() in ["cat", "dog", "tree"]: 
            target_bonus = 0.25
        confidence_floor = 0.3
        confidence_ceiling = 0.9
    elif selected_model_key == 'vit': 
        base_success_rate = 0.25
        if target_category_name.lower() in ["cat", "tree", "car"]:
            target_bonus = 0.2
        confidence_floor = 0.35
        confidence_ceiling = 0.92

    if random.random() < (base_success_rate + target_bonus):
        is_target_prediction = True
        confidence = round(random.uniform(confidence_floor, confidence_ceiling), 2)
    else:
        confidence = round(random.uniform(0.05, confidence_floor - 0.05 if confidence_floor > 0.1 else 0.3), 2)
        
    return is_target_prediction, confidence

AVAILABLE_TRANSFORMATIONS = {
    "none": {"name": "No Transformation (Baseline)", "func": no_transform},
    "blur": {"name": "Simple Blur", "func": simple_blur_transform},
    "my_best": {"name": "My Best Transformation (Demo)", "func": best_transform_placeholder},
}

AVAILABLE_ATTACKER_MODELS = {
    'yolov12': 'YOLOv-12',  
    'resnet18': 'ResNet-18',
    'vit': 'Vision Transformer (Mock)'
}

@app.before_request
def ensure_session_image_key():
    if 'session_image_store_key' not in session:
        session['session_image_store_key'] = str(uuid.uuid4())
        TEMP_IMAGE_STORE[session['session_image_store_key']] = {}

@app.route('/', methods=['GET', 'POST'])
def index_visual_attack():
    ai_message = ""
    ai_predictions_visual = []
    ai_solved_correctly = None
    target_category_display = "N/A"
    image_urls_for_user = [None]*9 

    session_img_key = session.get('session_image_store_key')
    if not session_img_key:
        session['session_image_store_key'] = str(uuid.uuid4())
        session_img_key = session['session_image_store_key']
        if session_img_key not in TEMP_IMAGE_STORE:
            TEMP_IMAGE_STORE[session_img_key] = {}


    default_transform_key = list(AVAILABLE_TRANSFORMATIONS.keys())[0]
    selected_transformation_key = request.form.get('transformation_type', session.get('current_transform_key', default_transform_key))
    session['current_transform_key'] = selected_transformation_key
    transformation_details = AVAILABLE_TRANSFORMATIONS.get(selected_transformation_key, AVAILABLE_TRANSFORMATIONS[default_transform_key])
    transform_function_to_apply = transformation_details["func"]
    current_transform_name = transformation_details["name"]

    # --- Get Attacker Model ---
    default_attacker_key = list(AVAILABLE_ATTACKER_MODELS.keys())[0]
    selected_attacker_model_key = request.form.get('attacker_model', session.get('current_attacker_key', default_attacker_key))
    session['current_attacker_key'] = selected_attacker_model_key

    # --- Generate New CAPTCHA & Run AI Attack ---
    # This always runs as there's no separate user submission anymore.
    # A POST request now means a change in dropdown selection.
    
    grid_pil_images, target_category, solution_indices = generate_3x3_image_captcha(
        transformation_func=transform_function_to_apply
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
        session['captcha_target_category'] = target_category
        session['captcha_solution_indices'] = solution_indices
        target_category_display = target_category

        ai_selected_indices = []
        current_ai_predictions_visual = []

        for i, pil_img in enumerate(grid_pil_images):
            is_predicted_target, confidence = mock_predict_with_model(selected_attacker_model_key, pil_img.copy(), target_category)
            current_ai_predictions_visual.append({'is_selected': is_predicted_target, 'confidence': confidence})
            if is_predicted_target:
                ai_selected_indices.append(i)
        
        ai_selected_indices.sort()
        if ai_selected_indices == solution_indices:
            ai_message = f"AI ({AVAILABLE_ATTACKER_MODELS[selected_attacker_model_key]}): CORRECT! (Selected: {ai_selected_indices})"
            ai_solved_correctly = True
        else:
            ai_message = f"AI ({AVAILABLE_ATTACKER_MODELS[selected_attacker_model_key]}): INCORRECT. (AI: {ai_selected_indices}, Correct: {solution_indices})"
            ai_solved_correctly = False
        
        session['ai_predictions_visual'] = current_ai_predictions_visual 
        # session['ai_solved_correctly'] = ai_solved_correctly 
        # session['ai_message'] = ai_message 
        ai_predictions_visual = current_ai_predictions_visual


        images_data_for_current_session = []
        for i, pil_img_item in enumerate(grid_pil_images):
            img_io_bytes = io.BytesIO()
            pil_img_item.save(img_io_bytes, 'PNG')
            img_io_bytes.seek(0)
            images_data_for_current_session.append(img_io_bytes.read())
        
        if session_img_key not in TEMP_IMAGE_STORE:
            TEMP_IMAGE_STORE[session_img_key] = {}
        TEMP_IMAGE_STORE[session_img_key]['captcha_images_data_for_user'] = images_data_for_current_session
        
        image_urls_for_user = [url_for('captcha_image_for_user_grid', grid_index=i) for i in range(9)]


    if not ai_predictions_visual or len(ai_predictions_visual) != 9:
        ai_predictions_visual = [{'is_selected': False, 'confidence': 0.0}] * 9
        if grid_pil_images is None and not ("Error" in ai_message) :
             ai_message = "AI Attacker: Waiting for CAPTCHA..."


    cache_buster = random.randint(100000, 999999)

    return render_template('index_visual_attack.html',
                           # user_message no longer needed
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
    
    placeholder_img = Image.new('RGB', (80, 80), color='grey')
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