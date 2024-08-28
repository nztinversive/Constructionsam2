from flask import Blueprint, render_template, request, flash, send_file, redirect, url_for, jsonify, current_app
from modules.sam_integration import segment_structure
from modules.image_processing import preprocess_image
import cv2
import numpy as np
import os
import logging
import traceback
import requests
from PIL import Image
import io
import time
import json
from datetime import datetime
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

def save_progress(image_filename, progress):
    progress_file = os.path.join(current_app.root_path, 'uploads', 'progress.json')
    try:
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
    except FileNotFoundError:
        progress_data = {}

    progress_data[image_filename] = {
        'progress': progress,
        'timestamp': datetime.now().isoformat()
    }

    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def get_progress_data():
    progress_file = os.path.join(current_app.root_path, 'uploads', 'progress.json')
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'upload':
            uploaded_files = request.files.getlist('file')
            if uploaded_files:
                filenames = []
                for file in uploaded_files:
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(current_app.root_path, 'uploads', filename)
                        file.save(file_path)
                        filenames.append(filename)
                return jsonify({
                    "success": True, 
                    "message": "Files uploaded successfully",
                    "filenames": filenames,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                return jsonify({"success": False, "message": "No files uploaded"})
        elif action == 'process':
            try:
                latest_images = get_latest_images()
                if not latest_images:
                    raise ValueError("No images found for processing")
                
                for image in latest_images:
                    image_path = os.path.join(current_app.root_path, 'uploads', image)
                    segmentation_result = segment_structure(image_path, "construction site")
                    logger.debug(f"Segmentation result for {image}: {segmentation_result}")
                    
                    mask_array = segmentation_result['combined_mask']
                    mask_image = Image.fromarray(mask_array)
                    
                    if mask_image.mode == 'RGBA':
                        mask_image = mask_image.convert('RGB')
                    
                    segmentation_filename = f'segmentation_{os.path.splitext(image)[0]}.png'
                    mask_image.save(os.path.join(current_app.root_path, 'uploads', segmentation_filename), format='PNG')
                    
                    progress = 50.0  # Replace with actual progress calculation
                    save_progress(image, progress)
                
                return jsonify({"success": True, "message": "Processing successful", "redirect": url_for('main.results')})
            except Exception as e:
                logger.exception("An error occurred during processing")
                return jsonify({"success": False, "message": str(e)})
    
    return render_template('index.html')

@main_bp.route('/results')
def results():
    latest_images = get_latest_images()
    if not latest_images:
        return redirect(url_for('main.index'))
    
    progress_data = get_progress_data()
    images = []
    for image in latest_images:
        progress = progress_data.get(image, {}).get('progress', 0)
        images.append({'filename': image, 'progress': progress})
    
    return render_template('results.html', images=images)

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    uploads_dir = os.path.join(current_app.root_path, 'uploads')
    
    # Check for both .jpg and .png extensions
    for ext in ['.jpg', '.png']:
        file_path = os.path.join(uploads_dir, os.path.splitext(filename)[0] + ext)
        if os.path.exists(file_path):
            return send_file(file_path)
    
    # If neither file exists, return a 404 error
    return "File not found", 404

def get_latest_image():
    uploads_dir = os.path.join(current_app.root_path, 'uploads')
    image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith(('processed_', 'segmentation_'))]
    if image_files:
        return max(image_files, key=lambda x: os.path.getctime(os.path.join(uploads_dir, x)))
    return None

def get_latest_images(limit=5):
    uploads_dir = os.path.join(current_app.root_path, 'uploads')
    image_files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith(('processed_', 'segmentation_'))]
    return sorted(image_files, key=lambda x: os.path.getctime(os.path.join(uploads_dir, x)), reverse=True)[:limit]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@main_bp.route('/progress_data')
def progress_data():
    return jsonify(get_progress_data())