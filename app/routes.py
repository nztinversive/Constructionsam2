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

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'upload':
            # Handle file upload
            file = request.files['file']
            if file:
                filename = file.filename
                file_path = os.path.join(current_app.root_path, 'uploads', filename)
                file.save(file_path)
                return jsonify({
                    "success": True, 
                    "message": "File uploaded successfully",
                    "filename": filename,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            else:
                return jsonify({"success": False, "message": "No file uploaded"})
        elif action == 'process':
            try:
                latest_image = get_latest_image()
                if not latest_image:
                    raise ValueError("No image found for processing")
                
                image_path = os.path.join(current_app.root_path, 'uploads', latest_image)
                segmentation_result = segment_structure(image_path, "construction site")
                logger.debug(f"Segmentation result: {segmentation_result}")
                
                mask_array = segmentation_result['combined_mask']
                mask_image = Image.fromarray(mask_array)
                
                # Convert to RGB if the image is in RGBA mode
                if mask_image.mode == 'RGBA':
                    mask_image = mask_image.convert('RGB')
                
                segmentation_filename = f'segmentation_{os.path.splitext(latest_image)[0]}.png'
                mask_image.save(os.path.join(current_app.root_path, 'uploads', segmentation_filename), format='PNG')
                
                # Simulate processing by creating a copy of the original image
                processed_image_path = os.path.join(current_app.root_path, 'uploads', f'processed_{latest_image}')
                Image.open(image_path).save(processed_image_path)
                
                progress = 50.0  # Replace with actual progress calculation
                
                with open(os.path.join(current_app.root_path, 'uploads', 'latest_progress.txt'), "w") as f:
                    f.write(str(progress))
                
                return jsonify({"success": True, "message": "Processing successful", "redirect": url_for('main.results')})
            except Exception as e:
                logger.exception("An error occurred during processing")
                return jsonify({"success": False, "message": str(e)})
    
    return render_template('index.html')

@main_bp.route('/results')
def results():
    latest_image = get_latest_image()
    if not latest_image:
        return redirect(url_for('main.index'))
    
    progress = 0
    try:
        with open(os.path.join(current_app.root_path, 'uploads', 'latest_progress.txt'), "r") as f:
            progress = float(f.read())
    except:
        pass
    return render_template('results.html', image_filename=latest_image, progress=progress)

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