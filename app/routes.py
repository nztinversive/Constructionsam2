from flask import Blueprint, render_template, request, flash, send_file, redirect, url_for
from modules.sam_integration import segment_structure, download_mask
from modules.image_processing import preprocess_image, align_images
from modules.progress_tracking import calculate_progress
from modules.reporting import generate_report, update_web_interface
import cv2
import numpy as np
import os
from config import Config
import logging
import traceback

main_bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@main_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return render_template('index.html')
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return render_template('index.html')
        if file:
            try:
                # Read and preprocess the image
                image_bytes = file.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                logger.debug(f"Original image shape: {image.shape}")
                
                preprocessed_image = preprocess_image(image)
                logger.debug(f"Preprocessed image shape: {preprocessed_image.shape}")
                
                # Segment the image
                segmentation_result = segment_structure(preprocessed_image, "construction site")
                logger.debug(f"Segmentation result: {segmentation_result}")
                
                if 'combined_mask' not in segmentation_result:
                    raise ValueError("Segmentation result does not contain 'combined_mask'")
                
                combined_mask_url = segmentation_result['combined_mask']
                combined_mask = download_mask(combined_mask_url)
                logger.debug(f"Combined mask shape: {combined_mask.shape}")
                
                # Resize the mask to match the preprocessed image size
                combined_mask = cv2.resize(combined_mask, (preprocessed_image.shape[1], preprocessed_image.shape[0]))
                logger.debug(f"Resized mask shape: {combined_mask.shape}")
                
                # Calculate progress
                previous_segmentation = get_previous_segmentation()
                if previous_segmentation is not None:
                    logger.debug(f"Previous segmentation shape: {previous_segmentation.shape}")
                    progress = calculate_progress(previous_segmentation, combined_mask)
                else:
                    progress = 0.0
                logger.debug(f"Calculated progress: {progress}")
                
                # Save current data for future comparison
                save_current_data(preprocessed_image, combined_mask)
                
                # Create the static directory if it doesn't exist
                os.makedirs('static', exist_ok=True)
                
                # Update progress file
                with open("static/latest_progress.txt", "w") as f:
                    f.write(str(progress))
                
                flash('Processing successful!')
                return redirect(url_for('main.results'))
            except Exception as e:
                logger.exception("An error occurred during processing")
                error_traceback = traceback.format_exc()
                logger.error(f"Full traceback:\n{error_traceback}")
                flash(f"Error during processing: {str(e)}")
    return render_template('index.html')

@main_bp.route('/results')
def results():
    progress = 0.0
    if os.path.exists("static/latest_progress.txt"):
        with open("static/latest_progress.txt", "r") as f:
            progress = float(f.read())
    return render_template('results.html', progress=progress)

@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(Config.UPLOAD_FOLDER, filename))

@main_bp.route('/static/<filename>')
def static_file(filename):
    return send_file(os.path.join('static', filename))

def get_previous_segmentation():
    if os.path.exists("static/latest_segmentation.png"):
        return cv2.imread("static/latest_segmentation.png", cv2.IMREAD_GRAYSCALE)
    return None

def save_current_data(image, segmentation):
    os.makedirs('static', exist_ok=True)
    cv2.imwrite("static/latest_image.png", image)
    cv2.imwrite("static/latest_segmentation.png", segmentation)