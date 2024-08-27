from typing import List
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from io import BytesIO
import cv2

def generate_report(images: List[np.ndarray], segmentations: List[np.ndarray], progress: List[float]) -> bytes:
    """
    Generate a PDF report with images, segmentations, and progress information.
    
    Args:
        images (List[np.ndarray]): List of input images.
        segmentations (List[np.ndarray]): List of segmentation masks.
        progress (List[float]): List of progress percentages.
    
    Returns:
        bytes: PDF report as bytes.
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    for i, (image, segmentation, prog) in enumerate(zip(images, segmentations, progress)):
        c.drawString(100, height - 100, f"Timepoint {i+1}")
        c.drawString(100, height - 120, f"Progress: {prog:.2f}%")
        
        # Convert image and segmentation to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmentation_rgb = np.zeros_like(image_rgb)
        segmentation_rgb[segmentation > 0] = [255, 0, 0]  # Red color for segmentation
        
        # Blend image and segmentation
        blended = cv2.addWeighted(image_rgb, 0.7, segmentation_rgb, 0.3, 0)
        
        # Save blended image to a temporary file
        temp_image_path = f"temp_image_{i}.png"
        cv2.imwrite(temp_image_path, blended)
        
        # Draw image on PDF
        c.drawImage(temp_image_path, 100, height - 500, width=400, height=300)
        
        c.showPage()
    
    c.save()
    return buffer.getvalue()

def update_web_interface(report: bytes, latest_image: np.ndarray, latest_progress: float) -> None:
    """
    Update the web interface with the latest report, image, and progress.
    
    Args:
        report (bytes): Generated PDF report.
        latest_image (np.ndarray): Latest processed image.
        latest_progress (float): Latest calculated progress.
    """
    # Save the report to a file
    with open("static/latest_report.pdf", "wb") as f:
        f.write(report)
    
    # Save the latest image
    cv2.imwrite("static/latest_image.png", latest_image)
    
    # Save the latest progress to a file
    with open("static/latest_progress.txt", "w") as f:
        f.write(f"{latest_progress:.2f}")