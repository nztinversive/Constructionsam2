import base64
import cv2
import numpy as np
import os
import requests
import logging
from PIL import Image
import io
from dotenv import load_dotenv
import time
import random

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def resize_image(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        if h > w:
            new_h, new_w = max_size, int(max_size * w / h)
        else:
            new_h, new_w = int(max_size * h / w), max_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def segment_structure(image_path: str, prompt: str, max_retries=3) -> dict:
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

    for attempt in range(max_retries):
        try:
            logger.debug(f"Attempt {attempt + 1} of {max_retries}")
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image = resize_image(image)
            
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {image.shape}")

            _, buffer = cv2.imencode('.png', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            payload = {
                "version": "fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
                "input": {
                    "image": f"data:image/png;base64,{img_base64}",
                    "use_m2m": True,
                    "points_per_side": 32,
                    "pred_iou_thresh": 0.88,
                    "stability_score_thresh": 0.95
                }
            }

            headers = {
                "Authorization": f"Token {api_token}",
                "Content-Type": "application/json"
            }
            response = requests.post("https://api.replicate.com/v1/predictions", json=payload, headers=headers)
            response.raise_for_status()
            prediction = response.json()

            logger.debug(f"Prediction started: {prediction['id']}")

            prediction_url = prediction['urls']['get']
            start_time = time.time()
            while prediction['status'] not in ["succeeded", "failed", "canceled"]:
                if time.time() - start_time > 300:
                    raise Exception("Prediction timed out after 5 minutes")
                time.sleep(1)
                response = requests.get(prediction_url, headers=headers)
                response.raise_for_status()
                prediction = response.json()

            if prediction['status'] == "failed":
                error_message = prediction.get('error', 'Unknown error occurred')
                if "CUDA out of memory" in error_message:
                    raise Exception("The server is currently overloaded. Retrying...")
                raise Exception(f"Prediction failed: {error_message}")
            elif prediction['status'] == "canceled":
                raise Exception("Prediction was canceled")

            logger.debug(f"Prediction completed: {prediction['output']}")

            mask_url = prediction['output']['combined_mask']
            mask_array = download_mask(mask_url)
            
            return {"combined_mask": mask_array}
        except Exception as e:
            logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.random()
                logger.info(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise

    raise Exception("Max retries reached. Unable to process the image.")

def download_mask(url: str) -> np.ndarray:
    response = requests.get(url)
    mask_image = Image.open(io.BytesIO(response.content))
    mask_array = np.array(mask_image)
    
    # Ensure the mask is binary (0 or 255)
    mask_array = (mask_array > 128).astype(np.uint8) * 255
    
    return mask_array

def test_segment_structure():
    """
    Test function to verify SAM 2 integration via Replicate API.
    """
    # Load a test image
    test_image_path = "path/to/your/test/image.jpg"
    image = cv2.imread(test_image_path)
    
    if image is None:
        print(f"Error: Could not load test image from {test_image_path}")
        return
    
    try:
        segmentation = segment_structure(test_image_path, "construction site")
        print("Segmentation successful!")
        print(f"Segmentation shape: {segmentation.shape}")
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")

if __name__ == "__main__":
    test_segment_structure()