import base64
import cv2
import numpy as np
import replicate
import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def segment_structure(image: np.ndarray, prompt: str) -> dict:
    # Check if the image is 2D (grayscale) or 3D (color)
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Encode image to base64
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Set up the Replicate client
    replicate_api_token = os.environ.get('REPLICATE_API_TOKEN')
    if not replicate_api_token:
        raise ValueError("REPLICATE_API_TOKEN environment variable is not set")

    client = replicate.Client(api_token=replicate_api_token)

    # Run the model
    output = client.run(
        "meta/sam-2:fe97b453a6455861e3bac769b441ca1f1086110da7466dbb65cf1eecfd60dc83",
        input={
            "image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "output_format": "mask"
        }
    )

    return output  # Return the dictionary of URLs

def download_mask(url: str) -> np.ndarray:
    response = requests.get(url)
    mask_image = Image.open(BytesIO(response.content))
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
        segmentation = segment_structure(image, "construction site")
        print("Segmentation successful!")
        print(f"Segmentation shape: {segmentation.shape}")
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")

if __name__ == "__main__":
    test_segment_structure()