import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create the upload folder if it doesn't exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)