from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

def create_app():
    app = Flask(__name__, template_folder='app/templates')
    app.secret_key = os.environ.get('SECRET_KEY') or 'your_fallback_secret_key_here'

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)