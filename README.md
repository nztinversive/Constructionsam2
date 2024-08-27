# Construction Progress Monitor

This project is a prototype for monitoring construction progress using drone imagery and the SAM 2 (Segment Anything Model 2) for image segmentation.

## Features

- Process and align drone images
- Segment structural elements using SAM 2
- Calculate construction progress
- Generate reports
- Provide a web interface for image upload and result viewing

## Prerequisites

- Python 3.7+
- pip

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/construction-progress-monitor.git
   cd construction-progress-monitor
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the SAM 2 model checkpoint and place it in the project directory.

5. Set up the environment variables:
   ```bash
   export SECRET_KEY=your-secret-key
   export REPLICATE_API_TOKEN=your_replicate_api_token_here
   ```
   Or create a `.env` file in the project root with these variables.

## Configuration

Update the `config.py` file to set your desired configuration:

- `SECRET_KEY`: A secret key for the application (set as an environment variable)
- `UPLOAD_FOLDER`: The directory where uploaded images will be stored
- `ALLOWED_EXTENSIONS`: The allowed file extensions for uploaded images

## Running the Application

1. Start the Flask development server:
   ```bash
   python main.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. Upload drone images through the web interface to process and monitor construction progress.

## Project Structure

- `app/`: Contains the Flask application and routes
- `modules/`: Contains the core functionality modules
- `utils/`: Contains utility functions
- `main.py`: The main entry point of the application
- `config.py`: Configuration settings for the application

## TODO

- Implement image alignment algorithm in `modules/image_processing.py`
- Refine SAM 2 integration in `modules/sam_integration.py`
- Implement a more sophisticated progress calculation algorithm in `modules/progress_tracking.py`
- Add logic to draw images and segmentations on the PDF report in `modules/reporting.py`
- Implement logic to update the web interface in `modules/reporting.py`
- Add logic to fetch and display results in the `results` route in `app/routes.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Disclaimer

This is a prototype implementation and may require further refinement and error handling for production use.#   C o n s t r u c t i o n s a m 2  
 #   C o n s t r u c t i o n s a m 2  
 