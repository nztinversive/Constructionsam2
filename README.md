# Construction Progress Monitor

## Description
Construction Progress Monitor is a web application that uses AI-powered image analysis to track and visualize construction progress over time. It allows users to upload images of construction sites, processes them using the Segment Anything Model (SAM), and displays the results in an interactive interface.

## Features
- Image upload and processing
- AI-powered segmentation of construction site images
- Interactive image comparison slider
- Filtering and sorting of processed images
- Dark mode toggle
- Responsive design

## Technologies Used
- Backend:
  - Python
  - Flask
  - OpenCV
  - NumPy
  - Pillow
- Frontend:
  - HTML
  - CSS (Tailwind CSS)
  - JavaScript (jQuery)
  - Flatpickr (for date picking)
- AI Model:
  - Segment Anything Model (SAM) via Replicate API

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nztinversive/Constructionsam2.git
   cd Constructionsam2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```bash
   REPLICATE_API_TOKEN=your_replicate_api_token
   SECRET_KEY=your_secret_key
   ```

## Usage

1. Run the Flask application:
   ```bash
   python main.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Upload construction site images and use the interface to track progress over time.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## GitHub Repository
[https://github.com/nztinversive/Constructionsam2](https://github.com/nztinversive/Constructionsam2)

## TODO

- Implement image alignment algorithm in `modules/image_processing.py`
- Refine SAM 2 integration in `modules/sam_integration.py`
- Implement a more sophisticated progress calculation algorithm in `modules/progress_tracking.py`
- Add logic to draw images and segmentations on the PDF report in `modules/reporting.py`
- Implement logic to update the web interface in `modules/reporting.py`
- Add logic to fetch and display results in the `results` route in `app/routes.py`

## Disclaimer

This is a prototype implementation and may require further refinement and error handling for production use.
