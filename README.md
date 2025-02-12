Air Canvas
Air Canvas is an interactive air painting application that leverages hand tracking and depth estimation to let you paint in mid-air using simple gestures. Built with OpenCV, MediaPipe, and PyTorch, the project offers an intuitive interface where you can select colors, change brush sizes, and even clear the canvas—all using your hand movements in front of a webcam.

Features
Air Painting: Draw on a virtual canvas by simply moving your hand.
Gesture-Based Controls:
Drawing: Use a pinching gesture (thumb and index finger) to draw.
UI Interaction: Move your hand to the right side of the screen to select colors, adjust brush sizes, or clear the canvas.
Real-Time Depth Estimation: Uses a pre-trained deep learning model (DepthAnythingV2) to display a depth map of the scene.
Customizable Settings: Easily adjust parameters like brush sizes, colors, and input dimensions through a YAML configuration file.
Multi-threaded Performance: Runs depth estimation in a separate thread to maintain smooth drawing performance.
Demo
(Include screenshots or a video demo here if available.)

Installation
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your_username/air-canvas.git
cd air-canvas
2. (Optional) Create a Virtual Environment
It is recommended to use a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
Install the required packages via pip. You can create a requirements.txt file with the following content (adjust versions as needed):

text
Copy
Edit
opencv-python
numpy
mediapipe
torch
pyyaml
Then run:

bash
Copy
Edit
pip install -r requirements.txt
4. Download the Pre-trained Model
Ensure that you have the pre-trained model file (depth_anything_v2_vits.pth) in the model_weights directory. If you don’t have it yet, please follow the instructions on the DepthAnythingV2 repository or the relevant source to obtain the model weights.

Usage
Run the application using:

bash
Copy
Edit
python your_script_name.py
Note: Replace your_script_name.py with the name of the main script (if different).

Controls
Drawing:
To draw, bring your thumb and index finger together so that the pinch distance is greater than the set threshold.
UI Interaction:
Move your hand to the right side of the screen to interact with the on-screen color palette, brush size options, and clear button.
Exit:
Press 'q' to quit the application.
Configuration
The application settings are defined in a YAML configuration file (config.yaml). Here is an example configuration:

yaml
Copy
Edit
device: "cuda"                # Use 'cuda' if available; otherwise, CPU will be used.
model_dir: "./model_weights"  # Directory containing the model weights.
model_filename: "depth_anything_v2_vits.pth"
model_type: "vits"            # Options: vits, vitb, vitl
input_size: 256               # Input size for the depth estimation model.
pen_down_threshold: 120       # Threshold for the pinch distance to start drawing.
window_width: 1280            # Width of the display window.
window_height: 720            # Height of the display window.
max_points: 1024              # Maximum number of points per drawing stroke.
brush_sizes: [5, 10, 15, 20]  # List of available brush sizes.
colors:
  - [255, 255, 255]           # White
  - [0, 0, 0]                 # Black
  - [255, 0, 0]               # Red
  - [0, 255, 0]               # Green
  - [0, 0, 255]               # Blue
  - [255, 255, 0]             # Yellow
You can modify this file to adjust the behavior and appearance of the application.

Dependencies
Python 3.7+
OpenCV: For image processing and displaying the canvas.
NumPy: For numerical operations.
MediaPipe: For real-time hand tracking.
PyTorch: For running the depth estimation model.
PyYAML: For parsing the configuration file.
Contributing
Contributions are welcome! If you have suggestions, bug fixes, or new features, please fork the repository and submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
MediaPipe: For providing the robust hand-tracking solution.
DepthAnythingV2: For the depth estimation model and code.
OpenCV Community: For the extensive computer vision tools.
And thanks to all the open-source contributors whose work made this project possible.
