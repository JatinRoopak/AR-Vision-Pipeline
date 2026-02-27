# AR Vision Pipeline

## Overview
A robust Computer Vision pipeline designed to detect Augmented Reality (AR) tags and seamlessly overlay 3D models or images onto the physical environment. This project handles camera stream processing, custom marker detection, pose estimation, and 3D rendering, utilizing core computer vision principles.

## Features
- **Real-Time AR Tag Detection**: Accurately detects and decodes AR markers in video feeds either live or prerecorded.
- **Core CV Implementations**: Includes scratch-built implementations of fundamental algorithms like the Harris Corner Detector, Sobel Edge Detection, and custom convolutions.
- **3D Model & Image Overlay**: Renders customized 3D assets (`.obj`) and 2D templates onto detected markers.
- **Pose Transformation**: Calculates homographies and transformations to accurately track and anchor digital assets in physical space.

## Prerequisites
Before running the pipeline, ensure you have the following dependencies installed:
* Python 3.8+
* OpenCV (`opencv-python`)
* NumPy
  
# Sample Ar tag:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/1f91c8fe-5c25-47b2-a812-6d37bd6e55cc" />

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/JatinRoopak/AR-Vision-Pipeline.git](https://github.com/JatinRoopak/AR-Vision-Pipeline.git)
   cd AR-Vision-Pipeline
   ```
Install the required dependencies:

## Directory Structure
 ```bash
AR-Vision-Pipeline/
├── .gitignore
└── src/
    ├── calibration.py       # Camera calibration logic
    ├── convolution.py       # Custom convolution operations
    ├── decode_tags.py       # AR tag decoding and ID extraction
    ├── detect_tags.py       # Tag detection algorithms
    ├── filter_tags.py       # Tag filtering and validation
    ├── harris_corner.py     # Harris Corner Detection implementation
    ├── main.py              # Main execution script
    ├── obj_loader.py        # 3D model (.obj) parsing and loading
    ├── overlay_img.py       # 2D image overlay logic
    ├── overlay_model.py     # 3D model rendering and overlay
    ├── sobel_edge.py        # Sobel edge detection implementation
    ├── testing.py           # Testing and utility scripts
    ├── transformation.py    # Homography and pose transformations
    └── [Assets]             # 3D models (.obj), templates (.jpg/.png), and test videos (.mp4)
```
Usage
1. Standard Execution
Navigate to the src directory and execute the main script. By default, this will run using your system's primary camera or the default test video.

 ```bash
cd src
python main.py
```
(Press q or ESC to exit the visual output window).

2. Setting Up Your Own Camera
To use your own external USB camera or a different video source:

Open main.py (or your specific capture script) inside the src/ directory.

Locate the cv2.VideoCapture() initialization line and change the device index or provide a video path:

```bash
Python
# 0 is usually the built-in webcam. Change to 1, 2, etc., for external cameras
cap = cv2.VideoCapture(0) 

# Or use a local video file (e.g., the included multipleTags.mp4)
cap = cv2.VideoCapture('multipleTags.mp4')

# Or use a local video file (e.g., the included multipleTags.mp4)
cap = cv2.VideoCapture('multipleTags.mp4')
```

3. Using Custom 3D Models or Images
3D Models: Place your .obj files in the src/ directory alongside the scripts (or create a dedicated assets folder). Update the file path in your code where obj_loader.py or overlay_model.py is called.

2D Overlay: If you want to overlay a specific image, ensure the .jpg/.png is in the directory and update the relevant variable in overlay_img.py or main.py.

4. Camera Calibration (For Accurate Projections)
For the 3D models to perfectly lock onto the physical tags without jittering, accurate camera intrinsic parameters are required.

# Sample checker board
<img width="250" alt="image" src="https://github.com/user-attachments/assets/80376d16-03cf-49c6-bbb0-1aff9d87b6cc" />


Run the calibration.py script with images of a standard checkerboard taken by your specific camera.

Take the resulting Camera Matrix and Distortion Coefficients and update the corresponding arrays in your code (likely within transformation.py or main.py).

Troubleshooting
Markers are not being detected: Ensure good room lighting, high contrast on the printed AR tag, and that the camera is in focus. Custom edge detection (sobel_edge.py) relies heavily on clear contrasts.

3D model is jittery or floating: Your camera intrinsic matrix likely does not match the physical camera. Run calibration.py to generate accurate parameters for your specific lens. (Zhang's Method)

Dependencies error: Ensure all files are in the same src directory as shown in the structure, as scripts like main.py depend on local imports (e.g., import obj_loader).

Contributing
Contributions, issues, and feature requests are very welcome!
