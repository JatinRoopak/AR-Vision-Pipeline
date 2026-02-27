# AR Vision Pipeline

## Overview
A robust Computer Vision pipeline designed to detect Augmented Reality (AR) tags and seamlessly overlay 3D models or images onto the physical environment. This project handles camera stream processing, custom marker detection, pose estimation, and 3D rendering, utilizing core computer vision principles.

## Features
- **Real-Time AR Tag Detection**: Accurately detects and decodes AR markers in video feeds either live or prerecorded.
- **Core CV Implementations**: Includes scratch-built implementations of fundamental algorithms like the Harris Corner Detector, Sobel Edge Detection, and custom convolutions.
- **3D Model & Image Overlay**: Renders customized 3D assets (`.obj`) and 2D templates onto detected markers.
- **Pose Transformation**: Calculates homographies and transformations to accurately track and anchor digital assets in physical space.

<p align="center">
  <img src="https://github.com/user-attachments/assets/3acc14d3-e70f-42c7-a46f-b54420f8fdbd" width="45%" />
  <img src="https://github.com/user-attachments/assets/1d62ac57-168d-4879-aec2-eaf4d24d409b" width="45%" />
</p>

## Prerequisites
Before running the pipeline, ensure you have the following dependencies installed:
* Python 3.8+
* OpenCV (`opencv-python`)
* NumPy
  
# Sample Ar tag:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/1f91c8fe-5c25-47b2-a812-6d37bd6e55cc" />

**The AR tag requirements:**
1. Should be 8X8 
2. There is a 2 cell width solid black outer border
3. The information about the tag is contained within the internal 4×4 grid
   
## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/JatinRoopak/AR-Vision-Pipeline.git](https://github.com/JatinRoopak/AR-Vision-Pipeline.git)
   cd AR-Vision-Pipeline
   ```
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

4. Camera Calibration (For Custom Accurate Projections)
   For the 3D models to perfectly lock onto the physical tags without jittering, accurate camera intrinsic parameters are required.
   1. Run the calibration.py script with multiple images of a standard checkerboard taken by your specific camera.
   2. The script and take atleast 5 photos by pressing "c" and it will output your camera's intrinsic Camera Matrix (K Matrix) and Distortion Coefficients.
   3. Open src/overlay_model.py.
   4. Locate the projection matrix configuration and replace the default K Matrix values with the newly generated ones to ensure the rendering matches your specific lens.
   5. Now you are ready to use the program for your camera live feed.
   6. For peerecorded videos you are gonna need the K matrix of the camera the video was shoot from.
Sample Checkerboard:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/80376d16-03cf-49c6-bbb0-1aff9d87b6cc" />

# Troubleshooting
1. Markers are not being detected: Ensure good room lighting, high contrast on the printed AR tag, and that the camera is in focus. Custom edge detection (sobel_edge.py) relies heavily on clear contrasts.
2. 3D model is jittery or floating: Your camera intrinsic matrix likely does not match the physical camera. Run calibration.py to generate accurate parameters for your specific lens. (Zhang's Method)
3. Dependencies error: Ensure all files are in the same src directory as shown in the structure, as scripts like main.py depend on local imports (e.g., import obj_loader).

# Contributing
Contributions, issues, and feature requests are very welcome!
