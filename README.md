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

**The AR tag requirements:**
1. Should be 8X8 
2. There is a 2 cell width solid black outer border
3. The information about the tag is contained within the internal 4×4 grid

**Sample Ar tag:**

<img width="250" alt="image" src="https://github.com/user-attachments/assets/1f91c8fe-5c25-47b2-a812-6d37bd6e55cc" />
   
## Installation

Clone the repository:
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

## Usage

### 1. Standard Execution
Navigate to the src directory and execute the main script. By default, this will run using your system's primary camera or the default test video.

 ```bash
cd src
python main.py
```
(Press q or ESC to exit the visual output window).

### 2. Setting Up Your Own Camera
To use your own external USB camera or a different video source:
1. Open `main.py` (or your specific capture script) inside the src/ directory.
2. Locate the `cv2.VideoCapture()` initialization line and change the device index or provide a video path:

   ```Python
    camera = cv2.VideoCapture(0) #open webcam
    # camera = cv2.VideoCapture('multipleTags.mp4') #use pre shot video
    ```

### 3. Using Custom 3D Models or Images
  * **3D Models**: Place your `.obj` files in the `src/` directory alongside the scripts and change the path of loading model in `main.py`

    ```python
    #3D OBJECT
    wolf = OBJ('model3.obj', swapyz=True)
    scale_factor = .10 # You may need to tune this up or down so the wolf fits perfectly!
    wolf.vertices = [[v[0]*scale_factor, v[1]*scale_factor, v[2]*scale_factor] for v in wolf.vertices]
    ```
  * **2D Overlay**: If you want to overlay a specific image, ensure the .jpg/.png is in the directory and update the relevant variable in `main.py`.

    ```python
    #2D IMAGE
    template_img = cv2.imread("flipped_quadruped.png")
    ```

### 4. Camera Calibration (For Custom Accurate Projections of 3D objects)
For the 3D models to perfectly lock onto the physical tags without jittering, accurate camera intrinsic parameters are required.
  * Run the `calibration.py` script and aim to checkerboard with your specific camera.
  * With the script running, take at least 5 photos by pressing `C`. It will output your camera's intrinsic Camera Matrix (K Matrix) and Distortion Coefficients.
  * Open `src/overlay_model.py`.
  * Locate the projection matrix configuration and replace the default K Matrix values with the newly generated one.

    ```python
    # the k matrix we got for our camera
    K = np.array([
        [518.9357033,   0.,         307.72634151],
        [  0.,         520.45365893, 212.95823523],
        [  0.,           0.,           1.        ]
    ])
    ```
  * Now you are ready to use the program for your live camera feed.
  * For prerecorded videos, you are going to need the K matrix of the camera the video was shot from.

### Sample Checkerboard:
<img width="250" alt="image" src="https://github.com/user-attachments/assets/80376d16-03cf-49c6-bbb0-1aff9d87b6cc" />


# Troubleshooting
1. Markers are not being detected: Ensure good room lighting, high contrast on the printed AR tag, and that the camera is in focus. Custom edge detection (`sobel_edge.py`) relies heavily on clear contrasts.
2. 3D model is jittery or floating: Your camera intrinsic matrix likely does not match the physical camera. Run `calibration.py` to generate accurate parameters for your specific lens. (Zhang's Method)
3. Dependencies error: Ensure all files are in the same src directory as shown in the structure, as scripts like `main.py` depend on local imports (e.g., import obj_loader).

# Contributing
Contributions, issues, and feature requests are very welcome!
