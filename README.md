# Stereo Vision Project

This project implements a stereo vision system using YOLO for object detection. It processes video frames to detect objects and calculate distances between them using stereo vision techniques.

## Features

- **Object Detection**: Utilizes the YOLO model for detecting objects in video frames.
- **Distance Calculation**: Computes distances between detected objects using stereo vision principles.
- **Real-time Processing**: Processes video streams in real-time using multithreading.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stereo-vision.git
   cd stereo-vision
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Load the YOLO model:
   ```python
   model = load_model('yolov5su.pt')
   ```

2. Process video frames:
   ```python
   cap = cv2.VideoCapture(0)  # or path to video file
   frame_queue = queue.Queue()
   video_stream(cap, frame_queue)
   ```

3. Calculate distances:
   ```python
   distance = calculate_distance(box1, box2, focal_length, baseline)
   ```

## Files

- `main.py`: Main script for processing video frames and detecting objects.
- `yolov5su.pt` and `yolov5nu.pt`: Pre-trained YOLO model weights.
- `stereo v1.pdf`: model dimension file.
- `Body1.stl`: 3D model file, possibly used for visualization or simulation.


## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/yolov5) for the object detection model. 
