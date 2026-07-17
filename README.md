# Chris Tracker — Real-Time Object Detection & Tracking

A real-time object detection and tracking system built with YOLOv8 and ByteTrack.

## Demo
*(Demo GIF coming soon)*

## Features
- Real-time object detection via webcam
- Multi-object tracking with unique IDs (ByteTrack)
- Trajectory tracing per object
- Live FPS display
- Object class filtering

## Tech Stack
- YOLOv8 (Ultralytics)
- OpenCV
- ByteTrack
- Python 3.11

## Installation

```bash
git clone https://github.com/christianfarrell/chris-tracker1.git
cd chris-tracker1
python -m venv chris-env
source chris-env/Scripts/activate
pip install -r requirements.txt
```

## Usage

```bash
# Image detection
python week1/detect_image.py

# Video detection
python week1/detect_video.py

# Webcam real-time detection
python week2/detect_webcam.py

python week2/detect_webcam_fps.py

python week2/detect_webcam_filtre_classes.py

# Object tracking with trajectory
python week3/track_objects.py
```

## Robotics Applications
- Autonomous navigation
- Target tracking for robotic arms
- Surveillance and security systems