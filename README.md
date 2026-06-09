# Face Detection — MediaPipe Face Pipeline
> ![Python](https://img.shields.io/badge/Python-3.x-blue) | Real-time face detection, 468-point face mesh, and combined multi-model pipeline using MediaPipe.

## What I Built It For
November 2024, part of a 12-hour computer vision tutorial that was my introduction to CV. This repo covers MediaPipe's face analysis capabilities — from simple bounding box detection to dense 468-landmark face mesh tracking. The combined pipeline (`mix.py`) runs four MediaPipe models simultaneously on a single webcam feed: face mesh, face detection, hand tracking, and pose estimation. Watching all four work in parallel on consumer hardware was the moment CV clicked for me.

## Features
- Face detection with confidence-scored bounding boxes and custom corner-line rendering
- Reusable `FaceDetector` class that other scripts can import
- 468-point face mesh with custom contour connections (cleaner than the default dense wireframe)
- Combined demo running face mesh + face detection + hand tracking + pose estimation simultaneously
- Real-time FPS display

## Architecture
Three layers of face analysis, each building on the previous:

| Script | What It Does | Model |
|--------|-------------|-------|
| `facedetection.py` | Detects faces, draws bounding boxes with confidence % | MediaPipe Face Detection |
| `facedetection_module.py` | Reusable class wrapping face detection | MediaPipe Face Detection |
| `facemesh.py` | 468 landmarks + custom contour lines | MediaPipe Face Mesh |
| `facemesh_module.py` | Reusable class wrapping face mesh | MediaPipe Face Mesh |
| `mix.py` | All four pipelines running simultaneously | Face Mesh + Face Detection + Hands + Pose |

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Computer Vision | OpenCV (cv2) |
| Face Analysis | MediaPipe (Face Detection, Face Mesh) |
| Body/Hand Tracking | MediaPipe (Pose, Hands) |
| Language | Python 3 |

## Setup & Usage

### Prerequisites
- Python 3.x
- A webcam or local video files

### Installation
```bash
git clone https://github.com/ShubhxYT/Face-Detection.git
cd Face-Detection
pip install opencv-python mediapipe
```

### Running
```bash
# Face detection with bounding boxes
python facedetection.py

# Face detection using the reusable module
python facedetection_module.py

# 468-point face mesh with custom contour
python facemesh.py

# Combined pipeline — all four models at once
python mix.py
```

Press `q` to quit any script.

## Project Structure
```
├── facedetection.py          # Standalone face detection demo
├── facedetection_module.py   # Reusable FaceDetector class
├── facemesh.py               # Face mesh with custom contour rendering
├── facemesh_module.py        # Reusable FaceMeshDetector class
├── mix.py                    # Combined pipeline — face + hands + pose
├── videos/                   # Test video files
```
