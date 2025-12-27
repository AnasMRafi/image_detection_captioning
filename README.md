# AI Vision - Image Captioning & Object Detection

A mobile AI application that combines **deep learning image captioning** with **real-time object detection** using a Flutter frontend and FastAPI backend.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flutter](https://img.shields.io/badge/Flutter-3.0+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Image Captioning
- CNN-LSTM architecture with **Bahdanau Attention**
- **InceptionV3** encoder (pretrained on ImageNet)
- Generates natural language descriptions of images
- Attention visualization showing which image regions influence each word

### Object Detection
- **YOLOv12** for state-of-the-art detection (80 COCO classes)
- Real-time inference with bounding box visualization
- High accuracy (55.2% mAP on COCO)

### Detection-Guided Captioning
- Combines detected objects with image captioning
- Object context improves caption accuracy
- Unified `/analyze` endpoint for both tasks

### Mobile Application
- Cross-platform Flutter app (iOS & Android)
- Camera capture and gallery upload
- Beautiful glassmorphism UI with animations
- History with Firebase Cloud Firestore

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flutter Mobile App                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │  Captioning │ │  Detection  │ │     AI Analysis         ││
│  │   Screen    │ │   Screen    │ │ (Caption + Detection)   ││
│  └──────┬──────┘ └──────┬──────┘ └───────────┬─────────────┘│
│         │               │                     │              │
│         └───────────────┴─────────────────────┘              │
│                         │ HTTP                               │
└─────────────────────────┼───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    /caption                              ││
│  │   Image → InceptionV3 → LSTM + Attention → Caption       ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    /detect                               ││
│  │   Image → YOLOv12 → Bounding Boxes + Labels              ││
│  └──────────────────────────────────────────────────────────┘│
│  ┌──────────────────────────────────────────────────────────┐│
│  │                    /analyze                              ││
│  │   Image → YOLO + Captioning → Combined Result            ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core language |
| **FastAPI** | REST API framework |
| **PyTorch** | Deep learning framework |
| **Ultralytics** | YOLOv12 object detection |
| **Pillow** | Image processing |

### Frontend
| Technology | Purpose |
|------------|---------|
| **Flutter 3.0+** | Cross-platform mobile framework |
| **Dart** | Programming language |
| **Firebase** | Cloud Firestore for history |
| **Provider** | State management |

### Models
| Model | Task | Architecture |
|-------|------|--------------|
| **Encoder** | Feature extraction | InceptionV3 (pretrained) |
| **Decoder** | Caption generation | LSTM + Bahdanau Attention |
| **Detector** | Object detection | YOLOv12x |

---

## Project Structure

```
Flutter_DL_Img_Captioning/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── config.py               # Configuration
│   ├── utils.py                # Utility functions
│   ├── image_captioning.ipynb  # Training notebook
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── lib/
│   │   ├── main.dart           # App entry point
│   │   ├── ui/                 # Screens (home, captioning, detection, analysis, history)
│   │   ├── services/           # API, Firebase, Detection services
│   │   └── models/             # Data models
│   └── pubspec.yaml
│
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- Flutter 3.0+
- Firebase project (for history feature)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/AnasMRafi/image_detection_captioning.git
cd image_detection_captioning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (see Model Downloads section)

# Run the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend Setup

```bash
cd frontend

# Get Flutter dependencies
flutter pub get

# Configure Firebase
# - Add google-services.json to android/app/
# - Add GoogleService-Info.plist to ios/Runner/

# Run on device/emulator
flutter run
```

### Model Downloads

Model weights are not included in the repository due to size. Download from:

| File | Size | Description |
|------|------|-------------|
| `encoder.pth` | ~90 MB | InceptionV3 encoder |
| `decoder.pth` | ~50 MB | LSTM decoder with attention |
| `vocab.pkl` | ~100 KB | Vocabulary file |
| `yolo12x.pt` | ~130 MB | YOLOv12 weights (auto-downloads) |

Place downloaded files in `backend/models/`

---

## Usage

### Starting the Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

### Running the Mobile App

1. Update the server URL in the app settings
2. Use your machine's local IP (e.g., `http://192.168.1.x:8000`)
3. Ensure phone and computer are on the same network

### Testing with curl

```bash
# Caption an image
curl -X POST "http://localhost:8000/caption" -F "image=@test.jpg"

# Detect objects
curl -X POST "http://localhost:8000/detect" -F "image=@test.jpg"

# Combined analysis
curl -X POST "http://localhost:8000/analyze" -F "image=@test.jpg"
```

---

## Model Training

### Dataset
- **Flickr8k**: 8,000 images with 5 captions each
- Download from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### Training on Kaggle

1. Upload `captioning_notebook_kaggle.py` to Kaggle
2. Enable GPU (P100 recommended)
3. Run all cells
4. Download generated weights:
   - `encoder.pth`
   - `decoder.pth`
   - `vocab.pkl`

### Hyperparameters

```python
EMBEDDING_DIM = 256
ENCODER_DIM = 2048
DECODER_DIM = 512
ATTENTION_DIM = 256
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 30
```

---

## API Endpoints

### `POST /caption`
Generate a caption for an image.

**Request:**
```
Content-Type: multipart/form-data
Body: image (file)
```

**Response:**
```json
{
  "success": true,
  "caption": "a dog running through the grass",
  "confidence": 0.85,
  "inference_time_ms": 150
}
```

### `POST /detect`
Detect objects in an image.

**Response:**
```json
{
  "success": true,
  "detections": [
    {"class_name": "dog", "score": 0.95, "box": [x1, y1, x2, y2]}
  ],
  "inference_time_ms": 80
}
```

### `POST /analyze`
Combined captioning and detection.

**Response:**
```json
{
  "success": true,
  "caption": "a dog running through the grass",
  "detections": [...],
  "combined_description": "a dog running through the grass (Detected: dog)",
  "inference_time_ms": 200
}
```

---

## Authors

- **Anas M Rafi** - [GitHub](https://github.com/AnasMRafi)

