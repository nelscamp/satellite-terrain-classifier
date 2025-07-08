# 🛰️ Satellite Terrain Classification

A high-performance deep learning system for classifying satellite imagery into 21 different terrain types using a ResNet50-based CNN architecture. Achieving **96.67% validation accuracy** on the UC Merced Land Use dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Performance](#performance)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🌍 Overview

This project implements a state-of-the-art convolutional neural network for automated classification of satellite imagery. The system can distinguish between 21 different terrain types including agricultural land, urban areas, natural features, and infrastructure.

### Supported Terrain Classes

The model classifies satellite images into the following 21 categories:

| Natural Terrain | Urban/Infrastructure | Transportation | Recreation |
|----------------|---------------------|----------------|-----------|
| Agricultural | Buildings | Airplane | Baseball Diamond |
| Beach | Dense Residential | Freeway | Golf Course |
| Chaparral | Medium Residential | Intersection | Tennis Court |
| Forest | Sparse Residential | Overpass | |
| River | Mobile Home Park | Parking Lot | |
| | Storage Tanks | Runway | |
| | Harbor | | |

## ✨ Features

- **High Accuracy**: 96.67% validation accuracy
- **Fast Inference**: ~50-100ms per image
- **Production-Ready API**: RESTful FastAPI with comprehensive documentation
- **Docker Support**: Containerized deployment
- **Transfer Learning**: ResNet50 backbone pre-trained on ImageNet
- **Data Augmentation**: Robust training with geometric transformations
- **Health Monitoring**: Built-in health checks for production deployment
- **Interactive Web UI**: User-friendly interface for image classification

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 96.67% |
| **Training Accuracy** | 92.62% |
| **Model Parameters** | 24.5M total (1.5M trainable) |
| **Inference Time** | 50-100ms |
| **Dataset Size** | 2,100 images (100 per class) |
| **Input Resolution** | 256×256 RGB |

### Training Results
- **Final Training Loss**: 0.3455
- **Final Validation Loss**: 0.1484
- **Training completed in**: 98 seconds (2 epochs shown in example)

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/satellite-terrain-classification.git
cd satellite-terrain-classification

# Build and run with Docker
docker build -t satellite-classifier .
docker run -p 8000:8000 satellite-classifier
```

### Local Installation

```bash
# Clone repository
git clone https://github.com/yourusername/satellite-terrain-classification.git
cd satellite-terrain-classification

# Create virtual environment
python -m venv satellite_env
source satellite_env/bin/activate  # On Windows: satellite_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API server
python src/api/app.py
```

Visit `http://localhost:8000` to access the web interface or `http://localhost:8000/docs` for API documentation.

## 💾 Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, for training)
- 4GB+ RAM
- 2GB+ disk space

### Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch>=2.5.1` - Deep learning framework
- `torchvision>=0.20.1` - Computer vision utilities
- `fastapi>=0.116.0` - Web API framework
- `Pillow>=11.3.0` - Image processing
- `scikit-learn>=1.7.0` - Data preprocessing
- `uvicorn>=0.35.0` - ASGI server

## 🎯 Usage

### Python API

```python
import torch
from PIL import Image
from src.models.satellite_cnn import SatelliteCNN
import torchvision.transforms as transforms

# Load pre-trained model
model = SatelliteCNN(num_classes=21)
model.load_state_dict(torch.load('notebooks/best_satellite_terrain_model.pth'))
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classify image
image = Image.open('path/to/satellite/image.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted = torch.max(probabilities, 0)

print(f"Predicted class: {class_names[predicted.item()]}")
print(f"Confidence: {confidence.item():.4f}")
```

### REST API

```bash
# Health check
curl http://localhost:8000/health

# Classify image
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@satellite_image.jpg"
```

### Web Interface

Navigate to `http://localhost:8000` for an interactive web interface where you can upload images and get instant classifications.

## 📚 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive web interface |
| `GET` | `/health` | Health check for load balancers |
| `POST` | `/predict` | Image classification endpoint |
| `GET` | `/model-info` | Detailed model information |
| `GET` | `/docs` | Swagger API documentation |

### Response Format

```json
{
  "status": "success",
  "prediction": {
    "class": "forest",
    "confidence": 0.9567
  },
  "top_3_predictions": [
    {
      "class": "forest",
      "probability": 0.9567,
      "confidence_level": "high"
    },
    {
      "class": "chaparral",
      "probability": 0.0234,
      "confidence_level": "low"
    },
    {
      "class": "agricultural",
      "probability": 0.0123,
      "confidence_level": "low"
    }
  ],
  "metadata": {
    "filename": "satellite_image.jpg",
    "image_size": [256, 256],
    "inference_time_ms": 67.23,
    "model_version": "ResNet50-satellite-v1.0",
    "accuracy": "96.67%"
  }
}
```

## 🏗️ Model Architecture

### Network Design

```
Input (3×256×256) 
    ↓
ResNet50 Backbone (ImageNet pre-trained)
    ↓
Global Average Pooling
    ↓
Dropout(0.3) → Linear(2048→512) → ReLU → Dropout(0.2) → Linear(512→21)
    ↓
Output (21 classes)
```

### Key Features

- **Transfer Learning**: ResNet50 backbone pre-trained on ImageNet
- **Custom Classification Head**: Two-layer fully connected network with dropout
- **Data Augmentation**: Random horizontal flips and rotations during training
- **Normalization**: ImageNet statistics for optimal transfer learning

### Training Configuration

```python
# Optimizer
optimizer = Adam(lr=0.0001)

# Learning Rate Scheduler
scheduler = StepLR(step_size=7, gamma=0.1)

# Loss Function
criterion = CrossEntropyLoss()

# Data Augmentation
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 🎓 Training

### Dataset

The model is trained on the **UC Merced Land Use Dataset**:
- **Total Images**: 2,100 (100 per class)
- **Image Size**: 256×256 pixels
- **Classes**: 21 terrain types
- **Split**: 80% training, 20% validation (stratified)

### Training Process

```bash
# Training notebook
jupyter notebook notebooks/02_model_training.ipynb

# Or run training script
python src/models/trainer.py
```

### Training Features

- **Stratified Split**: Ensures balanced class distribution
- **Early Stopping**: Saves best model based on validation accuracy
- **Progress Tracking**: Real-time loss and accuracy monitoring
- **Reproducible**: Fixed random seeds for consistent results

## 🚢 Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY notebooks/best_satellite_terrain_model.pth ./notebooks/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "src/api/app.py"]
```

### Production Considerations

- **Health Checks**: Built-in endpoint for load balancer monitoring
- **Error Handling**: Comprehensive exception handling and logging
- **Input Validation**: File type and size validation
- **Performance**: Optimized inference with torch.no_grad()
- **Scalability**: Stateless design for horizontal scaling

### Environment Variables

```bash
# Optional configuration
export MODEL_PATH="notebooks/best_satellite_terrain_model.pth"
export BATCH_SIZE=32
export NUM_WORKERS=4
export PORT=8000
```

## 📁 Project Structure

```
satellite-terrain-classification/
├── src/
│   ├── api/
│   │   └── app.py                 # FastAPI application
│   ├── data/
│   │   └── data_loader.py         # Data loading and preprocessing
│   └── models/
│       ├── satellite_cnn.py      # Model architecture
│       └── trainer.py             # Training utilities
├── notebooks/
│   ├── 02_model_training.ipynb    # Training notebook
│   ├── 03_model_evaluation.ipynb # Evaluation notebook
│   └── best_satellite_terrain_model.pth  # Pre-trained model
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
├── .gitignore                     # Git ignore rules
├── README.md                      # Project documentation
└── environment.yml                # Conda environment
```

## 🔬 Model Evaluation

The model performance is evaluated using comprehensive metrics:

```python
# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

# Generate detailed classification report
classification_report(y_true, y_pred, target_names=class_names)

# Confusion matrix visualization
confusion_matrix(y_true, y_pred)
```

### Key Results

- **Overall Accuracy**: 96.67%
- **Macro Average F1-Score**: 0.9667
- **Weighted Average F1-Score**: 0.9667
- **Per-class Performance**: Detailed in evaluation notebook

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone for development
git clone https://github.com/yourusername/satellite-terrain-classification.git
cd satellite-terrain-classification

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If available

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UC Merced** for providing the Land Use dataset
- **PyTorch Team** for the excellent deep learning framework
- **FastAPI** for the modern web framework
- **ResNet Authors** for the foundational architecture

## 📧 Contact

- **Author**: [Nelson Campos]
- **Email**: [ncampos@wm.edu]
- **Project Link**: [https://github.com/nelscamp/satellite-terrain-classifier](https://github.com/nelscamp/satellite-terrain-classifier)

---
