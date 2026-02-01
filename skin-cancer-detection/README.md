# Skin Cancer Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Deep Learning Image Classification for Healthcare**

Team: Dr. Homi Jehangir Bhabha | Problem Statement: PS 18

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Web Application](#web-application)
- [Results](#results)
- [Team](#team)
- [Disclaimer](#disclaimer)

---

## 🎯 Overview

This project implements an AI-powered skin cancer detection system using Convolutional Neural Networks (CNNs) to classify skin lesion images into benign or malignant categories. The system leverages transfer learning from state-of-the-art pre-trained models and provides a user-friendly web interface for easy access.

## 📝 Problem Statement

**Problem Statement ID: PS 18**

Design an image classification system that can accurately identify and classify images into predefined categories using deep learning techniques. Applied to healthcare, this system analyzes skin lesion images to distinguish between different types of skin conditions, including cancerous and non-cancerous lesions.

### Key Challenges:
- Learning visual patterns from skin lesion images
- Handling class imbalance in medical datasets
- Achieving high accuracy for medical diagnosis support
- Providing interpretable results with confidence scores

## 💡 Solution Approach

Our solution implements a comprehensive deep learning pipeline:

1. **Data Preprocessing**: Image normalization, resizing, and augmentation
2. **Transfer Learning**: Leveraging pre-trained models (EfficientNet, ResNet, VGG16, MobileNet)
3. **Model Training**: Supervised learning with labeled skin lesion datasets
4. **Evaluation**: Multi-metric assessment (accuracy, precision, recall, AUC)
5. **Deployment**: Web-based application for practical use

## 📁 Project Structure

```
skin-cancer-detection/
│
├── data/
│   ├── raw/
│   │   ├── train/          # Training images
│   │   │   ├── benign/
│   │   │   └── malignant/
│   │   ├── val/            # Validation images
│   │   └── test/           # Test images
│   └── processed/          # Preprocessed data
│
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuration settings
│   ├── dataset.py          # Data loading and preprocessing
│   ├── model.py            # Model architectures
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation metrics
│   ├── predict.py          # Prediction utilities
│   └── utils.py            # Helper functions
│
├── models/
│   ├── best_model.h5       # Trained model
│   └── checkpoints/        # Training checkpoints
│
├── logs/
│   ├── training.log        # Training logs
│   └── tensorboard/        # TensorBoard logs
│
├── static/
│   ├── css/               # Stylesheets
│   ├── js/                # JavaScript files
│   └── uploads/           # Uploaded images
│
├── templates/
│   ├── index.html         # Main page
│   └── about.html         # About page
│
├── app.py                 # Flask web application
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
```bash
cd skin-cancer-detection
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare your data**
- Place training images in `data/raw/train/benign/` and `data/raw/train/malignant/`
- Place validation images in `data/raw/val/`
- Place test images in `data/raw/test/`

## 📖 Usage

### Command Line Interface

The project provides a CLI through `main.py` for various operations:

#### 1. Train the Model
```bash
python main.py train --model efficientnetb0 --epochs 50 --fine-tune
```

Available models:
- `efficientnetb0` (default)
- `resnet50`
- `vgg16`
- `mobilenet`

#### 2. Evaluate the Model
```bash
python main.py evaluate
```

#### 3. Make Predictions
```bash
python main.py predict path/to/image.jpg
```

#### 4. Start Web Application
```bash
python main.py webapp
```
Or directly:
```bash
python app.py
```

Then open your browser and navigate to `http://localhost:5000`

### Web Application

The web interface provides:
- **Image Upload**: Drag & drop or browse to upload skin lesion images
- **Real-time Analysis**: Instant classification with confidence scores
- **Visual Results**: Clear presentation of predictions with probabilities
- **Responsive Design**: Works on desktop and mobile devices

## 🧠 Model Architecture

### Transfer Learning Approach

The system uses transfer learning with pre-trained models:

1. **Base Model**: Pre-trained on ImageNet (e.g., EfficientNetB0)
2. **Custom Layers**:
   - Global Average Pooling
   - Batch Normalization
   - Dense layer (512 units, ReLU)
   - Dropout (0.5)
   - Dense layer (256 units, ReLU)
   - Dropout (0.3)
   - Output layer (2 units, Softmax)

### Training Configuration

- **Input Size**: 224×224×3
- **Batch Size**: 32
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, AUC, Precision, Recall
- **Data Augmentation**:
  - Rotation (±20°)
  - Width/Height shift (±20%)
  - Horizontal flip
  - Zoom (±20%)

### Callbacks

- **ModelCheckpoint**: Save best model based on validation accuracy
- **EarlyStopping**: Stop training if no improvement for 10 epochs
- **ReduceLROnPlateau**: Reduce learning rate when validation loss plateaus
- **TensorBoard**: Log training metrics for visualization

## 🌐 Web Application

### Features

1. **Home Page**
   - Image upload interface
   - Drag & drop support
   - File validation
   - Real-time preview

2. **Results Display**
   - Predicted class (Benign/Malignant)
   - Confidence percentage
   - Visual confidence meter
   - Class probability breakdown

3. **About Page**
   - Project overview
   - Technical details
   - Team information
   - Medical disclaimer

### API Endpoints

- `GET /` - Home page
- `GET /about` - About page
- `POST /predict` - Image prediction endpoint
- `GET /health` - Health check

## 📊 Results

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: Visual representation of classification results

### Visualization

Training progress can be monitored using:
```bash
tensorboard --logdir=logs/tensorboard
```

## 👥 Team

**Team Name**: Dr. Homi Jehangir Bhabha

Named after the pioneering Indian nuclear physicist who founded the Tata Institute of Fundamental Research, our team brings together expertise in:
- Machine Learning & Deep Learning
- Computer Vision
- Healthcare Applications
- Full-Stack Development

## ⚠️ Disclaimer

**IMPORTANT MEDICAL DISCLAIMER**

This system is designed for **educational and research purposes only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.

- Always seek the advice of qualified healthcare providers
- Never disregard professional medical advice based on results from this application
- Do not delay seeking medical treatment due to results from this system
- This tool is meant to assist, not replace, medical professionals

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

## 📞 Support

For questions or issues:
- Check the documentation
- Review the code comments
- Consult the training logs

---

<div align="center">

**Deep Learning Image Classification for Healthcare**

Problem Statement: PS 18 | Team: Dr. Homi Jehangir Bhabha

Made with ❤️ using TensorFlow and Flask

</div>
