# Usage Examples

Complete guide with practical examples for using the Skin Cancer Detection System.

**Team**: Dr. Homi Jehangir Bhabha | **Problem Statement**: PS 18

---

## Example 1: First Time Setup

```bash
# Navigate to project
cd skin-cancer-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python test_setup.py
```

**Expected Output:**
```
✓ Directory Structure: PASS
✓ Required Files: PASS
✗ Python Imports: FAIL (missing dependencies - expected)
✗ Flask Application: FAIL (missing dependencies - expected)
```

After installing dependencies, all tests should pass.

---

## Example 2: Using Demo Data

```bash
# Generate synthetic demo images
python generate_demo_data.py

# When prompted, type 'y' to continue
# This creates 40 demo images for testing
```

**Output:**
```
Generating demo training images...
  Created 20 training images
Generating demo validation images...
  Created 10 validation images
Generating demo test images...
  Created 10 test images

Demo dataset generated successfully!
Total: 40 images (20 train, 10 val, 10 test)

⚠️  WARNING: These are synthetic demo images!
For actual training, replace with real skin lesion dataset.
```

---

## Example 3: Training a Model

```bash
# Train with default settings (EfficientNetB0, 50 epochs)
python main.py train
```

**Console Output:**
```
╔══════════════════════════════════════════════╗
║  Starting Training Process                   ║
╚══════════════════════════════════════════════╝

Model: efficientnetb0, Epochs: 50, Image Size: (224, 224)
Creating data generators...
Found 20 images belonging to 2 classes.
Found 10 images belonging to 2 classes.
Class weights: {0: 1.0, 1: 1.0}

Creating efficientnetb0 model...
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
efficientnetb0 (Functional)  (None, 7, 7, 1280)       4049571   
global_average_pooling2d     (None, 1280)             0         
batch_normalization          (None, 1280)             5120      
dense                        (None, 512)              655872    
dropout                      (None, 512)              0         
dense_1                      (None, 256)              131328    
dropout_1                    (None, 256)              0         
dense_2                      (None, 2)                514       
=================================================================
Total params: 4,842,405
Trainable params: 789,634
Non-trainable params: 4,052,771

Epoch 1/50
1/1 [==============================] - 5s 5s/step - loss: 0.7234 - accuracy: 0.5500 - val_loss: 0.6891 - val_accuracy: 0.6000

Epoch 2/50
1/1 [==============================] - 1s 1s/step - loss: 0.6892 - accuracy: 0.6000 - val_loss: 0.6745 - val_accuracy: 0.7000
...

Training complete!
Model saved to models/best_model.h5
```

### Training with Custom Settings

```bash
# Use ResNet50, train for 30 epochs
python main.py train --model resnet50 --epochs 30

# With fine-tuning enabled
python main.py train --model efficientnetb0 --epochs 50 --fine-tune
```

---

## Example 4: Model Evaluation

```bash
# Evaluate trained model on test set
python main.py evaluate
```

**Output:**
```
Starting model evaluation...
Model loaded from models/best_model.h5
Found 10 images belonging to 2 classes.

Evaluating on test data...
10/10 [==============================] - 2s 200ms/step

Test Loss: 0.3456
Test Accuracy: 0.9000
Test AUC: 0.9500
Test Precision: 0.8889
Test Recall: 0.9000

Classification Report:
              precision    recall  f1-score   support

      benign       0.89      0.90      0.89         5
   malignant       0.90      0.89      0.90         5

    accuracy                           0.90        10
   macro avg       0.89      0.90      0.89        10
weighted avg       0.89      0.90      0.89        10

Confusion Matrix:
[[4 1]
 [1 4]]
```

---

## Example 5: Making Predictions

### Single Image Prediction

```bash
# Predict a single image
python main.py predict data/raw/test/benign/demo_benign_001.png
```

**Output:**
```
==================================================
Prediction Results
==================================================
Image: data/raw/test/benign/demo_benign_001.png
Predicted Class: benign
Confidence: 92.45%

Class Probabilities:
  benign: 92.45%
  malignant: 7.55%
==================================================
```

### Programmatic Prediction

```python
# Python script example
from src.predict import predict_image

# Make prediction
result = predict_image('path/to/image.jpg')

# Access results
print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All probabilities: {result['class_probabilities']}")
```

---

## Example 6: Starting Web Application

### Development Mode

```bash
# Start Flask development server
python app.py
```

**Console Output:**
```
╔══════════════════════════════════════════════════════════╗
║  Skin Cancer Detection System                           ║
║  Team: Dr. Homi Jehangir Bhabha                         ║
║  Problem Statement: PS 18                                ║
╚══════════════════════════════════════════════════════════╝

Application starting...
Model path: models/best_model.h5
Model exists: True
Upload folder: static/uploads

Navigate to: http://localhost:5000

 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://0.0.0.0:5000
```

### Production Mode

```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn (4 workers)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Example 7: Using the Web Interface

### Step-by-Step Workflow

1. **Open Browser**
   - Navigate to `http://localhost:5000`
   - You'll see the home page with upload interface

2. **Upload Image**
   - Click the upload area or drag & drop
   - Select a skin lesion image (JPG, JPEG, or PNG)
   - Image preview appears

3. **Analyze**
   - Click "Analyze Image" button
   - Wait for processing (1-2 seconds)
   - Results appear below

4. **View Results**
   - See predicted class (BENIGN or MALIGNANT)
   - Check confidence percentage
   - Review probability breakdown
   - Read medical disclaimer

5. **Analyze Another**
   - Click "Analyze Another Image"
   - Returns to upload interface

---

## Example 8: API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/best_model.h5"
}
```

### Prediction API

```bash
# Using curl
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict
```

**Response:**
```json
{
  "predicted_class": "benign",
  "confidence": 0.9245,
  "class_probabilities": {
    "benign": 0.9245,
    "malignant": 0.0755
  },
  "image_url": "/static/uploads/20260201_123456_image.jpg",
  "filename": "20260201_123456_image.jpg"
}
```

### Python Requests Example

```python
import requests

# Upload and predict
url = 'http://localhost:5000/predict'
files = {'file': open('image.jpg', 'rb')}
response = requests.post(url, files=files)

# Parse results
if response.status_code == 200:
    result = response.json()
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print(f"Error: {response.json()['error']}")
```

---

## Example 9: Monitoring Training with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=logs/tensorboard

# Open browser to http://localhost:6006
# View:
# - Training/validation accuracy
# - Training/validation loss
# - Learning rate changes
# - Model graph
```

---

## Example 10: Working with Jupyter Notebooks

### Data Exploration

```bash
# Start Jupyter
jupyter notebook notebooks/data_exploration.ipynb

# The notebook includes:
# - Dataset overview
# - Class distribution visualization
# - Sample image display
# - Image properties analysis
```

### Model Experiments

```bash
# Start Jupyter
jupyter notebook notebooks/model_experiments.ipynb

# The notebook includes:
# - Architecture comparison
# - Model complexity analysis
# - Data augmentation visualization
# - Quick training tests
```

---

## Example 11: Docker Deployment

```bash
# Build Docker image
docker build -t skin-cancer-detection .

# Run container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/static/uploads:/app/static/uploads \
  --name skin-cancer-app \
  skin-cancer-detection

# Check logs
docker logs skin-cancer-app

# Stop container
docker stop skin-cancer-app

# Remove container
docker rm skin-cancer-app
```

---

## Example 12: Batch Processing

```python
# Python script for batch predictions
from src.predict import predict_batch
import glob

# Get all test images
image_paths = glob.glob('data/raw/test/**/*.png', recursive=True)

# Process all images
results = predict_batch(image_paths)

# Print results
for result in results:
    if result['success']:
        print(f"{result['image_path']}: {result['predicted_class']} ({result['confidence']:.2%})")
    else:
        print(f"{result['image_path']}: ERROR - {result['error']}")
```

**Output:**
```
data/raw/test/benign/demo_benign_000.png: benign (91.23%)
data/raw/test/benign/demo_benign_001.png: benign (89.45%)
data/raw/test/malignant/demo_malignant_000.png: malignant (87.67%)
data/raw/test/malignant/demo_malignant_001.png: malignant (92.34%)
...
```

---

## Example 13: Custom Configuration

```python
# Modify src/config.py for custom settings

# Change image size
IMAGE_SIZE = (299, 299)  # For InceptionV3

# Adjust batch size
BATCH_SIZE = 16  # Reduce if memory issues

# Modify learning rate
LEARNING_RATE = 0.0001  # Lower for fine-tuning

# Change augmentation parameters
ROTATION_RANGE = 30  # More aggressive rotation
ZOOM_RANGE = 0.3  # More aggressive zoom
```

---

## Example 14: Error Handling

### Common Errors and Solutions

**Error: Model not found**
```python
# Solution: Check if model exists
import os
from src.config import BEST_MODEL_PATH

if not os.path.exists(BEST_MODEL_PATH):
    print("Model not found. Train a model first:")
    print("  python main.py train")
```

**Error: CUDA out of memory**
```python
# Solution: Reduce batch size in config.py
BATCH_SIZE = 16  # or even 8
```

**Error: No images found**
```bash
# Solution: Check data directory structure
tree data/raw/train -L 2

# Should show:
# data/raw/train
# ├── benign
# │   └── images here
# └── malignant
#     └── images here
```

---

## Summary

This guide provides practical examples for:
- ✅ Setup and installation
- ✅ Data preparation
- ✅ Model training
- ✅ Evaluation
- ✅ Predictions
- ✅ Web application
- ✅ API usage
- ✅ Deployment
- ✅ Monitoring
- ✅ Troubleshooting

For more details, see other documentation files:
- **QUICKSTART.md**: Fast setup
- **README.md**: Complete documentation
- **DEPLOYMENT.md**: Production deployment
- **FEATURES.md**: Feature list

---

**Team**: Dr. Homi Jehangir Bhabha | **PS 18** | Deep Learning Image Classification
