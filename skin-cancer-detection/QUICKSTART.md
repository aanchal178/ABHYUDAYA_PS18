# Quick Start Guide

## Getting Started with Skin Cancer Detection System

### Step 1: Install Dependencies

```bash
cd skin-cancer-detection
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Place your dataset in the following structure:

```
data/raw/
├── train/
│   ├── benign/       # Benign skin lesion images
│   └── malignant/    # Malignant skin lesion images
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

### Step 3: Training Options

#### Option A: Use an existing trained model
- If you have a trained model (.h5 file), place it in `models/best_model.h5`
- Skip to Step 4

#### Option B: Train a new model
```bash
# Train with default settings (EfficientNetB0, 50 epochs)
python main.py train

# Or customize the training
python main.py train --model resnet50 --epochs 30 --fine-tune
```

Available models:
- `efficientnetb0` (recommended)
- `resnet50`
- `vgg16`
- `mobilenet`

### Step 4: Start the Web Application

```bash
python app.py
```

Or:

```bash
python main.py webapp
```

Then open your browser and navigate to:
```
http://localhost:5000
```

### Step 5: Use the Web Interface

1. Upload a skin lesion image (JPG, JPEG, or PNG)
2. Click "Analyze Image"
3. View the prediction results with confidence scores

### Additional Commands

#### Evaluate the Model
```bash
python main.py evaluate
```

#### Make Predictions from Command Line
```bash
python main.py predict path/to/image.jpg
```

#### View Training Logs
```bash
tensorboard --logdir=logs/tensorboard
```

### Troubleshooting

**Issue**: Model not found error
- **Solution**: Make sure you have a trained model at `models/best_model.h5` or train a new one

**Issue**: No training data found
- **Solution**: Add images to `data/raw/train/benign/` and `data/raw/train/malignant/`

**Issue**: TensorFlow installation issues
- **Solution**: Make sure you have Python 3.8+ and run `pip install --upgrade tensorflow`

### Project Structure

```
skin-cancer-detection/
├── app.py                 # Flask web application
├── main.py                # CLI entry point
├── requirements.txt       # Dependencies
├── README.md             # Full documentation
├── QUICKSTART.md         # This file
├── data/                 # Dataset directory
├── models/               # Trained models
├── logs/                 # Training logs
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── config.py        # Configuration
│   ├── dataset.py       # Data processing
│   ├── model.py         # Model architectures
│   ├── train.py         # Training logic
│   ├── evaluate.py      # Evaluation metrics
│   ├── predict.py       # Prediction utilities
│   └── utils.py         # Helper functions
├── static/              # Web assets (CSS, JS)
└── templates/           # HTML templates
```

### Team Information

**Team Name**: Dr. Homi Jehangir Bhabha  
**Problem Statement**: PS 18  
**Project**: Deep Learning Image Classification for Skin Cancer Detection

### Need Help?

- Check the full `README.md` for detailed documentation
- Review the training logs in `logs/training.log`
- Explore the Jupyter notebooks in `notebooks/`

---

**Important**: This is an educational tool. Always consult healthcare professionals for medical diagnosis.
