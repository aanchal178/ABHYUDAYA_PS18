# 🚀 HOW TO RUN - Skin Cancer Detection System

**Quick start guide for running the application**

Team: Dr. Homi Jehangir Bhabha | Problem Statement: PS 18

---

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

---

## Step 1: Navigate to Project Directory

```bash
cd skin-cancer-detection
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- Flask (web framework)
- TensorFlow (deep learning)
- Pillow, OpenCV (image processing)
- And other dependencies

**Installation may take 5-10 minutes** depending on your internet speed.

---

## Step 3: Choose Your Running Mode

### Option A: Run with Pre-trained Model (Recommended)

If you have a trained model file (.h5 or .ipynb):

1. **Place your trained model** in the models folder:
   ```bash
   # Copy your .h5 model file to:
   models/best_model.h5
   ```

2. **Start the web application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

4. **Upload a skin lesion image** and get instant predictions!

---

### Option B: Train Your Own Model

If you have a dataset:

1. **Organize your dataset**:
   ```
   data/raw/train/benign/       # Add benign images here
   data/raw/train/malignant/    # Add malignant images here
   data/raw/val/benign/         # Add validation images
   data/raw/val/malignant/      # Add validation images
   ```

2. **Train the model**:
   ```bash
   python main.py train --model efficientnetb0 --epochs 50
   ```

3. **Wait for training to complete** (this may take hours depending on your hardware)

4. **Start the web application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

---

### Option C: Demo Mode (Testing)

To test the application without a real dataset:

1. **Generate demo data**:
   ```bash
   python generate_demo_data.py
   ```
   
   When prompted, type `y` and press Enter

2. **Quick train with demo data** (just for testing):
   ```bash
   python main.py train --epochs 5
   ```

3. **Start the web application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to:
   ```
   http://localhost:5000
   ```

---

## 🎯 Using the Web Application

Once the application is running:

1. **Home Page**: You'll see an upload interface
2. **Upload Image**: Click or drag-and-drop a skin lesion image
3. **Analyze**: Click the "Analyze Image" button
4. **View Results**: See the prediction (benign/malignant) with confidence scores

---

## 📝 Command Reference

### Common Commands

```bash
# Verify installation
python test_setup.py

# Train model
python main.py train

# Train with specific model
python main.py train --model resnet50

# Evaluate model
python main.py evaluate

# Predict single image
python main.py predict path/to/image.jpg

# Start web application
python app.py
```

---

## ⚠️ Troubleshooting

### Issue: "Model not found"
**Solution**: 
- Make sure you have a trained model at `models/best_model.h5`
- Or train a new model: `python main.py train`
- Or generate demo data and train: `python generate_demo_data.py` then `python main.py train`

### Issue: "No module named 'flask'" or similar
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "Port already in use"
**Solution**: 
- Close other applications using port 5000
- Or modify `app.py` to use a different port

### Issue: "Out of memory" during training
**Solution**: 
- Reduce batch size in `src/config.py`
- Use a smaller model like MobileNet: `python main.py train --model mobilenet`

---

## 🔍 Verify Everything is Working

1. **Check dependencies**:
   ```bash
   python test_setup.py
   ```

2. **Check if model exists**:
   ```bash
   ls models/best_model.h5
   ```

3. **Test prediction** (if model exists):
   ```bash
   python main.py predict data/raw/test/benign/demo_benign_001.png
   ```

---

## 📚 Need More Help?

- **Quick Setup**: See `QUICKSTART.md`
- **Full Documentation**: See `README.md`
- **Deployment**: See `DEPLOYMENT.md`
- **Examples**: See `USAGE_EXAMPLES.md`

---

## ✅ Quick Checklist

Before running, make sure:
- [ ] Python 3.8+ is installed
- [ ] You're in the `skin-cancer-detection` directory
- [ ] Dependencies are installed (`pip install -r requirements.txt`)
- [ ] You have either:
  - [ ] A trained model in `models/best_model.h5`, OR
  - [ ] A dataset in `data/raw/train/`, OR
  - [ ] Generated demo data with `generate_demo_data.py`

---

## 🎉 That's It!

You're ready to use the skin cancer detection system!

Open `http://localhost:5000` in your browser and start analyzing skin lesion images.

---

**Important**: This is an educational tool. Always consult healthcare professionals for medical diagnosis.

---

**Team**: Dr. Homi Jehangir Bhabha | **PS 18**
