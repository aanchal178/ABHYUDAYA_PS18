"""
Configuration file for skin cancer detection project
"""

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TRAIN_DIR = os.path.join(RAW_DATA_DIR, 'train')
VAL_DIR = os.path.join(RAW_DATA_DIR, 'val')
TEST_DIR = os.path.join(RAW_DATA_DIR, 'test')

# Model directories
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.h5')

# Log directories
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRAINING_LOG = os.path.join(LOG_DIR, 'training.log')
TENSORBOARD_DIR = os.path.join(LOG_DIR, 'tensorboard')

# Model hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Model architecture
MODEL_NAME = 'efficientnetb0'  # Options: 'vgg16', 'resnet50', 'efficientnetb0', 'mobilenet'
NUM_CLASSES = 2  # benign and malignant

# Data augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
HORIZONTAL_FLIP = True
ZOOM_RANGE = 0.2

# Training parameters
VALIDATION_SPLIT = 0.2
CLASS_NAMES = ['benign', 'malignant']

# Web application settings
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
