"""
Dataset loading and preprocessing utilities
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from src.config import *


def create_data_generators(train_dir, val_dir, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE):
    """
    Create data generators for training and validation
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for training
        image_size: Target image size (height, width)
    
    Returns:
        train_generator, validation_generator
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        zoom_range=ZOOM_RANGE,
        fill_mode='nearest'
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator


def load_and_preprocess_image(image_path, target_size=IMAGE_SIZE):
    """
    Load and preprocess a single image for prediction
    
    Args:
        image_path: Path to the image file
        target_size: Target size for the image
    
    Returns:
        Preprocessed image array
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array


def get_class_weights(train_dir):
    """
    Calculate class weights for imbalanced datasets
    
    Args:
        train_dir: Path to training data directory
    
    Returns:
        Dictionary of class weights
    """
    from sklearn.utils import class_weight
    
    classes = os.listdir(train_dir)
    class_counts = {}
    
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            class_counts[cls] = len(os.listdir(cls_path))
    
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for i, cls in enumerate(sorted(classes)):
        class_weights[i] = total_samples / (len(classes) * class_counts[cls])
    
    return class_weights
