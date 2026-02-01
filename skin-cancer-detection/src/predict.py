"""
Prediction script for single images
"""

import os
import numpy as np
from PIL import Image

from src.config import *
from src.model import load_trained_model
from src.dataset import load_and_preprocess_image
from src.utils import setup_logging


def predict_image(image_path, model_path=BEST_MODEL_PATH, class_names=CLASS_NAMES):
    """
    Predict the class of a single image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the trained model
        class_names: List of class names
    
    Returns:
        Dictionary with prediction results
    """
    # Load model
    model = load_trained_model(model_path)
    
    # Preprocess image
    img_array = load_and_preprocess_image(image_path, IMAGE_SIZE)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    # Get all class probabilities
    class_probabilities = {
        class_names[i]: float(predictions[0][i]) 
        for i in range(len(class_names))
    }
    
    result = {
        'predicted_class': predicted_class,
        'confidence': float(confidence),
        'class_probabilities': class_probabilities,
        'all_predictions': predictions[0].tolist()
    }
    
    return result


def predict_batch(image_paths, model_path=BEST_MODEL_PATH, class_names=CLASS_NAMES):
    """
    Predict classes for multiple images
    
    Args:
        image_paths: List of image file paths
        model_path: Path to the trained model
        class_names: List of class names
    
    Returns:
        List of prediction results
    """
    # Load model once
    model = load_trained_model(model_path)
    
    results = []
    for image_path in image_paths:
        try:
            # Preprocess image
            img_array = load_and_preprocess_image(image_path, IMAGE_SIZE)
            
            # Make prediction
            predictions = model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = class_names[predicted_class_idx]
            
            result = {
                'image_path': image_path,
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'success': True
            }
        except Exception as e:
            result = {
                'image_path': image_path,
                'error': str(e),
                'success': False
            }
        
        results.append(result)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    result = predict_image(image_path)
    
    print(f"\nPrediction Results:")
    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        print(f"  {class_name}: {prob:.2%}")
