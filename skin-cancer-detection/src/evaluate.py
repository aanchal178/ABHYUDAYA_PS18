"""
Model evaluation script
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf

from src.config import *
from src.model import load_trained_model
from src.dataset import create_data_generators
from src.utils import setup_logging


def evaluate_model(model_path=BEST_MODEL_PATH, test_dir=TEST_DIR):
    """
    Evaluate the trained model on test data
    
    Args:
        model_path: Path to the trained model
        test_dir: Path to test data directory
    
    Returns:
        Evaluation metrics and predictions
    """
    logger = setup_logging(TRAINING_LOG)
    logger.info("Starting model evaluation...")
    
    # Load model
    model = load_trained_model(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Create test data generator
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate
    logger.info("Evaluating on test data...")
    test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(test_generator)
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    
    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Classification report
    class_names = list(test_generator.class_indices.keys())
    logger.info("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'predictions': predictions,
        'y_pred': y_pred,
        'y_true': y_true,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training history
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # AUC
    if 'auc' in history.history:
        axes[1, 0].plot(history.history['auc'], label='Train AUC')
        axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Model AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 1].plot(history.history['precision'], label='Train Precision')
        axes[1, 1].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 1].set_title('Model Precision')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Evaluate the model
    results = evaluate_model()
    
    # Plot confusion matrix
    plot_confusion_matrix(
        results['confusion_matrix'], 
        CLASS_NAMES, 
        'confusion_matrix.png'
    )
