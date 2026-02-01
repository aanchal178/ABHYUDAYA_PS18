"""
Model architecture definitions
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, EfficientNetB0, MobileNetV2
)
from src.config import *


def create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, input_shape=(224, 224, 3)):
    """
    Create a CNN model with transfer learning
    
    Args:
        model_name: Name of the pre-trained model to use
        num_classes: Number of output classes
        input_shape: Input image shape
    
    Returns:
        Compiled Keras model
    """
    # Load pre-trained model
    if model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.lower() == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.lower() == 'efficientnetb0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name.lower() == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model


def fine_tune_model(model, base_model_layers_to_unfreeze=30):
    """
    Fine-tune the model by unfreezing some layers
    
    Args:
        model: The model to fine-tune
        base_model_layers_to_unfreeze: Number of layers to unfreeze from the end
    
    Returns:
        Fine-tuned model
    """
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last few
    for layer in base_model.layers[:-base_model_layers_to_unfreeze]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    
    return model


def load_trained_model(model_path=BEST_MODEL_PATH):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    return tf.keras.models.load_model(model_path)
