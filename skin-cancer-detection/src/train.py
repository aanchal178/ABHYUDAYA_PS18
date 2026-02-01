"""
Training script for skin cancer detection model
"""

import os
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from src.config import *
from src.model import create_model, fine_tune_model
from src.dataset import create_data_generators, get_class_weights
from src.utils import setup_logging


def train_model(train_dir=TRAIN_DIR, val_dir=VAL_DIR, 
                model_name=MODEL_NAME, epochs=EPOCHS,
                fine_tune=False):
    """
    Train the skin cancer detection model
    
    Args:
        train_dir: Path to training data
        val_dir: Path to validation data
        model_name: Name of the model architecture to use
        epochs: Number of training epochs
        fine_tune: Whether to fine-tune the model after initial training
    
    Returns:
        Trained model and training history
    """
    # Setup logging
    logger = setup_logging(TRAINING_LOG)
    logger.info("Starting training process...")
    logger.info(f"Model: {model_name}, Epochs: {epochs}, Image Size: {IMAGE_SIZE}")
    
    # Create data generators
    logger.info("Creating data generators...")
    train_generator, val_generator = create_data_generators(
        train_dir, val_dir, BATCH_SIZE, IMAGE_SIZE
    )
    
    # Get class weights for imbalanced data
    class_weights = get_class_weights(train_dir)
    logger.info(f"Class weights: {class_weights}")
    
    # Create model
    logger.info(f"Creating {model_name} model...")
    model = create_model(model_name, NUM_CLASSES, IMAGE_SIZE + (3,))
    model.summary(print_fn=lambda x: logger.info(x))
    
    # Define callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, f'model_{timestamp}_{{epoch:02d}}_{{val_accuracy:.4f}}.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(TENSORBOARD_DIR, timestamp),
        histogram_freq=1
    )
    
    callbacks = [checkpoint_callback, early_stopping, reduce_lr, tensorboard_callback]
    
    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save(BEST_MODEL_PATH)
    logger.info(f"Model saved to {BEST_MODEL_PATH}")
    
    # Fine-tuning (optional)
    if fine_tune:
        logger.info("Starting fine-tuning...")
        model = fine_tune_model(model)
        
        # Train for additional epochs with fine-tuning
        fine_tune_epochs = epochs // 2
        history_fine = model.fit(
            train_generator,
            epochs=fine_tune_epochs,
            validation_data=val_generator,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save fine-tuned model
        model.save(BEST_MODEL_PATH.replace('.h5', '_finetuned.h5'))
        logger.info("Fine-tuning complete")
    
    logger.info("Training complete!")
    return model, history


if __name__ == "__main__":
    # Train the model
    model, history = train_model(fine_tune=False)
