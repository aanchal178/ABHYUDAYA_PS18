"""
Main entry point for the Skin Cancer Detection project
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.utils import setup_logging


def main():
    """Main function to provide CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Skin Cancer Detection - Deep Learning Image Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train
  
  # Evaluate the model
  python main.py evaluate
  
  # Make a prediction
  python main.py predict path/to/image.jpg
  
  # Start web application
  python main.py webapp
  
Team: Dr. Homi Jehangir Bhabha
Problem Statement: PS 18
        """
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'predict', 'webapp'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Additional arguments for the command'
    )
    
    parser.add_argument(
        '--model',
        default=MODEL_NAME,
        help='Model architecture to use (default: efficientnetb0)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--fine-tune',
        action='store_true',
        help='Fine-tune the model after training'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(TRAINING_LOG)
    
    if args.command == 'train':
        logger.info("Starting training...")
        from src.train import train_model
        model, history = train_model(
            model_name=args.model,
            epochs=args.epochs,
            fine_tune=args.fine_tune
        )
        logger.info("Training completed!")
        
    elif args.command == 'evaluate':
        logger.info("Starting evaluation...")
        from src.evaluate import evaluate_model, plot_confusion_matrix
        results = evaluate_model()
        plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES)
        logger.info("Evaluation completed!")
        
    elif args.command == 'predict':
        if not args.args:
            logger.error("Please provide image path")
            sys.exit(1)
        
        image_path = args.args[0]
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)
        
        from src.predict import predict_image
        result = predict_image(image_path)
        
        print(f"\n{'='*50}")
        print(f"Prediction Results")
        print(f"{'='*50}")
        print(f"Image: {image_path}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nClass Probabilities:")
        for class_name, prob in result['class_probabilities'].items():
            print(f"  {class_name}: {prob:.2%}")
        print(f"{'='*50}\n")
        
    elif args.command == 'webapp':
        logger.info("Starting web application...")
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
