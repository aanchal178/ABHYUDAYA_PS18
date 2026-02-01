"""
Demo data generator for testing the application without a full dataset

This script generates synthetic images to test the application functionality.
DO NOT use these for actual training - they are just for demo purposes.
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

def create_demo_image(label, index, output_path):
    """
    Create a demo image with some random patterns
    
    Args:
        label: 'benign' or 'malignant'
        index: Image index
        output_path: Where to save the image
    """
    # Create a 224x224 RGB image
    img = Image.new('RGB', (224, 224), color=(random.randint(200, 240), 
                                               random.randint(180, 220), 
                                               random.randint(180, 220)))
    draw = ImageDraw.Draw(img)
    
    # Add some random shapes to simulate skin lesion patterns
    if label == 'benign':
        # Benign: Regular, symmetric patterns
        color = (random.randint(150, 180), random.randint(100, 130), random.randint(80, 110))
        draw.ellipse([60, 60, 164, 164], fill=color, outline=(0, 0, 0))
        # Add some texture
        for _ in range(20):
            x = random.randint(70, 154)
            y = random.randint(70, 154)
            draw.point((x, y), fill=(random.randint(120, 160), 
                                    random.randint(80, 120), 
                                    random.randint(60, 100)))
    else:
        # Malignant: Irregular, asymmetric patterns
        color = (random.randint(100, 140), random.randint(60, 90), random.randint(50, 80))
        # Irregular shape
        points = []
        for i in range(8):
            angle = i * (2 * 3.14159 / 8)
            radius = random.randint(40, 60)
            x = 112 + int(radius * np.cos(angle))
            y = 112 + int(radius * np.sin(angle))
            points.append((x, y))
        draw.polygon(points, fill=color, outline=(0, 0, 0))
        # Add more irregular texture
        for _ in range(40):
            x = random.randint(60, 164)
            y = random.randint(60, 164)
            draw.point((x, y), fill=(random.randint(80, 120), 
                                    random.randint(40, 80), 
                                    random.randint(30, 70)))
    
    # Add label text (for demo purposes)
    try:
        draw.text((10, 10), f"{label} #{index}", fill=(255, 0, 0))
    except:
        pass  # Font issues
    
    # Save image
    img.save(output_path)

def generate_demo_dataset():
    """Generate a small demo dataset"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create demo images for train set
    print("Generating demo training images...")
    for label in ['benign', 'malignant']:
        train_dir = os.path.join(base_dir, 'data', 'raw', 'train', label)
        os.makedirs(train_dir, exist_ok=True)
        for i in range(10):  # 10 images per class
            output_path = os.path.join(train_dir, f'demo_{label}_{i:03d}.png')
            create_demo_image(label, i, output_path)
    print(f"  Created 20 training images")
    
    # Create demo images for validation set
    print("Generating demo validation images...")
    for label in ['benign', 'malignant']:
        val_dir = os.path.join(base_dir, 'data', 'raw', 'val', label)
        os.makedirs(val_dir, exist_ok=True)
        for i in range(5):  # 5 images per class
            output_path = os.path.join(val_dir, f'demo_{label}_{i:03d}.png')
            create_demo_image(label, i, output_path)
    print(f"  Created 10 validation images")
    
    # Create demo images for test set
    print("Generating demo test images...")
    for label in ['benign', 'malignant']:
        test_dir = os.path.join(base_dir, 'data', 'raw', 'test', label)
        os.makedirs(test_dir, exist_ok=True)
        for i in range(5):  # 5 images per class
            output_path = os.path.join(test_dir, f'demo_{label}_{i:03d}.png')
            create_demo_image(label, i, output_path)
    print(f"  Created 10 test images")
    
    print("\nDemo dataset generated successfully!")
    print("Total: 40 images (20 train, 10 val, 10 test)")
    print("\n⚠️  WARNING: These are synthetic demo images!")
    print("For actual training, replace with real skin lesion dataset.")

if __name__ == '__main__':
    print("="*60)
    print("Demo Dataset Generator")
    print("="*60)
    print("\nThis will create synthetic images for testing.")
    print("These are NOT real skin lesion images.")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() == 'y':
        try:
            generate_demo_dataset()
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure you have Pillow installed: pip install Pillow")
    else:
        print("Cancelled.")
