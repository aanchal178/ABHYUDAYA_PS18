#!/usr/bin/env python
"""
Test script to verify the skin cancer detection project setup
"""

import os
import sys

def test_directory_structure():
    """Test if all required directories exist"""
    print("Testing directory structure...")
    
    required_dirs = [
        'data/raw/train/benign',
        'data/raw/train/malignant',
        'data/raw/val',
        'data/raw/test',
        'data/processed',
        'notebooks',
        'src',
        'models/checkpoints',
        'logs/tensorboard',
        'static/css',
        'static/js',
        'static/uploads',
        'templates'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_files_exist():
    """Test if all required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        'app.py',
        'main.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        '.gitignore',
        'src/__init__.py',
        'src/config.py',
        'src/dataset.py',
        'src/model.py',
        'src/train.py',
        'src/evaluate.py',
        'src/predict.py',
        'src/utils.py',
        'templates/index.html',
        'templates/about.html',
        'static/css/style.css',
        'static/js/main.js',
        'notebooks/data_exploration.ipynb',
        'notebooks/model_experiments.ipynb'
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def test_imports():
    """Test if Python modules can be imported"""
    print("\nTesting Python imports...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    modules = [
        'config',
        'dataset',
        'model',
        'train',
        'evaluate',
        'predict',
        'utils'
    ]
    
    all_imported = True
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module} - Error: {e}")
            all_imported = False
    
    return all_imported

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\nTesting Flask application...")
    
    try:
        from app import app
        print(f"  ✓ Flask app created successfully")
        print(f"  ✓ Routes: {[rule.rule for rule in app.url_map.iter_rules()]}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Skin Cancer Detection - Setup Verification")
    print("Team: Dr. Homi Jehangir Bhabha")
    print("Problem Statement: PS 18")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Directory Structure", test_directory_structure()))
    results.append(("Required Files", test_files_exist()))
    results.append(("Python Imports", test_imports()))
    results.append(("Flask Application", test_flask_app()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "✓" if passed else "✗"
        print(f"{icon} {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! The project setup is complete.")
        print("\nNext steps:")
        print("1. Add training data to data/raw/train/")
        print("2. Train the model: python main.py train")
        print("3. Start web app: python app.py")
        print("\nFor more details, see QUICKSTART.md")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit(main())
