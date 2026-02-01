"""
Flask Web Application for Skin Cancer Detection
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.predict import predict_image
from src.utils import allowed_file, ensure_dir, format_bytes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.secret_key = 'your-secret-key-here'  # Change this in production

# Ensure upload directory exists
ensure_dir(UPLOAD_FOLDER)


@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = get_timestamp()
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if model exists
        if not os.path.exists(BEST_MODEL_PATH):
            return jsonify({
                'error': 'Model not found. Please train the model first.',
                'note': 'You can add a pre-trained model file to models/best_model.h5'
            }), 500
        
        # Make prediction
        result = predict_image(filepath, BEST_MODEL_PATH, CLASS_NAMES)
        
        # Add image URL to result
        result['image_url'] = url_for('static', filename=f'uploads/{filename}')
        result['filename'] = filename
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    model_exists = os.path.exists(BEST_MODEL_PATH)
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_exists,
        'model_path': BEST_MODEL_PATH
    }), 200


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Maximum size: {format_bytes(MAX_FILE_SIZE)}'}), 413


if __name__ == '__main__':
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  Skin Cancer Detection System                           ║
    ║  Team: Dr. Homi Jehangir Bhabha                         ║
    ║  Problem Statement: PS 18                                ║
    ╚══════════════════════════════════════════════════════════╝
    
    Application starting...
    Model path: {BEST_MODEL_PATH}
    Model exists: {os.path.exists(BEST_MODEL_PATH)}
    Upload folder: {UPLOAD_FOLDER}
    
    Navigate to: http://localhost:5000
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
