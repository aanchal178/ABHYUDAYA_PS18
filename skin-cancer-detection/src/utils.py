"""
Utility functions for the skin cancer detection project
"""

import os
import logging
from datetime import datetime


def setup_logging(log_file):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to the log file
    
    Returns:
        Logger object
    """
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_file_extension(filename):
    """
    Get file extension
    
    Args:
        filename: Name of the file
    
    Returns:
        File extension (lowercase)
    """
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''


def allowed_file(filename, allowed_extensions):
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions
    
    Returns:
        Boolean indicating if file is allowed
    """
    return '.' in filename and get_file_extension(filename) in allowed_extensions


def format_bytes(bytes):
    """
    Format bytes to human readable format
    
    Args:
        bytes: Number of bytes
    
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def get_timestamp():
    """
    Get current timestamp string
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
