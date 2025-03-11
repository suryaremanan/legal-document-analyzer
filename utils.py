"""
Utility functions for the Legal Document Analyzer.
"""

import os
import tempfile
import shutil
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to the saved file
    """
    # Create temp directory if it doesn't exist
    temp_dir = tempfile.mkdtemp()
    
    # Construct a path for the uploaded file
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Write the file to the temp location
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    logging.info(f"Saved uploaded file {uploaded_file.name} to {file_path}")
    
    return file_path

def get_temp_file_path(file_name: str) -> str:
    """
    Get a temporary file path for a given file name.
    
    Args:
        file_name: Name of the file
        
    Returns:
        Temporary file path
    """
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_name)
    return path

def format_bytes(size: int) -> str:
    """
    Format bytes to a human-readable format.
    
    Args:
        size: Size in bytes
        
    Returns:
        Formatted size string
    """
    power = 2**10  # 1024
    n = 0
    labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    
    while size > power:
        size /= power
        n += 1
    
    return f"{size:.2f} {labels[n]}"

def generate_document_id() -> str:
    """
    Generate a unique document ID.
    
    Returns:
        Unique document ID
    """
    import uuid
    return str(uuid.uuid4())

def clean_filename(filename: str) -> str:
    """
    Clean a filename to ensure it's valid.
    
    Args:
        filename: Original filename
        
    Returns:
        Cleaned filename
    """
    # Remove invalid characters
    import re
    cleaned = re.sub(r'[\\/*?:"<>|]', "", filename)
    
    # Truncate if too long
    if len(cleaned) > 100:
        cleaned = cleaned[:100]
    
    return cleaned 