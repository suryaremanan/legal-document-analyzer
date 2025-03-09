import os
import tempfile
import streamlit as st
from typing import Any, Optional
import uuid

def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to a temporary location.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Path to the saved file
    """
    # Create a temporary directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), 'streamlit_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")
    
    # Save the file
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_file_path

def get_temp_file_path(base_name: str) -> str:
    """
    Get a temporary file path with the given base name.
    
    Args:
        base_name: Base name for the file
        
    Returns:
        Full path to the temporary file
    """
    temp_dir = os.path.join(tempfile.gettempdir(), 'streamlit_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    return os.path.join(temp_dir, base_name)

def format_bytes(size_bytes: int) -> str:
    """
    Format bytes to human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "4.2 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB" 