"""
PDF processor module for extracting and cleaning text from PDF documents.
"""

import logging
import re
import os
import fitz  # PyMuPDF
from typing import List, Optional, Tuple
from utils import save_uploaded_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(uploaded_file):
    """
    Extract text content from a PDF file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Extracted text as string
    """
    try:
        # Save the uploaded file to a temporary location
        temp_path = save_uploaded_file(uploaded_file)
        
        # Extract text using PyMuPDF (fitz)
        logging.info(f"Extracting text from PDF: {uploaded_file}")
        doc = fitz.open(temp_path)  # Use temp_path instead of just the filename
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        # Clean up the text
        text = clean_text(text)
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    try:
        logger.info("Cleaning extracted text")
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR/extraction issues
        cleaned = cleaned.replace('•', '* ')  # Replace bullets with asterisks
        cleaned = re.sub(r'([a-z])(\.)([A-Z])', r'\1\2 \3', cleaned)  # Add space after periods
        
        # Remove page numbers and headers/footers (simplified approach)
        cleaned = re.sub(r'\n\s*\d+\s*\n', '\n', cleaned)  # Remove standalone page numbers
        
        logger.info(f"Text cleaning complete, final length: {len(cleaned)} characters")
        
        return cleaned
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text  # Return original text on error

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={overlap})")
        
        # Split text by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, store the current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep the overlap from the end of the previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks")
        
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {str(e)}")
        # Return a single chunk with the whole text on error
        return [text] 