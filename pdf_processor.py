"""
PDF processor module for extracting and cleaning text from PDF documents.
"""

import logging
import re
import os
import fitz  # PyMuPDF
from typing import List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        logger.info(f"Extracting text from PDF: {file_path}")
        
        # Open the PDF file
        doc = fitz.open(file_path)
        
        # Extract text from each page
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            full_text += page_text + "\n\n"  # Add extra newlines between pages
        
        logger.info(f"Successfully extracted {len(full_text)} characters from {len(doc)} pages")
        
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text: {str(e)}"

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