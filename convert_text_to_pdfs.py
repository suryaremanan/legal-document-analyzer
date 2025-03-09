#!/usr/bin/env python3
"""
Text to PDF Batch Converter

This script converts all text files in the 'output' folder to PDF format
and saves them in a new 'pdf_output' folder.
"""

import os
import sys
import logging
from pathlib import Path

# Import functions from parse_pdf.py
try:
    from parse_pdf import (
        generate_pdf_from_text, 
        escape_html_chars,
        clean_document_text
    )
except ImportError:
    print("Error: Could not import functions from parse_pdf.py")
    print("Make sure parse_pdf.py is in the same directory as this script")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory(directory_path):
    """
    Ensure that the specified directory exists.
    
    Args:
        directory_path (str): Path to the directory to create if it doesn't exist.
    """
    path = Path(directory_path)
    if not path.exists():
        path.mkdir(parents=True)
        logger.info(f"Created directory: {directory_path}")

def convert_text_file_to_pdf(text_file_path, pdf_output_path, title=None):
    """
    Convert a text file to PDF format.
    
    Args:
        text_file_path (str): Path to the text file to convert.
        pdf_output_path (str): Path where the PDF file will be saved.
        title (str, optional): Title for the PDF document.
        
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    try:
        # Check if text file exists
        if not os.path.isfile(text_file_path):
            logger.error(f"Text file not found: {text_file_path}")
            return False
            
        # Read the text file
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Clean and prepare text for PDF generation
        cleaned_text = clean_document_text(text)
        escaped_text = escape_html_chars(cleaned_text)
        
        # Generate PDF
        pdf_title = title if title else f"Converted from {os.path.basename(text_file_path)}"
        success = generate_pdf_from_text(escaped_text, pdf_output_path, pdf_title)
        
        if success:
            logger.info(f"Successfully converted {text_file_path} to {pdf_output_path}")
            return True
        else:
            logger.error(f"Failed to generate PDF for {text_file_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error converting text to PDF: {str(e)}")
        return False

def batch_convert_text_to_pdf(input_dir="output", output_dir="pdf_output"):
    """
    Convert all text files in the input directory to PDF format and save them in the output directory.
    
    Args:
        input_dir (str): Directory containing text files to convert.
        output_dir (str): Directory to save generated PDF files.
        
    Returns:
        tuple: (success_count, failure_count) indicating the number of successful and failed conversions.
    """
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return 0, 0
    
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Find all text files in input directory
    text_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.txt'):
            text_files.append(os.path.join(input_dir, file))
    
    if not text_files:
        logger.warning(f"No text files found in {input_dir}")
        return 0, 0
    
    logger.info(f"Found {len(text_files)} text files to convert")
    
    # Convert each text file to PDF
    success_count = 0
    failure_count = 0
    
    for text_file_path in text_files:
        # Determine output PDF path
        base_name = os.path.splitext(os.path.basename(text_file_path))[0]
        pdf_output_path = os.path.join(output_dir, f"{base_name}.pdf")
        
        # Convert text file to PDF
        if convert_text_file_to_pdf(text_file_path, pdf_output_path):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info(f"Batch conversion complete. Success: {success_count}, Failures: {failure_count}")
    return success_count, failure_count

def main():
    # Default directories
    input_dir = "output"
    output_dir = "pdf_output"
    
    # Parse command line arguments for custom directories
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Run batch conversion
    success_count, failure_count = batch_convert_text_to_pdf(input_dir, output_dir)
    
    # Print summary
    total = success_count + failure_count
    print(f"\nConversion Summary:")
    print(f"  Total text files: {total}")
    print(f"  Successfully converted: {success_count}")
    print(f"  Failed: {failure_count}")
    
    if failure_count > 0:
        print("\nSome files failed to convert. Check the log for details.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 