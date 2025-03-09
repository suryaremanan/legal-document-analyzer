#!/usr/bin/env python3
"""
Batch PDF Processing Script

This script processes multiple PDF files in a directory, extracting text,
optionally generating PDFs from the extracted text, and optionally analyzing
the content with Together AI.
"""

import os
import sys
import logging
import argparse
import shutil
import concurrent.futures
from pathlib import Path
from datetime import datetime

# Import functions from parse_pdf.py
try:
    from parse_pdf import (
        extract_text_from_pdf, extract_text_with_ocr, 
        generate_pdf_from_text, analyze_text_with_together, 
        HAS_OCR, HAS_PDF_GENERATION
    )
except ImportError:
    print("Error: Could not import functions from parse_pdf.py")
    print("Make sure parse_pdf.py is in the same directory as this script")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
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

def process_pdf_file(pdf_path, output_dir, options):
    """
    Process a single PDF file, extracting text and optionally generating PDFs and analysis.
    
    Args:
        pdf_path (str): Path to the PDF file.
        output_dir (str): Directory to save the output files.
        options (dict): Processing options.
        
    Returns:
        tuple: (success, message) indicating success status and result message
    """
    try:
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]
        
        # Output file paths
        txt_output_path = os.path.join(output_dir, f"{base_name}.txt")
        pdf_output_path = os.path.join(output_dir, f"{base_name}_extracted.pdf") if options.get('create_pdf') else None
        analysis_output_path = os.path.join(output_dir, f"{base_name}_analysis.txt") if options.get('analyze') else None
        analysis_pdf_path = os.path.join(output_dir, f"{base_name}_analysis.pdf") if options.get('analyze') and options.get('create_pdf') else None
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text from PDF
        if options.get('ocr_only') and HAS_OCR:
            logger.info(f"Using OCR for {pdf_filename}")
            extracted_text = extract_text_with_ocr(pdf_path, dpi=options.get('dpi', 300))
        else:
            extracted_text = extract_text_from_pdf(pdf_path)
        
        # Check if text was extracted
        if not extracted_text.strip():
            return False, f"No text was extracted from {pdf_filename}"
        
        # Save extracted text
        with open(txt_output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        logger.info(f"Text extracted and saved to {txt_output_path}")
        
        # Generate PDF from extracted text if requested
        if pdf_output_path and HAS_PDF_GENERATION:
            pdf_title = f"Extracted text from {pdf_filename}"
            success = generate_pdf_from_text(extracted_text, pdf_output_path, pdf_title)
            if success:
                logger.info(f"Generated PDF of extracted text: {pdf_output_path}")
            else:
                logger.warning(f"Failed to generate PDF for {pdf_filename}")
        
        # Analyze text if requested
        if analysis_output_path:
            logger.info(f"Analyzing text from {pdf_filename}")
            analysis_result = analyze_text_with_together(extracted_text, options.get('format', 'analysis'))
            
            # Save analysis text
            with open(analysis_output_path, "w", encoding="utf-8") as f:
                f.write(analysis_result["analysis"])
            logger.info(f"Analysis saved to {analysis_output_path}")
            
            # Generate PDF of analysis if requested
            if analysis_pdf_path and HAS_PDF_GENERATION:
                pdf_title = f"Analysis of {pdf_filename}"
                success = generate_pdf_from_text(analysis_result["analysis"], analysis_pdf_path, pdf_title)
                if success:
                    logger.info(f"Generated PDF of analysis: {analysis_pdf_path}")
                else:
                    logger.warning(f"Failed to generate PDF of analysis for {pdf_filename}")
        
        return True, f"Successfully processed {pdf_filename}"
        
    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def batch_process_pdfs(input_dir, output_dir, options):
    """
    Process all PDF files in the input directory and save results to the output directory.
    
    Args:
        input_dir (str): Directory containing PDF files to process.
        output_dir (str): Directory to save the output files.
        options (dict): Processing options.
        
    Returns:
        tuple: (success_count, failure_count) indicating numbers of successful and failed processes
    """
    # Ensure output directory exists
    ensure_directory(output_dir)
    
    # Find all PDF files in input directory
    pdf_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.pdf'):
            pdf_files.append(os.path.join(input_dir, file))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return 0, 0
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process files with a thread pool to parallelize
    success_count = 0
    failure_count = 0
    
    if options.get('parallel', True):
        max_workers = min(options.get('max_workers', 4), len(pdf_files))
        logger.info(f"Processing files in parallel with {max_workers} workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_pdf_file, pdf_path, output_dir, options): pdf_path for pdf_path in pdf_files}
            
            for future in concurrent.futures.as_completed(futures):
                pdf_path = futures[future]
                success, message = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                logger.info(message)
    else:
        logger.info("Processing files sequentially")
        for pdf_path in pdf_files:
            success, message = process_pdf_file(pdf_path, output_dir, options)
            if success:
                success_count += 1
            else:
                failure_count += 1
            logger.info(message)
    
    logger.info(f"Batch processing complete. Success: {success_count}, Failures: {failure_count}")
    return success_count, failure_count

def main():
    parser = argparse.ArgumentParser(description="Batch process PDF files")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing PDF files to process")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save processed files")
    parser.add_argument("--create-pdf", action="store_true",
                        help="Create PDF versions of extracted text")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze text content with Together AI")
    parser.add_argument("--format", type=str, default="analysis",
                        help="Type of analysis to perform (analysis, summary, etc.)")
    parser.add_argument("--ocr-only", action="store_true",
                        help="Only use OCR for text extraction")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for OCR processing")
    parser.add_argument("--sequential", action="store_true",
                        help="Process files sequentially instead of in parallel")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Maximum number of worker threads for parallel processing")
    
    args = parser.parse_args()
    
    # Prepare options dictionary
    options = {
        'create_pdf': args.create_pdf,
        'analyze': args.analyze,
        'format': args.format,
        'ocr_only': args.ocr_only,
        'dpi': args.dpi,
        'parallel': not args.sequential,
        'max_workers': args.max_workers
    }
    
    # Run batch processing
    success_count, failure_count = batch_process_pdfs(args.input_dir, args.output_dir, options)
    
    # Print summary
    total = success_count + failure_count
    print(f"\nProcessing Summary:")
    print(f"  Total files: {total}")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {failure_count}")
    
    if failure_count > 0:
        print("\nSome files failed to process. Check the log file for details.")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 