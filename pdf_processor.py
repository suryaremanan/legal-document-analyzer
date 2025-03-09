import logging
import re
import fitz  # PyMuPDF
import PyPDF2
from typing import Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file using multiple libraries for reliability.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    extracted_text = ""
    
    # Try PyMuPDF first (usually better quality)
    try:
        logger.info(f"Extracting text from {pdf_path} using PyMuPDF")
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text()
        doc.close()
        
        if extracted_text.strip():
            logger.info(f"Successfully extracted {len(extracted_text)} characters with PyMuPDF")
            return extracted_text
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed: {str(e)}")
    
    # Fallback to PyPDF2
    try:
        logger.info(f"Extracting text from {pdf_path} using PyPDF2")
        extracted_text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                extracted_text += page.extract_text()
        
        if extracted_text.strip():
            logger.info(f"Successfully extracted {len(extracted_text)} characters with PyPDF2")
            return extracted_text
    except Exception as e:
        logger.warning(f"PyPDF2 extraction failed: {str(e)}")
    
    if not extracted_text.strip():
        logger.error(f"Failed to extract text from {pdf_path}")
        return "Text extraction failed. The PDF might be scanned or have security restrictions."
    
    return extracted_text

def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive whitespace
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove form feed characters
    text = re.sub(r'\f', '', text)
    
    # Strip lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove document artifacts often found in PDFs
    text = re.sub(r'Formatted: .*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Commented \[.*?\].*?$', '', text, flags=re.MULTILINE)
    
    return text

def split_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    
    if len(text) <= chunk_size:
        chunks.append(text)
    else:
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            
            # If this is not the last chunk, try to find a natural breakpoint
            if end < len(text):
                # Look for a paragraph break or a period within the last 100 chars of the chunk
                search_zone = text[end-100:end+100]
                
                # Try to find paragraph break
                paragraph_break = search_zone.find('\n\n')
                if paragraph_break != -1:
                    end = end - 100 + paragraph_break + 2  # +2 for the newline chars
                else:
                    # Try to find sentence break (period followed by space)
                    sentence_break = search_zone.find('. ')
                    if sentence_break != -1:
                        end = end - 100 + sentence_break + 2  # +2 for period and space
            
            # Add the chunk
            chunks.append(text[start:end])
            
            # Move the start position, considering overlap
            start = end - overlap
    
    return chunks 