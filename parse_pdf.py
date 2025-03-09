import os
import sys
import logging
import argparse
import tempfile
import re
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try multiple PDF extraction libraries
try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not installed. Only some extraction methods will be available.")

try:
    import pdfplumber
except ImportError:
    logger.warning("pdfplumber not installed. Only some extraction methods will be available.")

# OCR dependencies
try:
    from pdf2image import convert_from_path
    import pytesseract
    HAS_OCR = True
except ImportError:
    logger.warning("OCR dependencies (pdf2image, pytesseract) not installed. OCR will not be available.")
    HAS_OCR = False

# PDF creation dependencies
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    HAS_PDF_GENERATION = True
except ImportError:
    logger.warning("PDF generation dependencies (reportlab) not installed. PDF output will not be available.")
    HAS_PDF_GENERATION = False

try:
    from together import Together
except ImportError:
    logger.error("Together API client not installed. Please run: pip install together")
    sys.exit(1)

# Replace with your actual Together AI API key
API_KEY = "65af5427bf690167d5f3e99960fba3191aaa0f34c5c14031374f64d446de1f79"

def extract_text_with_pypdf2(file_path):
    """
    Extracts text from a PDF file using PyPDF2.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            logger.info(f"PDF has {len(pdf_reader.pages)} pages")
            
            for page_num in range(len(pdf_reader.pages)):
                logger.info(f"Extracting text from page {page_num+1}")
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n\n"
                else:
                    logger.warning(f"No text extracted from page {page_num+1}")
                    
            if not text.strip():
                logger.warning("PyPDF2 extraction resulted in empty text")
                
        return text
    except Exception as e:
        logger.error(f"Error extracting text with PyPDF2: {str(e)}")
        return ""

def extract_text_with_pdfplumber(file_path):
    """
    Extracts text from a PDF file using pdfplumber.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages (pdfplumber)")
            
            for i, page in enumerate(pdf.pages):
                logger.info(f"Extracting text from page {i+1} (pdfplumber)")
                page_text = page.extract_text()
                
                if page_text:
                    text += page_text + "\n\n"
                else:
                    logger.warning(f"No text extracted from page {i+1} (pdfplumber)")
                    
            if not text.strip():
                logger.warning("pdfplumber extraction resulted in empty text")
                
        return text
    except Exception as e:
        logger.error(f"Error extracting text with pdfplumber: {str(e)}")
        return ""

def extract_text_with_ocr(file_path, dpi=300):
    """
    Extracts text from a PDF file using OCR (Optical Character Recognition).
    
    Args:
        file_path (str): Path to the PDF file.
        dpi (int): DPI for conversion (higher values may increase accuracy but take longer)
    
    Returns:
        str: Extracted text from the PDF.
    """
    if not HAS_OCR:
        logger.error("OCR dependencies not installed")
        return ""
    
    text = ""
    try:
        logger.info(f"Converting PDF to images at {dpi} DPI")
        # Create a temporary directory for the images
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert the PDF to images
            images = convert_from_path(file_path, dpi=dpi, output_folder=temp_dir)
            logger.info(f"Converted PDF to {len(images)} images")
            
            # Process each image with Tesseract OCR
            for i, image in enumerate(images):
                logger.info(f"Performing OCR on page {i+1}")
                page_text = pytesseract.image_to_string(image)
                if page_text.strip():
                    text += page_text + "\n\n"
                else:
                    logger.warning(f"No text extracted from page {i+1} with OCR")
                    
        if not text.strip():
            logger.warning("OCR extraction resulted in empty text")
            
        return text
    except Exception as e:
        logger.error(f"Error extracting text with OCR: {str(e)}")
        return ""

def extract_text_from_pdf(file_path):
    """
    Tries multiple methods to extract text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    logger.info(f"Attempting to extract text from {file_path}")
    
    # Check if file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    # Try extraction with PyPDF2 first
    if 'PyPDF2' in sys.modules:
        text = extract_text_with_pypdf2(file_path)
        if text.strip():
            logger.info("Successfully extracted text with PyPDF2")
            return text
        logger.warning("PyPDF2 extraction failed or returned empty text")
    
    # If PyPDF2 failed or is not available, try pdfplumber
    if 'pdfplumber' in sys.modules:
        text = extract_text_with_pdfplumber(file_path)
        if text.strip():
            logger.info("Successfully extracted text with pdfplumber")
            return text
        logger.warning("pdfplumber extraction failed or returned empty text")
    
    # If standard extraction methods failed, try OCR
    if HAS_OCR:
        logger.info("Attempting extraction with OCR")
        text = extract_text_with_ocr(file_path)
        if text.strip():
            logger.info("Successfully extracted text with OCR")
            return text
        logger.warning("OCR extraction failed or returned empty text")
    
    # If all methods failed, raise an exception
    available_methods = []
    if 'PyPDF2' in sys.modules:
        available_methods.append("PyPDF2")
    if 'pdfplumber' in sys.modules:
        available_methods.append("pdfplumber")
    if HAS_OCR:
        available_methods.append("OCR")
    
    if not available_methods:
        raise ImportError("No PDF extraction libraries available. Please install PyPDF2, pdfplumber, or OCR dependencies.")
    
    raise Exception(f"All PDF extraction methods ({', '.join(available_methods)}) failed. The PDF might be encrypted, heavily redacted, or corrupted.")

def analyze_text_with_together(text, output_format="analysis"):
    """
    Sends extracted text to Together AI for analysis.
    
    Args:
        text (str): Text extracted from PDF.
        output_format (str): Type of output requested (analysis, summary, etc.)
    
    Returns:
        dict: JSON response from the API.
    """
    if not text.strip():
        raise ValueError("Cannot analyze empty text. PDF extraction failed.")
        
    try:
        # Initialize Together client
        client = Together(api_key=API_KEY)
        
        # Truncate text if it's too long
        truncated_text = text[:4000]
        if len(text) > 4000:
            logger.warning(f"Text truncated from {len(text)} to 4000 characters for API call")
        
        # Create a prompt that instructs the model what to do with the PDF text
        prompt = f"""The following is text extracted from a PDF document. 
        Please provide a detailed {output_format} of the content.
        Note that this text was extracted from a possibly redacted legal document, 
        so there may be missing or garbled text:
        
        {truncated_text}
        """
        
        logger.info(f"Sending text to Together AI for {output_format}")
        
        # Query a model using Together AI
        response = client.chat.completions.create(
            model="meta-llama/Llama-3-8b-chat-hf",
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"Received analysis ({len(analysis)} characters)")
        return {"analysis": analysis}
    
    except Exception as e:
        logger.error(f"Error with Together AI API: {str(e)}")
        raise Exception(f"Error with Together AI API: {str(e)}")

def escape_html_chars(text):
    """
    Escape HTML special characters in text to prevent ReportLab parsing errors.
    
    Args:
        text (str): Text that may contain HTML-like tags or special characters
        
    Returns:
        str: Text with HTML special characters escaped
    """
    # Replace angle brackets with their HTML entities
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    
    # Handle other special characters as needed
    text = text.replace('&', '&amp;')
    
    # Clean up any existing HTML entities to prevent double escaping
    text = text.replace('&amp;lt;', '&lt;')
    text = text.replace('&amp;gt;', '&gt;')
    
    return text

def clean_document_text(text):
    """
    Clean up document formatting marks and notations.
    
    Args:
        text (str): Text with formatting marks and comments
        
    Returns:
        str: Cleaned text suitable for PDF generation
    """
    # Remove formatting marks (commonly found in exported Word documents)
    text = re.sub(r'Formatted: .*?$', '', text, flags=re.MULTILINE)
    
    # Remove comment tags
    text = re.sub(r'Commented \[.*?\].*?$', '', text, flags=re.MULTILINE)
    
    # Remove other document-specific notations
    text = re.sub(r'Tab stops: .*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'Font: .*?$', '', text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

def generate_pdf_from_text(text, output_path, title=None):
    """
    Generates a PDF file from the given text.
    
    Args:
        text (str): Text to include in the PDF.
        output_path (str): Path where the PDF file will be saved.
        title (str, optional): Title for the PDF document.
    
    Returns:
        bool: True if PDF generation was successful, False otherwise.
    """
    if not HAS_PDF_GENERATION:
        logger.error("Cannot generate PDF: reportlab is not installed")
        return False
    
    try:
        logger.info(f"Generating PDF at {output_path}")
        
        # Clean and prepare the text for PDF generation
        cleaned_text = clean_document_text(text)
        escaped_text = escape_html_chars(cleaned_text)
        
        logger.info("Text cleaned and prepared for PDF generation")
        
        # Create a PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='Justify',
            fontName='Helvetica',
            fontSize=10,
            alignment=4,  # Justified
            leading=14
        ))
        
        # Add title if provided
        if title:
            elements.append(Paragraph(title, styles['Title']))
            elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated on: {timestamp}", styles['Italic']))
        elements.append(Spacer(1, 0.25*inch))
        
        # Process text - be cautious with line breaks
        try:
            # First try with paragraph splitting
            paragraphs = escaped_text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    # Replace newlines with HTML line breaks
                    para_with_breaks = para.replace('\n', '<br/>')
                    p = Paragraph(para_with_breaks, styles['Justify'])
                    elements.append(p)
                    elements.append(Spacer(1, 0.1*inch))
        except Exception as e:
            logger.warning(f"Error with paragraph processing: {e}. Trying simpler approach.")
            # If the above fails, try a simpler approach with less formatting
            lines = escaped_text.split('\n')
            for line in lines:
                if line.strip():
                    p = Paragraph(line, styles['Normal'])
                    elements.append(p)
                else:
                    elements.append(Spacer(1, 0.1*inch))
        
        # Build the PDF
        doc.build(elements)
        logger.info(f"PDF generated successfully at {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating PDF: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract and analyze text from PDF files")
    parser.add_argument("--pdf", type=str, default="/home/suryaremanan/Documents/legal/Macromex_JBS_redacted/MCX_JBS agreement_03 02 2020_trk_Redacted.pdf", 
                        help="Path to the PDF file")
    parser.add_argument("--output", type=str, default="parsed_text.txt",
                        help="Path to save extracted text")
    parser.add_argument("--analysis", type=str, default="analysis_result.txt",
                        help="Path to save analysis results")
    parser.add_argument("--format", type=str, default="analysis",
                        help="Type of analysis to perform (analysis, summary, etc.)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for OCR processing (higher values may improve accuracy)")
    parser.add_argument("--ocr-only", action="store_true",
                        help="Skip other extraction methods and use OCR directly")
    parser.add_argument("--pdf-output", type=str, default="",
                        help="Path to save the extracted text as a PDF file")
    parser.add_argument("--analysis-pdf", type=str, default="",
                        help="Path to save the analysis as a PDF file")
    
    args = parser.parse_args()
    
    try:
        # Step 1: Extract text from PDF
        logger.info(f"Extracting text from PDF: {args.pdf}")
        
        if args.ocr_only and HAS_OCR:
            logger.info("Using OCR-only mode as requested")
            extracted_text = extract_text_with_ocr(args.pdf, dpi=args.dpi)
        else:
            extracted_text = extract_text_from_pdf(args.pdf)
        
        # Check if text was extracted
        if not extracted_text.strip():
            logger.error("No text was extracted from the PDF")
            return
        
        # Save extracted text to file
        with open(args.output, "w", encoding="utf-8") as txt_file:
            txt_file.write(extracted_text)
        logger.info(f"Extracted text ({len(extracted_text)} chars) saved to {args.output}")
        
        # Generate PDF output if requested
        if args.pdf_output and HAS_PDF_GENERATION:
            pdf_title = f"Extracted text from {os.path.basename(args.pdf)}"
            generate_pdf_from_text(extracted_text, args.pdf_output, pdf_title)
        
        # Step 2: Analyze the text using Together AI
        logger.info(f"Analyzing text with Together AI ({args.format})")
        analysis_result = analyze_text_with_together(extracted_text, args.format)
        
        # Save analysis to file
        with open(args.analysis, "w", encoding="utf-8") as analysis_file:
            analysis_file.write(analysis_result["analysis"])
        logger.info(f"Analysis saved to {args.analysis}")
        
        # Generate analysis PDF if requested
        if args.analysis_pdf and HAS_PDF_GENERATION:
            pdf_title = f"Analysis of {os.path.basename(args.pdf)}"
            generate_pdf_from_text(analysis_result["analysis"], args.analysis_pdf, pdf_title)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")

# Simple function to convert existing text file to PDF
def convert_text_file_to_pdf(text_file_path, pdf_output_path, title=None):
    """
    Converts an existing text file to PDF format.
    
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
            
        # Generate PDF
        return generate_pdf_from_text(text, pdf_output_path, title)
        
    except Exception as e:
        logger.error(f"Error converting text to PDF: {str(e)}")
        return False

if __name__ == "__main__":
    # If script is directly called with 'convert' command, convert text to PDF
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        if len(sys.argv) < 4:
            print("Usage: python parse_pdf.py convert <input_text_file> <output_pdf_file> [title]")
            sys.exit(1)
            
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        title = sys.argv[4] if len(sys.argv) > 4 else f"Converted from {os.path.basename(input_file)}"
        
        if convert_text_file_to_pdf(input_file, output_file, title):
            print(f"Successfully converted {input_file} to {output_file}")
        else:
            print(f"Failed to convert {input_file} to PDF")
            sys.exit(1)
    else:
        main()
