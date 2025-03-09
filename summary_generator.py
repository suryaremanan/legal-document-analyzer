"""
Summary Generator for PDF documents.
Generates summaries and key point extraction using SambaNova's Llama 3.1 model.
"""

import logging
from typing import Dict, Any, List, Tuple
from sambanova_api import get_llama_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummaryGenerator:
    """Generate document summaries using SambaNova's Llama 3.1 model."""
    
    def __init__(self):
        self.max_context_length = 6000  # Maximum context length to send to the model
    
    def generate_summary(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate a concise summary of the document.
        
        Args:
            text: Document text
            metadata: Optional metadata for context
            
        Returns:
            Dictionary with summary text and key points
        """
        # Truncate text if too long
        context_text = text[:self.max_context_length]
        
        # Create prompt with context and metadata hints
        metadata_hints = ""
        if metadata:
            if metadata.get("document_type") != "Unknown":
                metadata_hints += f"This is a {metadata.get('document_type')}. "
            
            if metadata.get("organizations"):
                # Handle both string and dictionary organizations
                org_names = []
                for org in metadata.get("organizations")[:3]:
                    if isinstance(org, dict):
                        # If organization is a dictionary, extract the name
                        org_name = org.get("name", str(org))
                        org_names.append(org_name)
                    else:
                        # If organization is already a string
                        org_names.append(str(org))
                
                if org_names:
                    orgs = ", ".join(org_names)
                    metadata_hints += f"It involves these organizations: {orgs}. "
                
            if metadata.get("dates") and len(metadata.get("dates")) > 0:
                # Handle potential non-string dates
                date = metadata.get("dates")[0]
                if isinstance(date, (dict, list)):
                    date = str(date)
                metadata_hints += f"The primary date is {date}. "
        
        # Build the summary prompt
        summary_prompt = f"""
You are a legal document analysis assistant. Please provide a concise summary of the following document.
{metadata_hints}

BEGIN DOCUMENT TEXT:
{context_text}
END DOCUMENT TEXT

Please provide:
1. A concise summary (3-5 sentences) of what this document is about
2. The main parties involved
3. Key dates mentioned
4. Important terms or conditions
5. Any notable obligations or requirements

Format your response in clear sections with headings.
"""

        logger.info("Generating document summary")
        summary_response = get_llama_response(summary_prompt, temperature=0.1, max_tokens=1000)
        
        # Generate key points separately
        key_points_prompt = f"""
You are a legal document analysis assistant. Based on the text below, identify the 5-7 most important key points or takeaways. 
{metadata_hints}

BEGIN DOCUMENT TEXT:
{context_text}
END DOCUMENT TEXT

List only the most important points that someone should know about this document. Format as a bulleted list.
"""

        logger.info("Generating key points")
        key_points_response = get_llama_response(key_points_prompt, temperature=0.1, max_tokens=800)
        
        return {
            "summary": summary_response,
            "key_points": key_points_response
        }

    def generate_document_comparison(self, texts: List[str], titles: List[str]) -> str:
        """
        Generate a comparison between multiple documents.
        
        Args:
            texts: List of document texts
            titles: List of document titles
            
        Returns:
            Comparison text
        """
        if len(texts) < 2:
            return "Need at least two documents to compare."
        
        # Limit to comparing 2-3 documents
        texts = texts[:3]
        titles = titles[:3]
        
        # Create context snippets (first 2000 chars of each doc)
        context_snippets = [text[:2000] for text in texts]
        
        comparison_prompt = f"""
You are a legal document comparison assistant. Compare the following {len(texts)} documents:

"""
        
        # Add each document with its title
        for i, (title, snippet) in enumerate(zip(titles, context_snippets)):
            comparison_prompt += f"\nDOCUMENT {i+1}: {title}\n{snippet}\n"
        
        comparison_prompt += """
Please provide:
1. The main similarities between these documents
2. The key differences between these documents
3. Which document appears to be more favorable and why

Format your response in clear sections with headings.
"""
        
        logger.info(f"Generating comparison between {len(texts)} documents")
        comparison_response = get_llama_response(comparison_prompt, temperature=0.1, max_tokens=1200)
        
        return comparison_response 