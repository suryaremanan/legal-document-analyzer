"""
Summary Generator for PDF documents.
Generates summaries and key point extraction using SambaNova's Llama 3.1 model.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from sambanova_api import get_llama_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummaryGenerator:
    """
    Class for generating summaries of documents using LLMs.
    """
    
    def __init__(self):
        """Initialize the summary generator."""
        self.max_context_length = 6000  # Maximum context length to send to the model
    
    def generate_summary(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Generate a summary of the document text in clean markdown format.
        
        Args:
            text: The document text to summarize
            metadata: Optional metadata about the document
            
        Returns:
            Dictionary containing summary and key points
        """
        try:
            logger.info("Generating document summary")
            
            # Truncate text if too long
            context_text = text[:self.max_context_length]
            
            # Create prompt with context and metadata hints
            metadata_hints = ""
            if metadata:
                if metadata.get("title"):
                    metadata_hints += f"Title: {metadata.get('title')}. "
                
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
            
            # Build the summary prompt
            summary_prompt = f"""
You are a legal document analysis assistant. Please provide a clear, comprehensive summary of the following document.
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

Format your response in clean markdown with sections and bullet points where appropriate.
Write using only information directly from the document. Do not include information not found in the document.
"""
            
            # Get summary from LLM
            summary_response = get_llama_response(summary_prompt, temperature=0.1, max_tokens=1000)
            
            # Check if response is valid - handle None case
            if summary_response is None:
                logger.warning("Received None response for summary generation")
                summary = self._generate_fallback_summary(text, metadata)
            elif summary_response.startswith("Error:"):
                logger.warning(f"Error in summary generation: {summary_response}")
                summary = self._generate_fallback_summary(text, metadata)
            else:
                # Ensure the summary is properly formatted as markdown
                summary = self._format_as_markdown(summary_response)
            
            # Generate key points
            key_points = self.generate_key_points(text, metadata)
            
            return {
                "summary": summary,
                "key_points": key_points
            }
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Provide fallback summary
            return {
                "summary": self._generate_fallback_summary(text, metadata),
                "key_points": "- Unable to generate key points due to an error\n- Please try again later"
            }
    
    def _format_as_markdown(self, text: str) -> str:
        """Ensure text is properly formatted as markdown"""
        if not text:
            return "No summary could be generated."
        
        # Add line breaks after sections if not present
        text = re.sub(r'(#+\s.*?)(\n[^#])', r'\1\n\2', text)
        
        # Ensure bullet points have proper spacing
        text = re.sub(r'(\n[*-])', r'\n\1', text)
        
        return text
    
    def generate_key_points(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate key points from the document text.
        
        Args:
            text: The document text
            metadata: Optional metadata about the document
            
        Returns:
            Key points as a string
        """
        try:
            logger.info("Generating key points")
            
            # Use only the first part of the text to avoid token limits
            context_text = text[:self.max_context_length]
            
            # Create a metadata hint string if available
            metadata_hint = ""
            if metadata:
                if metadata.get("document_type") != "Unknown":
                    metadata_hint = f"This is a {metadata.get('document_type')}. "
            
            # Create prompt for key points extraction
            key_points_prompt = f"""
You are a legal document analyzer. Extract 5-7 key points from the following document text.
{metadata_hint}

DOCUMENT TEXT:
{context_text}

For each key point:
1. Focus on the most important legal and business aspects
2. Be specific and precise
3. Include any critical dates, values, or conditions
4. Avoid general statements that could apply to any document
5. Use concise bullet points

KEY POINTS:
"""
            
            # Get key points from LLM
            key_points_response = get_llama_response(key_points_prompt, temperature=0.1, max_tokens=800)
            
            # Handle case where the API returns None
            if key_points_response is None or key_points_response.startswith("Error:"):
                logger.warning(f"Error in key points generation: {key_points_response}")
                # Fall back to rule-based key points
                return self._generate_fallback_key_points(text, metadata)
            
            return key_points_response
        except Exception as e:
            logger.error(f"Error generating key points: {str(e)}")
            # Fall back to rule-based key points
            return self._generate_fallback_key_points(text, metadata)
    
    def _generate_fallback_summary(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a fallback summary when the API fails, using document analysis.
        
        Args:
            text: Document text
            metadata: Optional document metadata
            
        Returns:
            A fallback summary based on text analysis
        """
        # Create a meaningful fallback summary using metadata and text analysis
        summary_parts = []
        
        # 1. Start with document type and title if available
        doc_type = "document"
        doc_title = "untitled document"
        
        if metadata:
            if metadata.get('document_type') and metadata.get('document_type') != "Unknown":
                doc_type = metadata.get('document_type').lower()
            if metadata.get('title') and metadata.get('title') != "Unknown Document":
                doc_title = metadata.get('title')
        
        summary_parts.append(f"# {doc_title.title()}")
        summary_parts.append(f"\n## Overview")
        
        # 2. Basic document information
        overview = f"This {doc_type} contains approximately {len(text.split())} words."
        
        # Add organizations if available
        if metadata and metadata.get('organizations') and len(metadata.get('organizations', [])) > 0:
            orgs = metadata.get('organizations', [])
            org_names = []
            
            for org in orgs:
                if isinstance(org, dict):
                    org_names.append(org.get('name', 'Unknown'))
                else:
                    org_names.append(str(org))
            
            if len(org_names) == 1:
                overview += f" It involves {org_names[0]}."
            elif len(org_names) == 2:
                overview += f" It involves {org_names[0]} and {org_names[1]}."
            elif len(org_names) > 2:
                org_list = ", ".join(org_names[:-1]) + f" and {org_names[-1]}"
                overview += f" It involves multiple parties including {org_list}."
        
        summary_parts.append(overview)
        
        # 3. Find key sections from the document
        summary_parts.append("\n## Document Sections")
        
        # Look for section headers in the text (uppercase words followed by newlines, or numbered sections)
        section_patterns = [
            r'\n([A-Z][A-Z\s]{5,})\n',  # UPPERCASE SECTIONS
            r'\n((?:Article|Section|Clause)\s+\d+[.:]\s*[A-Z][A-Za-z\s]+)',  # Article/Section headings
            r'\n(\d+\.\s+[A-Z][A-Za-z\s]+)'  # Numbered sections
        ]
        
        sections = []
        for pattern in section_patterns:
            section_matches = re.findall(pattern, "\n" + text)
            sections.extend([s.strip() for s in section_matches])
        
        # Include up to 5 key sections
        if sections:
            sections = sections[:5]
            section_text = "The document appears to contain the following sections:\n"
            for section in sections:
                section_text += f"- {section}\n"
            summary_parts.append(section_text)
        else:
            summary_parts.append("The document doesn't have clearly identifiable sections.")
        
        # 4. Extract dates if available
        if metadata and metadata.get('dates') and len(metadata.get('dates', [])) > 0:
            summary_parts.append("\n## Key Dates")
            date_text = "Important dates mentioned in the document:\n"
            
            for date in metadata.get('dates')[:3]:
                date_text += f"- {date}\n"
            
            summary_parts.append(date_text)
        
        # 5. Disclaimer about automatic analysis
        summary_parts.append("\n*Note: This is an automated analysis of the document. For complete details, please review the full text.*")
        
        # Combine all summary parts
        return "\n".join(summary_parts)
    
    def _generate_fallback_key_points(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate fallback key points based on text analysis without relying on an LLM.
        
        Args:
            text: Document text
            metadata: Optional document metadata
            
        Returns:
            Fallback key points as a string
        """
        # Generate key points using text analysis
        key_points = []
        
        # 1. Document type and parties
        if metadata:
            if metadata.get('document_type') and metadata.get('document_type') != "Unknown":
                key_points.append(f"Document type: {metadata.get('document_type')}")
            
            if metadata.get('organizations') and len(metadata.get('organizations', [])) > 0:
                orgs = metadata.get('organizations', [])
                org_names = []
                
                for org in orgs[:2]:  # Limit to top 2 organizations
                    if isinstance(org, dict):
                        org_names.append(org.get('name', 'Unknown'))
                    else:
                        org_names.append(str(org))
                
                if org_names:
                    key_points.append(f"Main parties: {', '.join(org_names)}")
        
        # 2. Look for key terms like "payment", "term", "termination", "confidentiality"
        key_terms = {
            "payment": r'\b(?:payment|fee|compensation|amount)\b.{10,70}',
            "term": r'\bterm\s+of\s+(?:this|the)\s+(?:agreement|contract).{10,70}',
            "termination": r'\btermination.{10,70}',
            "confidentiality": r'\bconfidentiality\b.{10,70}',
            "liability": r'\bliability\b.{10,70}'
        }
        
        for term, pattern in key_terms.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Get the first match and clean it up
                match = matches[0].strip()
                match = re.sub(r'\s+', ' ', match)  # Remove extra whitespace
                key_points.append(f"Contains {term} provisions: \"{match}...\"")
        
        # 3. Add dates if available
        if metadata and metadata.get('dates') and len(metadata.get('dates', [])) > 0:
            first_date = metadata.get('dates')[0]
            key_points.append(f"Key date mentioned: {first_date}")
        
        # 4. Add monetary values if available
        if metadata and metadata.get('monetary_values') and len(metadata.get('monetary_values', [])) > 0:
            monetary_values = metadata.get('monetary_values')
            if monetary_values:
                key_points.append(f"Contains monetary values: {', '.join(monetary_values[:2])}")
        
        # 5. Add basic statistics
        word_count = len(text.split())
        key_points.append(f"Document length: Approximately {word_count} words")
        
        # Format key points with bullet points
        return "\n".join([f"- {point}" for point in key_points])

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

def generate_document_summary(text: str, metadata: Dict = None, max_length: int = 1500) -> str:
    """
    Generate a comprehensive summary of a legal document using LLM.
    
    Args:
        text: Full text of the document
        metadata: Optional metadata extracted from the document
        max_length: Maximum length of text to send to the LLM
        
    Returns:
        Generated summary text
    """
    try:
        logger.info("Generating document summary")
        
        # Prepare a truncated version of the text if it's too long
        truncated_text = text
        if len(text) > max_length:
            # Take the first part, likely containing important info
            first_part = text[:int(max_length * 0.7)]
            # And some from the end, which might have conclusions
            last_part = text[-int(max_length * 0.3):]
            truncated_text = first_part + "\n[...]\n" + last_part
            logger.info(f"Truncated text from {len(text)} to {len(truncated_text)} characters")
        
        # Create a prompt for the LLM
        prompt = f"""You are a legal expert tasked with summarizing a legal document. 
Please provide a comprehensive summary of the following document text.
Focus on key provisions, parties involved, obligations, dates, and any significant legal implications.

DOCUMENT TEXT:
{truncated_text}

Please structure your summary with the following sections:
1. Overview (brief description of document type and purpose)
2. Key Parties
3. Key Dates
4. Main Provisions
5. Notable Clauses or Terms
6. Potential Legal Implications
"""

        # Add metadata hints if available
        if metadata:
            prompt += "\n\nMetadata extracted from the document:"
            for key, value in metadata.items():
                if value and key not in ["source_document", "filename"]:
                    prompt += f"\n- {key}: {value}"
        
        # Get summary from LLM
        summary_response = get_llama_response(prompt, max_tokens=1500)
        
        if summary_response:
            # Clean up the response if needed
            summary = summary_response.strip()
            logger.info("Successfully generated document summary")
            return summary
        else:
            logger.error("Failed to generate summary: Empty response from LLM")
            return "Failed to generate summary. Please try again."
    
    except Exception as e:
        logger.error(f"Error generating document summary: {str(e)}")
        return f"Error generating summary: {str(e)}"

def compare_documents(doc1_text: str, doc2_text: str) -> str:
    """
    Compare two legal documents and highlight differences.
    
    Args:
        doc1_text: Text of the first document
        doc2_text: Text of the second document
        
    Returns:
        Markdown formatted comparison summary
    """
    try:
        logger.info("Comparing two documents")
        
        # Create a truncated version if documents are too long
        max_length = 1500  # characters per document
        
        doc1_truncated = doc1_text[:max_length] if len(doc1_text) > max_length else doc1_text
        doc2_truncated = doc2_text[:max_length] if len(doc2_text) > max_length else doc2_text
        
        # Create prompt for the LLM
        prompt = f"""You are a legal expert tasked with comparing two legal documents.
Please identify and explain the significant differences between them.

DOCUMENT 1:
{doc1_truncated}

DOCUMENT 2:
{doc2_truncated}

Please provide your analysis in the following format:
1. Overview of Documents
2. Key Differences (by section or topic)
3. Implications of Changes
4. Recommendation

Focus on substantive differences that would impact legal interpretation, rather than minor wording changes.
"""
        
        # Get comparison from LLM
        comparison_response = get_llama_response(prompt, max_tokens=2000)
        
        if comparison_response:
            # Format the response in Markdown
            comparison = comparison_response.strip()
            logger.info("Successfully generated document comparison")
            return comparison
        else:
            logger.error("Failed to generate comparison: Empty response from LLM")
            return "Failed to compare documents. Please try again."
    
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        return f"Error comparing documents: {str(e)}" 