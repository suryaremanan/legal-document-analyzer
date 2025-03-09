from typing import List, Dict, Any
import logging
from sambanova_api import get_llama_response

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_response(query: str, context_chunks: List[str], documents: Dict[str, Dict] = None) -> str:
    """
    Generate a response to the query using retrieved context chunks and document metadata/summaries.
    
    Args:
        query: User query
        context_chunks: List of relevant text chunks retrieved from the document
        documents: Dictionary of document information including metadata and summaries
        
    Returns:
        Generated response
    """
    # Combine context chunks
    combined_context = "\n\n".join(context_chunks)
    
    # Add metadata and summary information if available
    metadata_context = ""
    if documents:
        metadata_context = "DOCUMENT OVERVIEW:\n"
        for doc_id, doc in documents.items():
            metadata = doc.get("metadata", {})
            metadata_context += f"\nDocument: {doc.get('filename')}\n"
            metadata_context += f"Type: {metadata.get('document_type', 'Unknown')}\n"
            
            if metadata.get('organizations'):
                # Handle both string and dictionary organizations
                org_names = []
                for org in metadata.get('organizations')[:3]:
                    if isinstance(org, dict):
                        # If organization is a dictionary, extract the name
                        org_name = org.get("name", str(org))
                        org_names.append(org_name)
                    else:
                        # If organization is already a string
                        org_names.append(str(org))
                
                if org_names:
                    orgs = ", ".join(org_names)
                    metadata_context += f"Organizations: {orgs}\n"
            
            if metadata.get('dates') and len(metadata.get('dates')) > 0:
                # Handle potential non-string dates
                date = metadata.get('dates')[0]
                if isinstance(date, (dict, list)):
                    date = str(date)
                metadata_context += f"Key date: {date}\n"
                
            # Add a brief summary if available
            if doc.get("summary"):
                first_para = doc.get("summary").split("\n\n")[0]
                if len(first_para) > 300:
                    first_para = first_para[:300] + "..."
                metadata_context += f"Summary: {first_para}\n"
    
    # Create prompt with enhanced context
    prompt = f"""You are a legal document analysis assistant that helps answer questions based on provided document context.
    
{metadata_context}

DETAILED CONTEXT:
{combined_context}

USER QUESTION:
{query}

Your response should be:
1. Concise (1-3 sentences maximum)
2. Accurate based only on information in the provided context
3. Direct and to the point without ANY repetition
4. Clear in stating "I don't have enough information" if the answer isn't in the context

IMPORTANT: Do not repeat any sentence or information. Do not restate the same answer. Provide a single, clear response.
"""
    
    logger.info(f"Generating response for query: {query}")
    
    # Get response from SambaNova Llama API
    response = get_llama_response(prompt)
    
    return response 