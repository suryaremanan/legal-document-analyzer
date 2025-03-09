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
    logger.info("Generating response using retrieved context")
    
    # Combine context chunks into a single context
    combined_context = "\n\n".join(context_chunks)
    
    # Add metadata and summary information if available
    metadata_context = ""
    if documents:
        metadata_context = "DOCUMENT METADATA:\n"
        for doc_id, doc in documents.items():
            metadata_context += f"Document: {doc.get('filename', 'Unknown')}\n"
            
            # Add metadata information
            if doc.get("metadata"):
                metadata = doc.get("metadata")
                metadata_context += f"Title: {metadata.get('title', 'Unknown')}\n"
                metadata_context += f"Type: {metadata.get('document_type', 'Unknown')}\n"
                
                # Add organizations
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
                
                # Add people information
                if metadata.get('people'):
                    people_names = []
                    for person in metadata.get('people')[:3]:
                        if isinstance(person, dict):
                            person_name = person.get("name", str(person))
                            people_names.append(person_name)
                        else:
                            people_names.append(str(person))
                    
                    if people_names:
                        people = ", ".join(people_names)
                        metadata_context += f"People: {people}\n"
                
                # Add date information
                if metadata.get('dates') and len(metadata.get('dates')) > 0:
                    # Handle potential non-string dates
                    date = metadata.get('dates')[0]
                    if isinstance(date, (dict, list)):
                        date = str(date)
                    metadata_context += f"Key date: {date}\n"
                
                # Add monetary values if available
                if metadata.get('monetary_values') and len(metadata.get('monetary_values')) > 0:
                    value = metadata.get('monetary_values')[0]
                    metadata_context += f"Key value: {value}\n"
                    
            # Add a brief summary if available
            if doc.get("summary"):
                summary = doc.get("summary")
                # Take just the first paragraph to keep it brief
                first_para = summary.split("\n\n")[0] if "\n\n" in summary else summary
                if len(first_para) > 300:
                    first_para = first_para[:300] + "..."
                metadata_context += f"Summary: {first_para}\n"
                
            # Add key points if available
            if doc.get("key_points"):
                key_points = doc.get("key_points")
                metadata_context += f"Key Points: {key_points[:300]}...\n" if len(key_points) > 300 else f"Key Points: {key_points}\n"
                
            metadata_context += "\n"
    
    # Construct the prompt with context
    prompt = f"""You are a legal document analysis assistant. Use the document context and metadata provided below to answer the user's question accurately. If the information needed is not in the provided context, respond with "I don't have enough information to answer that question."

METADATA AND DOCUMENT SUMMARY:
{metadata_context}

DOCUMENT CONTEXT:
{combined_context}

USER QUESTION:
{query}

Your response should be:
1. Concise (1-3 sentences maximum)
2. Accurate based only on information in the provided context and metadata
3. Direct and to the point without ANY repetition
4. Clear in stating "I don't have enough information" if the answer isn't in the context

IMPORTANT: Do not repeat any sentence or information. Do not restate the same answer. Provide a single, clear response.
"""
    
    logger.info(f"Generating response for query: {query}")
    
    # Get response from SambaNova Llama API
    response = get_llama_response(prompt)
    
    return response 