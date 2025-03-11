from typing import List, Dict, Any
import logging
from sambanova_api import get_llama_response
from embedding import search_similar_chunks

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_response(query: str, 
                     chunks: List[Dict[str, Any]], 
                     index: Any, 
                     k: int = 5, 
                     temperature: float = 0.1) -> str:
    """
    Generate a response to a query using retrieval-augmented generation.
    
    Args:
        query: User query
        chunks: List of text chunks
        index: Vector index for similarity search
        k: Number of chunks to retrieve
        temperature: Temperature for response generation
        
    Returns:
        Generated response
    """
    try:
        logger.info(f"Generating response for query: {query}")
        
        # Get similar chunks for context
        similar_chunks = search_similar_chunks(query, index, chunks, k)
        
        # If no similar chunks found, return generic response
        if not similar_chunks:
            return "I couldn't find any relevant information to answer your question. Could you please rephrase or ask something else?"
        
        # Combine chunks into context
        context = ""
        for i, chunk in enumerate(similar_chunks):
            context += f"\nDocument {i+1} (Score: {chunk['similarity']:.2f}):\n{chunk['text']}\n"
        
        # Calculate estimated token count - assuming ~4 chars per token
        estimated_tokens = (len(context) + len(query)) // 4
        
        # If context is too large, limit it
        max_context_tokens = 14000  # Leave room for response
        
        if estimated_tokens > max_context_tokens:
            logger.warning(f"Context too large ({estimated_tokens} tokens). Trimming to {max_context_tokens} tokens.")
            
            # Prioritize most similar chunks
            similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Reset context
            context = ""
            remaining_tokens = max_context_tokens
            
            # Add chunks until we hit token limit
            for i, chunk in enumerate(similar_chunks):
                chunk_tokens = len(chunk['text']) // 4
                if chunk_tokens < remaining_tokens:
                    context += f"\nDocument {i+1} (Score: {chunk['similarity']:.2f}):\n{chunk['text']}\n"
                    remaining_tokens -= chunk_tokens
                else:
                    # Add partial chunk if possible
                    partial_text = chunk['text'][:remaining_tokens * 4]
                    context += f"\nDocument {i+1} (Score: {chunk['similarity']:.2f}) [truncated]:\n{partial_text}\n"
                    break
        
        # Create the prompt
        prompt = f"""
You are a helpful assistant that answers questions based on the provided document context.
Use only information from the context to answer the question.
If the context doesn't contain the answer, say so clearly - don't make up information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a detailed, accurate answer based only on the context above.
"""
        
        # Get response from API
        response = get_llama_response(prompt, temperature=temperature, max_tokens=800)
        
        # Handle failed API response
        if not response:
            return "I apologize, but I couldn't generate a response at this time. Please try again later."
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"An error occurred while generating a response: {str(e)}" 