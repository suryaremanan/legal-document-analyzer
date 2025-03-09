import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from pdf_processor import split_into_chunks
import os
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model for embeddings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Optional environment variable to disable embeddings for testing
DISABLE_EMBEDDINGS = os.environ.get("DISABLE_EMBEDDINGS", "").lower() in ["true", "1", "yes"]

def create_embeddings(text: str, chunk_size: int = 1000, overlap: int = 200, 
                     model_name: str = DEFAULT_EMBEDDING_MODEL) -> Tuple[List[str], np.ndarray]:
    """
    Split text into chunks and create embeddings.
    
    Args:
        text: Text to process
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        Tuple of (list of text chunks, numpy array of embeddings)
    """
    # Split text into chunks
    logger.info(f"Splitting text into chunks (size={chunk_size}, overlap={overlap})")
    chunks = split_into_chunks(text, chunk_size, overlap)
    logger.info(f"Created {len(chunks)} chunks")
    
    if DISABLE_EMBEDDINGS:
        logger.warning("Embeddings are disabled. Using random embeddings for testing.")
        # Create random embeddings for testing
        embeddings = np.random.random((len(chunks), 384))  # Common embedding size
        return chunks, embeddings
    
    try:
        # Load the model
        logger.info(f"Loading embedding model: {model_name}")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        except ImportError as e:
            logger.error(f"Error importing SentenceTransformer: {str(e)}")
            raise ImportError(f"Failed to import SentenceTransformer. Please install with: pip install sentence-transformers==2.1.0")
        
        # Create embeddings
        logger.info("Generating embeddings")
        embeddings = model.encode(chunks, show_progress_bar=True)
        
        logger.info(f"Created embeddings with shape {embeddings.shape}")
        
        return chunks, embeddings
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def search_similar_chunks(query: str, chunks: List[str], embeddings: np.ndarray, 
                         k: int = 3, model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[str]:
    """
    Search for chunks similar to the query using cosine similarity.
    
    Args:
        query: Query text
        chunks: List of text chunks
        embeddings: Numpy array of embeddings for chunks
        k: Number of similar chunks to return
        model_name: Name of the sentence-transformers model to use
        
    Returns:
        List of similar chunks
    """
    if DISABLE_EMBEDDINGS:
        logger.warning("Embeddings are disabled. Returning first chunks for testing.")
        return chunks[:min(k, len(chunks))]
    
    try:
        # Load the model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        
        # Encode the query
        query_embedding = model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings, query_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k similar chunks
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return the similar chunks
        return [chunks[i] for i in top_k_indices]
    except Exception as e:
        logger.error(f"Error in search_similar_chunks: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback to returning first k chunks
        return chunks[:min(k, len(chunks))] 