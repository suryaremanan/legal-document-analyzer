# First attempt to import SentenceTransformer with error handling
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    import logging
    logging.warning("SentenceTransformer could not be imported. Using fallback embedding method.")

import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from pdf_processor import split_into_chunks
import os
import traceback
import streamlit as st
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default model for embeddings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Optional environment variable to disable embeddings for testing
DISABLE_EMBEDDINGS = os.environ.get("DISABLE_EMBEDDINGS", "").lower() in ["true", "1", "yes"]

# Set default embedding model or cached path
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for the all-MiniLM-L6-v2 model

# Alternative simple embedding method using bag of words or TF-IDF
def text_to_bow_vector(text, dimension=EMBEDDING_DIMENSION):
    """Convert text to a bag of words vector (simplified embedding)"""
    import re
    from collections import Counter
    import hashlib
    
    # Clean and tokenize text
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    
    # Count tokens
    token_counts = Counter(tokens)
    
    # Create a simple embedding using hash functions to map to dimensions
    vector = np.zeros(dimension, dtype=np.float32)
    
    for token, count in token_counts.items():
        # Hash the token to determine which dimension to increment
        hash_value = int(hashlib.md5(token.encode()).hexdigest(), 16)
        dim = hash_value % dimension
        # Use the count as the value
        vector[dim] += count
    
    # Normalize vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


# Create a class that will load the model or use the fallback
class EmbeddingModel:
    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                logger.info(f"Loading SentenceTransformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("SentenceTransformer model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer model: {str(e)}")
                SENTENCE_TRANSFORMER_AVAILABLE = False
        
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.warning("Using fallback embedding method")
    
    def encode(self, texts, show_progress_bar=True):
        """Encode text into embeddings"""
        if SENTENCE_TRANSFORMER_AVAILABLE and self.model:
            return self.model.encode(texts, show_progress_bar=show_progress_bar)
        else:
            # Use fallback method
            if isinstance(texts, str):
                return text_to_bow_vector(texts)
            else:
                return np.array([text_to_bow_vector(text) for text in texts])


# Cache the model instance using Streamlit's caching
@st.cache_resource
def get_embedding_model(model_name=DEFAULT_MODEL_NAME):
    return EmbeddingModel(model_name)


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
        model = get_embedding_model()
        
        # Create embeddings
        logger.info("Generating embeddings")
        embeddings = model.encode(chunks, show_progress_bar=True)
        
        logger.info(f"Created embeddings with shape {embeddings.shape}")
        
        return chunks, embeddings
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def search_similar_chunks(query: str, chunks: List[str], index: Any, top_k: int = 5) -> List[str]:
    """
    Search for chunks similar to the query.
    
    Args:
        query: Query text
        chunks: Original text chunks
        index: FAISS index
        top_k: Number of results to return
        
    Returns:
        List of similar text chunks
    """
    try:
        logger.info(f"Searching for chunks similar to: {query}")
        
        model = get_embedding_model()
        
        # Generate query embedding
        query_embedding = model.encode(query)
        
        # Reshape for FAISS
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search the index
        distances, indices = index.search(query_embedding, min(top_k, len(chunks)))
        
        # Get the text of the most similar chunks
        similar_chunks = [chunks[idx] for idx in indices[0]]
        
        logger.info(f"Found {len(similar_chunks)} similar chunks")
        return similar_chunks
    
    except Exception as e:
        logger.error(f"Error searching similar chunks: {str(e)}")
        # Return a subset of random chunks if search fails
        # This prevents the app from crashing
        import random
        sample_size = min(top_k, len(chunks))
        return random.sample(chunks, sample_size) if sample_size > 0 else [] 