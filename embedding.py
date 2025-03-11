"""
Embedding module for creating and searching text embeddings.
"""

import numpy as np
import logging
import streamlit as st
import faiss
from typing import List, Tuple, Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import SentenceTransformer with fallback to simple embedding
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
    logger.info("SentenceTransformer successfully imported")
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("SentenceTransformer could not be imported. Using fallback embedding method.")

# Set default embedding dimension
EMBEDDING_DIMENSION = 384  # Common dimension for embedding models

# Alternative simple embedding method using bag of words
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

# Embedding model class
class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        global SENTENCE_TRANSFORMER_AVAILABLE
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
            logger.info("Using fallback bag-of-words embedding method")
            if isinstance(texts, str):
                return text_to_bow_vector(texts)
            else:
                return np.array([text_to_bow_vector(text) for text in texts])

# Cache the model instance using Streamlit's caching
@st.cache_resource
def get_embedding_model(model_name="all-MiniLM-L6-v2"):
    return EmbeddingModel(model_name)

def create_embeddings(text: str) -> Tuple[List[str], Any]:
    """
    Split text into chunks and create embeddings.
    
    Args:
        text: Text to process
        
    Returns:
        Tuple of (list of text chunks, faiss index with embeddings)
    """
    from pdf_processor import split_into_chunks
    
    try:
        # Split text into chunks
        chunks = split_into_chunks(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Get the embedding model
        model = get_embedding_model()
        
        # Create embeddings
        logger.info("Generating embeddings")
        embeddings = model.encode(chunks, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        logger.info(f"Created embeddings with shape {embeddings.shape}")
        
        return chunks, index
    except Exception as e:
        logger.error(f"Error in create_embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        # Return empty results to avoid breaking the application
        empty_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        return [], empty_index

def search_similar_chunks(query_text: str, 
                         index: Any, 
                         chunks: List[Dict[str, Any]], 
                         k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for chunks similar to the query text.
    
    Args:
        query_text: Query text to search for
        index: Vector index (faiss.Index or numpy.ndarray)
        chunks: List of text chunks with embeddings
        k: Number of results to return
        
    Returns:
        List of similar chunks with similarity scores
    """
    try:
        logger.info(f"Searching for chunks similar to query: {query_text[:50]}...")
        
        # Validate input types
        if not chunks or not isinstance(chunks, list):
            logger.error(f"Invalid chunks: {type(chunks)}")
            return []
            
        # Check if index is actually a string (error case)
        if isinstance(index, str):
            logger.error("Index is a string, not a vector index. Using fallback.")
            # Return first few chunks with mock similarity scores
            results = []
            for i, chunk in enumerate(chunks[:k]):
                if isinstance(chunk, dict):
                    chunk_copy = chunk.copy()
                else:
                    # If chunk is a string, convert to dict
                    chunk_copy = {"text": chunk, "id": i}
                chunk_copy['similarity'] = 0.9 - (i * 0.1)  # Mock similarity
                results.append(chunk_copy)
            return results
            
        # Get the embedding model and generate embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode([query_text])
        
        # Reshape for compatibility with search functions
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Ensure k is not larger than the number of chunks
        k = min(k, len(chunks))
        
        # If no chunks or k is 0, return empty list
        if k == 0:
            return []
        
        # Check if we're using faiss or numpy array
        if hasattr(index, 'search'):
            # Using faiss index
            distances, indices = index.search(query_embedding, k)
            
            # Get the similar chunks with distances
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(chunks):  # Safety check
                    # Check if chunk is dict or string
                    if isinstance(chunks[idx], dict):
                        chunk = chunks[idx].copy()
                    else:
                        chunk = {"text": chunks[idx], "id": idx}
                    chunk['similarity'] = float(1.0 - distances[0][i])  # Convert distance to similarity
                    results.append(chunk)
        else:
            # Using numpy array as fallback
            try:
                # Calculate cosine similarity manually
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(query_embedding, index)[0]
                
                # Get indices of top k similarities
                top_indices = similarities.argsort()[-k:][::-1]
                
                # Get the similar chunks with similarities
                results = []
                for idx in top_indices:
                    if idx < len(chunks):  # Safety check
                        if isinstance(chunks[idx], dict):
                            chunk = chunks[idx].copy()
                        else:
                            chunk = {"text": chunks[idx], "id": idx}
                        chunk['similarity'] = float(similarities[idx])
                        results.append(chunk)
            except Exception as e:
                logger.error(f"Error computing similarities: {str(e)}")
                # Emergency fallback - return first few chunks
                results = []
                for i, chunk in enumerate(chunks[:k]):
                    if isinstance(chunk, dict):
                        chunk_copy = chunk.copy()
                    else:
                        chunk_copy = {"text": chunk, "id": i}
                    chunk_copy['similarity'] = 0.9 - (i * 0.1)  # Mock similarity
                    results.append(chunk_copy)
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
    
    except Exception as e:
        logger.error(f"Error in search_similar_chunks: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Emergency fallback - return first few chunks with mock data
        try:
            results = []
            for i, chunk in enumerate(chunks[:k]):
                if isinstance(chunk, dict):
                    chunk_copy = chunk.copy()
                else:
                    # If chunk is a string, convert to dict
                    chunk_copy = {"text": str(chunk), "id": i}
                chunk_copy['similarity'] = 0.9 - (i * 0.1)  # Mock similarity
                results.append(chunk_copy)
            return results
        except:
            return [] 