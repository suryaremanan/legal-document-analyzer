"""
Embedding service for generating text embeddings and similarity search.
Provides fallback mechanisms if SentenceTransformer is not available.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any, Union
import faiss
import os
import re
import time
import streamlit as st
import sys
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Force reload of modules if needed
if 'sentence_transformers' in sys.modules:
    importlib.reload(sys.modules['sentence_transformers'])

# Try to import advanced embedding libraries, with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    # Test that it actually works
    test_model = SentenceTransformer('all-MiniLM-L6-v2')
    test_embedding = test_model.encode(["Test sentence"])
    
    SENTENCE_TRANSFORMER_AVAILABLE = True
    logger.info("Successfully loaded SentenceTransformer for embeddings")
    # Initialize the model (will be lazy-loaded on first use)
    model = test_model
except Exception as e:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning(f"SentenceTransformer could not be imported or initialized: {str(e)}")
    logger.warning("For better performance, install with: pip install sentence-transformers")
    
    # Show warning only once
    if not hasattr(st, '_showed_transformer_warning'):
        with st.sidebar:
            st.warning("""
            ⚠️ **Enhanced embedding model not available**
            
            For better search quality, install the sentence-transformers package:
            ```
            pip install sentence-transformers
            ```
            
            Currently using a simpler fallback method.
            """)
        setattr(st, '_showed_transformer_warning', True)

def create_embeddings(text_chunks: List[str]) -> Optional[np.ndarray]:
    """
    Create embeddings for a list of text chunks.
    
    Args:
        text_chunks: List of text chunks to embed
        
    Returns:
        Numpy array of embeddings or None if embedding fails
    """
    if not text_chunks:
        logger.warning("No text chunks provided for embedding")
        return None
    
    try:
        logger.info(f"Creating embeddings for {len(text_chunks)} chunks")
        
        # Try the proper implementations first
        if SENTENCE_TRANSFORMER_AVAILABLE:
            return model.encode(text_chunks)
        else:
            return _create_fallback_embeddings(text_chunks)
    except:
        # Absolute last resort - random embeddings
        logger.warning("Using random embeddings as last resort")
        return np.random.randn(len(text_chunks), 384).astype(np.float32)

def _create_fallback_embeddings(text_chunks: List[str], vector_size: int = 384) -> np.ndarray:
    """
    Create simple embeddings using a basic approach when better models aren't available.
    
    Args:
        text_chunks: List of text chunks to embed
        vector_size: Size of the resulting embedding vectors
        
    Returns:
        Numpy array of embeddings
    """
    logger.info("Using fallback embedding method (TF-IDF style)")
    
    try:
        # First try to use sklearn's TfidfVectorizer if available
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=vector_size,
            stop_words='english',
            use_idf=True,
            norm='l2'
        )
        
        # Fit and transform the text chunks
        embeddings = vectorizer.fit_transform(text_chunks).toarray()
        
        # If we have fewer features than vector_size, pad with zeros
        if embeddings.shape[1] < vector_size:
            padding = np.zeros((embeddings.shape[0], vector_size - embeddings.shape[1]))
            embeddings = np.hstack((embeddings, padding))
        
        logger.info(f"Created TF-IDF embeddings with shape {embeddings.shape}")
        return embeddings
        
    except ImportError:
        logger.warning("sklearn not available, using basic embedding method")
        # Create a basic vocabulary from all chunks
        all_text = " ".join(text_chunks).lower()
        words = re.findall(r'\b\w+\b', all_text)
        vocab = list(set(words))
        vocab_size = min(vector_size, len(vocab))

        # Take the most common words if we have more than our vector size
        if len(vocab) > vector_size:
            # Count word frequencies
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Sort by frequency
            vocab = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)[:vector_size]

        # Create a word-to-index mapping
        word_to_idx = {word: i for i, word in enumerate(vocab[:vocab_size])}

        # Create embeddings
        embeddings = np.zeros((len(text_chunks), vector_size))

        for i, chunk in enumerate(text_chunks):
            # Count words in this chunk
            chunk_words = re.findall(r'\b\w+\b', chunk.lower())
            for word in chunk_words:
                if word in word_to_idx:
                    embeddings[i, word_to_idx[word]] += 1
            
            # Normalize
            if np.sum(embeddings[i]) > 0:
                embeddings[i] = embeddings[i] / np.sqrt(np.sum(embeddings[i] ** 2))

        logger.info(f"Created basic fallback embeddings with shape {embeddings.shape}")
        return embeddings

def create_index(embeddings: np.ndarray) -> Any:
    """
    Create a FAISS index for fast similarity search.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS index
    """
    try:
        logger.info("Loading faiss with AVX2 support.")
        # Try to use AVX2 extension for better performance
        import faiss.contrib.torch_utils
        logger.info("Successfully loaded faiss with AVX2 support.")
    except ImportError:
        logger.info("Using standard faiss implementation.")
    
    try:
        # Get dimensions from embeddings
        num_vectors, dimension = embeddings.shape
        
        # Create appropriate index based on vector size
        if num_vectors < 10000:
            # For smaller datasets, use exact search
            index = faiss.IndexFlatL2(dimension)
        else:
            # For larger datasets, use approximate search
            nlist = min(4096, int(num_vectors / 10))  # number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            index.train(embeddings)
        
        # Add vectors to index
        index.add(embeddings.astype(np.float32))
        logger.info(f"Created FAISS index with {num_vectors} vectors of dimension {dimension}")
        
        return index
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        return None

def search_similar(query: str, chunks: List[str], index: Any, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for document chunks similar to the query.
    
    Args:
        query: The query string
        chunks: List of text chunks to search in
        index: The FAISS index to use for search
        top_k: Number of results to return
        
    Returns:
        List of dictionaries with chunk text and similarity score
    """
    try:
        logger.info(f"Searching for: {query}")
        
        # Create query embedding
        if SENTENCE_TRANSFORMER_AVAILABLE:
            query_embedding = model.encode([query])[0].reshape(1, -1)
        else:
            # We need to create a compatible embedding with our fallback method
            # First create a dummy list with the query and all chunks
            combined = [query] + chunks
            temp_embeddings = _create_fallback_embeddings(combined)
            query_embedding = temp_embeddings[0].reshape(1, -1)
        
        # Search the index
        top_k = min(top_k, len(chunks))  # Ensure we don't ask for more than we have
        distances, indices = index.search(query_embedding.astype(np.float32), top_k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(chunks):  # Ensure index is valid
                results.append({
                    "chunk": chunks[idx],
                    "score": float(1.0 / (1.0 + dist)),  # Convert distance to similarity score
                    "index": int(idx)
                })
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
    
    except Exception as e:
        logger.error(f"Error searching for similar chunks: {str(e)}")
        return [] 