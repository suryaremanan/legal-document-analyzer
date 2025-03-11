"""
Module for interacting with the SambaNova API for LLaMA model access.
"""

import requests
import json
import os
import logging
import time
import random
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    load_dotenv()
    logger.info("Environment variables loaded from .env file")
except Exception as e:
    logger.warning(f"Could not load .env file: {str(e)}")

# API configuration with the correct chat completions endpoint
SAMBANOVA_API_ENDPOINT = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MODEL = "Meta-Llama-3.1-8B-Instruct"

# Use the provided API key
SAMBANOVA_API_KEY = "9c0f7791-a5ee-4b3f-adfb-e83a086d8e6e"

# Rate limiting parameters
MAX_REQUESTS_PER_MINUTE = 10  # Maximum allowed requests per minute
REQUEST_WINDOW_SIZE = 60  # Window size in seconds
_request_timestamps = []  # List to track request timestamps

# Print API key status (but not the actual key)
if SAMBANOVA_API_KEY:
    logger.info("✅ SambaNova API key is set")
    print("✅ SambaNova API key is set")
else:
    logger.warning("⚠️ SambaNova API key is not set")
    print("⚠️ SambaNova API key is not set")

def _throttle_requests():
    """
    Implement client-side rate limiting by checking recent request count
    and delaying if necessary.
    """
    global _request_timestamps
    current_time = time.time()
    
    # Clean up old timestamps
    _request_timestamps = [ts for ts in _request_timestamps if current_time - ts < REQUEST_WINDOW_SIZE]
    
    # Check if we're at the rate limit
    if len(_request_timestamps) >= MAX_REQUESTS_PER_MINUTE:
        # Calculate required wait time
        oldest_timestamp = min(_request_timestamps)
        wait_time = REQUEST_WINDOW_SIZE - (current_time - oldest_timestamp) + 1
        
        if wait_time > 0:
            logger.info(f"Rate limit approaching. Waiting {wait_time:.2f} seconds before next request.")
            time.sleep(wait_time)
    
    # Add current timestamp to the list
    _request_timestamps.append(time.time())

def get_llama_response(prompt: str, 
                      temperature: float = 0.0, 
                      max_tokens: int = 800,
                      top_p: float = 0.95,
                      stop_sequences: Optional[list] = None) -> Optional[str]:
    """
    Get a response from SambaNova's Llama 3.1 model using the chat completions API.
    
    Args:
        prompt: The prompt text to send to the model
        temperature: Controls randomness (0 to 1)
        max_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
        stop_sequences: Optional list of sequences to stop generation
        
    Returns:
        Generated text response or None on failure
    """
    if not SAMBANOVA_API_KEY:
        logger.error("SambaNova API key not set.")
        return None
    
    # Check prompt length and truncate if needed
    # Llama 3.1 has a 16k token limit, conservatively estimate 4 chars per token
    max_prompt_chars = 15000 * 4  # Leave room for response
    
    if len(prompt) > max_prompt_chars:
        logger.warning(f"Prompt too long ({len(prompt)} chars). Truncating to {max_prompt_chars} chars...")
        
        # Find a good breaking point (paragraph)
        breakpoint = prompt.rfind("\n\n", 0, max_prompt_chars)
        if breakpoint == -1:
            # If no good paragraph break, find a sentence break
            breakpoint = prompt.rfind(". ", 0, max_prompt_chars)
        
        if breakpoint == -1:
            # If no good sentence break, just cut at the limit
            prompt = prompt[:max_prompt_chars]
        else:
            prompt = prompt[:breakpoint]
            
        # Add note about truncation
        prompt += "\n\n[Note: Context was truncated due to length limits]"
    
    # Log information about the request
    logger.info(f"Sending request to SambaNova chat API with prompt length: {len(prompt)}")
    
    # Apply client-side rate limiting
    _throttle_requests()
    
    # Set headers according to SambaNova documentation
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}"
    }
    
    # Prepare the payload for chat completions API
    payload = {
        "model": SAMBANOVA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }
    
    if stop_sequences:
        payload["stop"] = stop_sequences
    
    # Implement retry logic for resilience with exponential backoff
    max_retries = 5
    base_delay = 1.0  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                SAMBANOVA_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Log the response details for debugging
            logger.info(f"API Response Status: {response.status_code}")
            
            if response.status_code == 200:
                # Successfully got a response
                try:
                    data = response.json()
                    logger.info(f"API Response structure: {list(data.keys())}")
                    
                    # Parse response format for chat completions API
                    if "choices" in data and len(data["choices"]) > 0:
                        if "message" in data["choices"][0]:
                            content = data["choices"][0]["message"].get("content", "")
                            logger.info(f"Successfully extracted content: {content[:50]}...")
                            return content
                        else:
                            logger.warning(f"No 'message' field in choices: {data['choices'][0]}")
                            return str(data["choices"][0])
                    else:
                        logger.warning(f"Unexpected API response format: {data}")
                        return None
                except Exception as e:
                    logger.error(f"Error parsing API response: {str(e)}")
                    return None
            elif response.status_code == 429:
                # Rate limit exceeded - implement exponential backoff
                retry_after = response.headers.get('Retry-After')
                
                if retry_after and retry_after.isdigit():
                    # If server specifies retry time, use that
                    wait_time = int(retry_after)
                else:
                    # Otherwise use exponential backoff with jitter
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                logger.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                
                # Retry on 5xx errors
                if (500 <= response.status_code < 600) and attempt < max_retries - 1:
                    # Calculate delay with exponential backoff and jitter
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.info(f"Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                return None
                
        except Exception as e:
            logger.error(f"Request exception: {str(e)}")
            
            if attempt < max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.info(f"Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            return None
    
    logger.error("All API attempts failed")
    return None

def get_models() -> Dict[str, Any]:
    """
    Get a list of available models from the SambaNova API.
    
    Returns:
        Dictionary containing model information or error
    """
    if not SAMBANOVA_API_KEY:
        logger.error("SambaNova API key not set.")
        return {"error": "API key not configured", "models": [SAMBANOVA_MODEL]}
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SAMBANOVA_API_KEY}"
        }
        
        # Use the models endpoint matching our completions endpoint
        response = requests.get(
            "https://api.sambanova.ai/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"SambaNova API error: {response.status_code} - {response.text}")
            return {"error": f"API returned status code {response.status_code}", "models": [SAMBANOVA_MODEL]}
    
    except Exception as e:
        logger.error(f"Error getting models from SambaNova API: {str(e)}")
        return {"error": str(e), "models": [SAMBANOVA_MODEL]} 
