import os
import requests
import json
import logging
from typing import Dict, Any, Optional

try:
    import dotenv
    dotenv.load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed, using environment variables directly")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SambaNova API endpoint and credentials
SAMBANOVA_API_URL = "https://api.sambanova.ai/v1/completions"
# If environment variable is not set, use the hardcoded key
# SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY", "")
SAMBANOVA_API_KEY = "59e66a78-70e4-4846-9d76-a3f64513bd39"  # Hardcoded for testing

# Print API key status (but not the actual key)
if SAMBANOVA_API_KEY:
    print("✅ SambaNova API key is set")
else:
    print("❌ SambaNova API key is not set")

def get_llama_response(prompt: str, 
                      temperature: float = 0.0, 
                      max_tokens: int = 800,
                      top_p: float = 0.95,
                      stop_sequences: Optional[list] = None) -> str:
    """
    Get a response from SambaNova's Llama 3.1 model.
    
    Args:
        prompt: The prompt text to send to the model
        temperature: Controls randomness (0 to 1)
        max_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
        stop_sequences: Optional list of sequences to stop generation
        
    Returns:
        Generated text response
    """
    if not SAMBANOVA_API_KEY:
        logger.warning("SAMBANOVA_API_KEY not set! Using mock response.")
        return _get_mock_response(prompt)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}"
    }
    
    # Prepare the payload
    payload = {
        "model": "Meta-Llama-3.1-8B-Instruct",
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    
    if stop_sequences:
        payload["stop_sequences"] = stop_sequences
        
    try:
        logger.info("Sending request to SambaNova API")
        logger.info(f"URL: {SAMBANOVA_API_URL}")
        logger.info(f"Headers: {headers}")
        logger.info(f"Payload: {json.dumps(payload)}")
        
        response = requests.post(SAMBANOVA_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            # Print the raw response for debugging
            logger.info(f"Raw response: {response.text}")
            
            # Try to parse the response JSON
            try:
                data = response.json()
                logger.info(f"Response JSON structure: {json.dumps(data)}")
                
                # Check different possible response formats
                if "response" in data:
                    return data["response"]
                elif "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0].get("text", "")
                elif "generations" in data and len(data["generations"]) > 0:
                    return data["generations"][0].get("text", "")
                elif "completion" in data:
                    return data["completion"]
                elif "output" in data:
                    return data["output"]
                else:
                    # Return the entire response as a string if we can't find a specific field
                    logger.warning("Could not find expected response field, returning full response")
                    return str(data)
            except Exception as e:
                logger.error(f"Error parsing response JSON: {str(e)}")
                return f"Error processing API response: {response.text}"
        else:
            logger.error(f"API error: {response.status_code} - {response.text}")
            return f"Error: Failed to get response from API (Status code: {response.status_code})"
            
    except Exception as e:
        logger.error(f"Exception when calling SambaNova API: {str(e)}")
        return _get_mock_response(prompt)

def _get_mock_response(prompt: str) -> str:
    """
    Generate a mock response when the API is not available.
    
    Args:
        prompt: The prompt text
        
    Returns:
        Mock response
    """
    logger.info("Generating mock response")
    
    # Extract question from prompt
    question_start = prompt.find("USER QUESTION:")
    if question_start != -1:
        question = prompt[question_start:].split("\n")[1].strip()
    else:
        question = "your question"
    
    return f"""Based on the document provided, I found some information related to {question}. 
    
The document appears to contain contractual information, possibly related to a business agreement or legal document. However, without specific details from the document context, I can only provide a general response.

If you need specific information from the document, please ensure the document has been properly processed and that your question is specific to the content within it.

(Note: This is a simulated response as the SambaNova API is not configured. Please set the SAMBANOVA_API_KEY environment variable to enable actual API calls.)""" 