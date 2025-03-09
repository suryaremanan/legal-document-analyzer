import os
import requests
import json

# Hardcode the API credentials directly for testing
API_URL = "https://api.sambanova.net/llm/v1/generate"
API_KEY = "16c40030-fb45-41ee-9cea-ae79da50778c"

def test_api():
    """Test the SambaNova API with a simple prompt"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Say hello world",
        "temperature": 0.0,
        "max_tokens": 100,
    }
    
    try:
        print("Sending request to SambaNova API...")
        response = requests.post(API_URL, headers=headers, json=payload)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("API Response:")
            print(data.get("response", ""))
            return True
        else:
            print(f"API error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Exception: {str(e)}")
        return False

if __name__ == "__main__":
    test_api() 