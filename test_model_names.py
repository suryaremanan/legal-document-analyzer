import requests
import json

API_URL = "https://api.sambanova.ai/llm/generate"
API_KEY = "16c40030-fb45-41ee-9cea-ae79da50778c"

def test_model_name(model_name):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model_name,
        "prompt": "Hello world",
        "temperature": 0.0,
        "max_tokens": 100,
    }
    
    print(f"Testing model name: {model_name}")
    response = requests.post(API_URL, headers=headers, json=payload)
    
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}\n")

# Test various model name formats
model_names = [
    "Llama-3.1-8B-Instruct",
    "llama3-8b-instruct",
    "llama3",
    "meta-llama/Llama-3.1-8B-Instruct"
]

for model_name in model_names:
    test_model_name(model_name) 