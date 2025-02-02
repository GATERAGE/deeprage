# src/models.py

import requests
import logging

logger = logging.getLogger('rage.models')

class OllamaHandler:
    """Handler for Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def generate_response(self, prompt: str, context: Optional[str] = None, temperature: float = 0.11) -> str:
        """Generate response using Ollama"""
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:" if context else prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "default",  # Replace with actual model name
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": temperature
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                logger.error(f"API Error: {response.text}")
                return "Error generating response"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"

class GroqHandler:
    """Handler for Groq models"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def generate_response(self, prompt: str, context: Optional[str] = None, temperature: float = 0.11) -> str:
        """Generate response using Groq"""
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:" if context else prompt
        
        try:
            response = requests.post(
                "https://api.groq.com/v1/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": temperature
                }
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"API Error: {response.text}")
                return "Error generating response"
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Error generating response"
