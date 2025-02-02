# src/openmind.py (c) 2025 RAGE

import os
from dotenv import load_dotenv

class OpenMind:
    """Central configuration and resource management for RAGE"""
    
    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for the specified service"""
        if service.lower() == "openai":
            return self.openai_api_key
        elif service.lower() == "groq":
            return self.groq_api_key
        return None
