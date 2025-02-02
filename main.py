# main.py (c) 2025 Gregory L. Magnusson
# Retrieval Augmented Generative Engine CLI
"""
RAGE: Retrieval Augmented Generative Engine (CLI Mode)
(c) 2025 Gregory L. Magnusson MIT
"""

import argparse
import logging
from src.memory import MemoryManager
from src.models import OllamaHandler, GroqHandler
from src.openmind import OpenMind

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/rage.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("rage")

def run_query(model_name: str, query: str):
    """Run a query through the selected AI model."""
    openmind = OpenMind()
    
    if model_name.lower() == "groq":
        api_key = openmind.get_api_key("groq")
        if not api_key:
            print("Groq API key is missing. Check your .env file.")
            return
        model = GroqHandler(api_key)
    elif model_name.lower() == "ollama":
        model = OllamaHandler()
    else:
        print("Unsupported model. Choose 'groq' or 'ollama'.")
        return
    
    memory = MemoryManager()
    context = memory.retrieve_context(query_embedding=[0.1] * 384)  # Replace with real embeddings
    response = model.generate_response(query, context)
    
    print("\nAI Response:\n", response)
    
    memory.store_conversation(query, response)

def main():
    """CLI entry point for RAGE"""
    parser = argparse.ArgumentParser(description="Run RAGE from the command line")
    parser.add_argument("model", type=str, help="Model to use (groq or ollama)")
    parser.add_argument("query", type=str, help="User query for AI processing")
    
    args = parser.parse_args()
    run_query(args.model, args.query)

if __name__ == "__main__":
    main()
