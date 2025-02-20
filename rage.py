# rage.py
"""
RAGE: Retrieval Augmented Generative Engine (Streamlit UI)
(c) 2025 Gregory L. Magnusson MIT
"""

import streamlit as st
from pathlib import Path
from src.models import OllamaHandler, GroqHandler
from src.memory import MemoryManager
from src.openmind import OpenMind
import logging

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

class RAGE_UI:
    """RAGE - Retrieval Augmented Generative Engine with Streamlit UI"""
    
    def __init__(self):
        self.setup_session_state()
        self.load_css()
        self.load_system_prompt()
        
        # Initialize core components
        self.memory = MemoryManager()
        self.openmind = OpenMind()

    def setup_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "provider" not in st.session_state:
            st.session_state.provider = None
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.11  # Default temperature

    def load_css(self):
        """Load external CSS for better UI"""
        try:
            with open("styles.css") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error loading CSS: {e}")

    def load_system_prompt(self):
        """Load system prompt from prompt.txt"""
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            self.system_prompt = "Default system prompt."

    def initialize_model(self, provider: str):
        """Initialize the AI model based on the selected provider"""
        try:
            if not provider:
                st.info("Please select an AI Provider")
                return None
            
            if provider == "Groq":
                key = self.openmind.get_api_key("groq")
                return GroqHandler(key) if key else None
            
            elif provider == "Ollama":
                return OllamaHandler()
            
            return None
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            st.error(f"Error initializing model: {str(e)}")
            return None

    def process_message(self, prompt: str):
        """Process user message and generate response"""
        try:
            if not st.session_state.provider:
                st.warning("Please select an AI Provider first")
                return
            
            model = self.initialize_model(st.session_state.provider)
            if not model:
                return
            
            # Add user message to session
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Processing with RAGE..."):
                    try:
                        # Retrieve relevant context
                        context = self.memory.retrieve_context(query_embedding=[0.1] * 384)  # Replace with real embeddings
                        
                        # Generate AI response
                        response = model.generate_response(
                            prompt=prompt,
                            context=context,
                            temperature=st.session_state.temperature
                        )
                        
                        # Store conversation in STM
                        self.memory.store_conversation(query=prompt, response=response)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Update session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        st.error(f"Error generating response: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            st.error("An error occurred while processing your message")

    def upload_files_to_context(self):
        """Upload files to the context folder for retrieval"""
        uploaded_files = st.file_uploader(
            "Upload files for context",
            type=["txt", "md", "json"],
            accept_multiple_files=True,
            key="file_upload"
        )
        
        if uploaded_files:
            context_dir = Path("./data/conversations/context")
            context_dir.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = context_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.success(f"Uploaded: {uploaded_file.name}")

    def setup_sidebar(self):
        """Setup sidebar configuration"""
        with st.sidebar:
            st.header("RAGE Configuration")
            st.session_state.provider = st.selectbox("Select AI Provider", ["Groq", "Ollama"])
            st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.11, 0.01)

    def run(self):
        """Main UI loop"""
        st.title("🔥 RAGE: Retrieval Augmented Generative Engine")
        
        self.setup_sidebar()
        
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        user_input = st.chat_input("Type your message...")
        if user_input:
            self.process_message(user_input)

if __name__ == "__main__":
    RAGE_UI().run()
