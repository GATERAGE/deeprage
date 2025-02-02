# rage.py (c) 2025 Gregory L. Magnusson

import streamlit as st
from pathlib import Path
from src.models import OllamaHandler, GroqHandler
from src.memory import MemoryManager
from src.openmind import OpenMind
import logging

logger = logging.getLogger('rage')

class RAGE_UI:
    """RAGE - Retrieval Augmented Generative Engine with Streamlit UI"""
    
    def __init__(self):
        self.setup_session_state()
        self.config = {}
        self.model_config = {}
        self.load_css()
        self.load_system_prompt()
        
        # Initialize systems
        self.memory = MemoryManager()
        self.openmind = OpenMind()

    def setup_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if 'provider' not in st.session_state:
            st.session_state.provider = None
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'temperature' not in st.session_state:
            st.session_state.temperature = 0.11  # Default temperature setting

    def load_css(self):
        """Load CSS styling"""
        try:
            with open('styles.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error loading CSS: {e}")

    def load_system_prompt(self):
        """Load system prompt from prompt.txt"""
        try:
            with open('prompt.txt', 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        except Exception as e:
            logger.error(f"Error loading system prompt: {e}")
            self.system_prompt = "Default system prompt."

    def initialize_model(self, provider: str):
        """Initialize or retrieve model instance"""
        try:
            if not provider:
                st.info("Please select an AI Provider")
                return None
            
            if provider == "Groq":
                key = self.openmind.get_api_key('groq')
                if key:
                    return GroqHandler(key)
                else:
                    st.error("Groq API key not found")
                    return None
            
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
            
            # Add message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Processing with RAGE..."):
                    try:
                        # Get relevant context
                        context = self.memory.retrieve_context(query_embedding=[0.1] * 384)  # Replace with actual embedding
                        
                        # Generate response
                        response = model.generate_response(
                            prompt=prompt,
                            context=context,
                            temperature=st.session_state.temperature
                        )
                        
                        # Store conversation
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
