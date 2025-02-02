# src/memory.py (c) 2025 Gregory L. Magnusson MIT

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging
import faiss
import numpy as np

logger = logging.getLogger('rage.memory')

class MemoryManager:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger('rage.memory')
        
        # Define memory structure
        self.base_dir = Path('./data')
        self.ltm_dir = self.base_dir / 'knowledge'  # Long-term memory
        self.stm_dir = self.base_dir / 'conversations'  # Short-term memory
        self.conversation_file = self.stm_dir / 'conversation.json'
        
        # Initialize system
        self._initialize_memory_system()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self._load_existing_index()

    def _initialize_memory_system(self):
        """Initialize memory system and create directories"""
        try:
            self.ltm_dir.mkdir(parents=True, exist_ok=True)
            self.stm_dir.mkdir(parents=True, exist_ok=True)
            
            if not self.conversation_file.exists():
                self._write_json(self.conversation_file, {"entries": []})
            
            self.logger.info("Memory system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system: {e}")
            raise

    def _write_json(self, filepath: Path, data: Dict) -> bool:
        """Write data to JSON file with error handling"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {filepath}: {e}")
            return False

    def _read_json(self, filepath: Path) -> Optional[Dict]:
        """Read JSON file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {filepath}: {e}")
            return None

    def _load_existing_index(self):
        """Load existing index and documents if available"""
        try:
            index_path = self.ltm_dir / "faiss_index.bin"
            docs_path = self.ltm_dir / "documents.json"
            
            if index_path.exists() and docs_path.exists():
                self.index = faiss.read_index(str(index_path))
                docs_data = self._read_json(docs_path)
                if docs_data:
                    self.documents = [{"content": doc["content"], "embedding": doc["embedding"]} for doc in docs_data]
                self.logger.info(f"Loaded {len(self.documents)} documents from existing index")
        except Exception as e:
            self.logger.error(f"Error loading existing index: {e}")
            self.index = None
            self.documents = []

    def store_conversation(self, query: str, response: str) -> bool:
        """Store conversation entry in STM"""
        try:
            entry = {
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            conversation_data = self._read_json(self.conversation_file) or {"entries": []}
            conversation_data["entries"].append(entry)
            return self._write_json(self.conversation_file, conversation_data)
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            return False

    def store_document(self, content: str, embedding: List[float]) -> bool:
        """Store document in LTM"""
        try:
            if self.index is None:
                dim = len(embedding)
                self.index = faiss.IndexFlatL2(dim)
            
            self.index.add(np.array([embedding]))
            self.documents.append({"content": content, "embedding": embedding})
            
            # Save updated index and documents
            index_path = self.ltm_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))
            
            docs_path = self.ltm_dir / "documents.json"
            self._write_json(docs_path, self.documents)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to store document: {e}")
            return False

    def retrieve_context(self, query_embedding: List[float], k: int = 3) -> List[str]:
        """Retrieve relevant documents based on query embedding"""
        if not self.index or not self.documents:
            return []
        
        D, I = self.index.search(np.array([query_embedding]), k)
        results = [self.documents[i]["content"] for i in I[0] if i < len(self.documents)]
        return results
