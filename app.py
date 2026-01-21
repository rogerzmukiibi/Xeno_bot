"""
XENO Bot - AI-powered customer service assistant
Main application file with Gradio interface
"""

import os

from src.config import (COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL_NAME,
                        SERVER_NAME, SERVER_PORT)
from src.intent_classifier import IntentClassifier
from src.interface import create_interface
from src.knowledge_base import get_knowledge_base_data
# Import custom modules
from src.utils import PipelineTimer
from src.vector_store import initialize_vector_store

# Initialize components
timer = PipelineTimer()

# === Configuration ===
# Ensure API Key is set
if "GEMINI_API_KEY" not in os.environ:
    print("WARNING: GEMINI_API_KEY environment variable not found.")

# Initialize the client
embedding_model = EMBEDDING_MODEL
llm_model_name = LLM_MODEL_NAME
collection_name = COLLECTION_NAME

# === Intent Classification System ===
intent_classifier = IntentClassifier()

# === Load and Clean Knowledge Base ===
documents, metadatas, ids = get_knowledge_base_data()

# === Setup ChromaDB ===
collection, vector_store, retriever = initialize_vector_store()

# === Main Interface Logic ===

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        share=False, server_name=SERVER_NAME, server_port=SERVER_PORT, ssr_mode=False
    )
