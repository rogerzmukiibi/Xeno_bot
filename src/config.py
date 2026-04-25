"""
Configuration module for XENO Bot
Handles environment variables and application settings
"""

import os

# === Model Configuration ===
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "google/gemma-2-2b-it")
HF_TOKEN = os.environ.get("HF_TOKEN")
LLM_TIMEOUT_SECONDS = int(os.environ.get("LLM_TIMEOUT_SECONDS", "90"))
LLM_MAX_NEW_TOKENS = int(os.environ.get("LLM_MAX_NEW_TOKENS", "256"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# === Database Configuration ===
COLLECTION_NAME = "xeno_collection"
CHROMA_DB_PATH = "/tmp/xeno_db"
SQLITE_DB_PATH = "xeno_memory.db"

# === Knowledge Base Configuration ===
KNOWLEDGE_BASE_PATH = "XENO_Uganda_KnowledgeBase_Advisory.json"

# === RAG Configuration ===
RAG_TOP_K = 4
RAG_MAX_RESULTS = 2
SIMILARITY_THRESHOLD = 0.4

# === Server Configuration ===
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860

# === Prompt Configuration ===
SYSTEM_PROMPT = """You are a friendly XENO Support Assistant, an AI-powered helpful and professional customer service representative.
Use only the information provided in the knowledge base context to answer user queries.
Do not hallucinate. If context doesn't contain relevant info, say so in a calm polite manner by saying I'm sorry, I can't assist with that.
Only use context that is clearly relevant to the user's question.
For greetings like "hi" or "hello", respond politely without using the context.
remember previous conversations."""
