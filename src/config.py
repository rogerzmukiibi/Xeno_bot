"""
Configuration module for XENO Bot
Handles environment variables and application settings
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# === API Configuration ===
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

genai.configure(api_key=GEMINI_API_KEY)

# === Model Configuration ===
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL_NAME = "models/gemma-3-4b-it"

# === Database Configuration ===
COLLECTION_NAME = "xeno_collection"
CHROMA_DB_PATH = "/tmp/xeno_db"
SQLITE_DB_PATH = "xeno_memory.db"

# === Knowledge Base Configuration ===
KNOWLEDGE_BASE_PATH = "XENO_Uganda_KnowledgeBase_Advisory.json"

# === Google Sheets Configuration ===
GOOGLE_SHEETS_CREDENTIALS_ENV = "GOOGLE_SHEETS_CREDENTIALS"
SPREADSHEET_NAME = "Response_Log"
RESPONSE_SHEET_INDEX = 0  # sheet1
TIMING_SHEET_NAME = "Timing_Log"

# === RAG Configuration ===
RAG_TOP_K = 10
RAG_MAX_RESULTS = 5
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
