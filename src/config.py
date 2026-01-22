"""
Configuration module for XENO Bot
Handles environment variables and application settings
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# ==================== VALIDATION & LOGGING ====================
def log_status(item: str, status: bool, message: str = ""):
    """Helper for consistent status logging"""
    icon = "✅" if status else "❌"
    print(f"{icon} {item}: {message}")

print("\n" + "="*50)
print("XENO BOT - Configuration Loading")
print("="*50)

# ==================== GEMINI API CONFIGURATION ====================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        log_status("Gemini API", True, "Configured successfully")
    except Exception as e:
        log_status("Gemini API", False, f"Configuration failed: {e}")
else:
    log_status("Gemini API", False, "GEMINI_API_KEY not found in .env")

# ==================== NEO4J CONFIGURATION ====================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# Validate Neo4j credentials
NEO4J_CONFIGURED = all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD])
if NEO4J_CONFIGURED:
    log_status("Neo4j Database", True, f"URI: {NEO4J_URI}, User: {NEO4J_USER}")
else:
    missing = []
    if not NEO4J_URI: missing.append("NEO4J_URI")
    if not NEO4J_USER: missing.append("NEO4J_USER") 
    if not NEO4J_PASSWORD: missing.append("NEO4J_PASSWORD")
    log_status("Neo4j Database", False, f"Missing in .env: {', '.join(missing)}")

# ==================== GOOGLE SHEETS CONFIGURATION ====================
GOOGLE_SHEETS_CREDENTIALS_PATH = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_PATH")
GOOGLE_SHEETS_CREDENTIALS = None

if GOOGLE_SHEETS_CREDENTIALS_PATH:
    if os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
        try:
            with open(GOOGLE_SHEETS_CREDENTIALS_PATH, 'r') as f:
                GOOGLE_SHEETS_CREDENTIALS = json.load(f)
            log_status("Google Sheets", True, f"Credentials loaded from: {GOOGLE_SHEETS_CREDENTIALS_PATH}")
        except json.JSONDecodeError as e:
            log_status("Google Sheets", False, f"Invalid JSON format: {e}")
        except Exception as e:
            log_status("Google Sheets", False, f"Error loading file: {e}")
    else:
        log_status("Google Sheets", False, f"File not found: {GOOGLE_SHEETS_CREDENTIALS_PATH}")
else:
    log_status("Google Sheets", False, "GOOGLE_SHEETS_CREDENTIALS_PATH not in .env")

# ==================== MODEL CONFIGURATION ====================
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL_NAME = "models/gemma-3-4b-it"
log_status("AI Models", True, f"LLM: {LLM_MODEL_NAME}, Embedding: {EMBEDDING_MODEL}")

# ==================== DATABASE CONFIGURATION ====================
COLLECTION_NAME = "xeno_collection"
CHROMA_DB_PATH = "/tmp/xeno_db"
SQLITE_DB_PATH = "xeno_memory.db"
log_status("Local Databases", True, f"Chroma: {CHROMA_DB_PATH}, SQLite: {SQLITE_DB_PATH}")

# ==================== KNOWLEDGE BASE CONFIGURATION ====================
KNOWLEDGE_BASE_PATH = "D:\XENO_DOCUMENTS\XENO_CHATBOT\Xeno_bot\XENO_Uganda_KnowledgeBase_Advisory.json"
if os.path.exists(KNOWLEDGE_BASE_PATH):
    log_status("Knowledge Base", True, f"File found: {KNOWLEDGE_BASE_PATH}")
else:
    log_status("Knowledge Base", False, f"File not found: {KNOWLEDGE_BASE_PATH}")

# ==================== GOOGLE SHEETS SETTINGS ====================
SPREADSHEET_NAME = "Response_Log"
RESPONSE_SHEET_INDEX = 0  # sheet1
TIMING_SHEET_NAME = "Timing_Log"

# ==================== RAG CONFIGURATION ====================
RAG_TOP_K = 4
RAG_MAX_RESULTS = 2
SIMILARITY_THRESHOLD = 0.4

# ==================== SERVER CONFIGURATION ====================
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7860
log_status("Server", True, f"Host: {SERVER_NAME}:{SERVER_PORT}")

# ==================== PROMPT CONFIGURATION ====================
SYSTEM_PROMPT = """You are a friendly XENO Support Assistant, an AI-powered helpful and professional customer service representative.
Use only the information provided in the knowledge base context to answer user queries.
Do not hallucinate. If context doesn't contain relevant info, say so in a calm polite manner by saying I'm sorry, I can't assist with that.
Only use context that is clearly relevant to the user's question.
For greetings like "hi" or "hello", respond politely without using the context.
Remember previous conversations."""

log_status("System Prompt", True, f"Loaded ({len(SYSTEM_PROMPT)} chars)")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*50)
print("CONFIGURATION SUMMARY")
print("="*50)

config_status = {
    "Gemini API": bool(GEMINI_API_KEY),
    "Neo4j Database": NEO4J_CONFIGURED,
    "Google Sheets": bool(GOOGLE_SHEETS_CREDENTIALS),
    "Knowledge Base": os.path.exists(KNOWLEDGE_BASE_PATH),
}

all_configured = all(config_status.values())

for service, configured in config_status.items():
    status = "✅ READY" if configured else "❌ MISSING"
    print(f"{service:20} {status}")

print("="*50)
if all_configured:
    print("✅ ALL SYSTEMS READY - XENO Bot is fully configured!")
else:
    print("⚠️  SOME SERVICES MISSING - Check .env file and paths above")
print("="*50 + "\n")

# Export all configurations
__all__ = [
    # API Keys
    "GEMINI_API_KEY",
    
    # Neo4j
    "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_CONFIGURED",
    
    # Google Sheets
    "GOOGLE_SHEETS_CREDENTIALS", "GOOGLE_SHEETS_CREDENTIALS_PATH",
    "SPREADSHEET_NAME", "RESPONSE_SHEET_INDEX", "TIMING_SHEET_NAME",
    
    # Models
    "EMBEDDING_MODEL", "LLM_MODEL_NAME",
    
    # Databases
    "COLLECTION_NAME", "CHROMA_DB_PATH", "SQLITE_DB_PATH",
    
    # Knowledge Base
    "KNOWLEDGE_BASE_PATH",
    
    # RAG
    "RAG_TOP_K", "RAG_MAX_RESULTS", "SIMILARITY_THRESHOLD",
    
    # Server
    "SERVER_NAME", "SERVER_PORT",
    
    # Prompts
    "SYSTEM_PROMPT",
]