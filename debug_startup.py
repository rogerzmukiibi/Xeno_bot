
import os
import sys
import traceback
from dotenv import load_dotenv

print("Loading .env...")
load_dotenv()

print("Checking API Key...")
if "GEMINI_API_KEY" not in os.environ:
    print("GEMINI_API_KEY missing!")
else:
    print("GEMINI_API_KEY found.")

print("Checking Google Sheets Creds...")
if "GOOGLE_SHEETS_CREDENTIALS" not in os.environ:
    print("GOOGLE_SHEETS_CREDENTIALS missing!")
else:
    print("GOOGLE_SHEETS_CREDENTIALS found.")

print("Importing modules...")
try:
    import google.generativeai as genai
    print("Imported genai")
    import chromadb
    print("Imported chromadb")
    from src.config import COLLECTION_NAME
    print("Imported config")
    from src.vector_store import initialize_vector_store
    print("Imported vector_store")
except Exception as e:
    print(f"Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Initializing Vector Store...")
try:
    initialize_vector_store()
    print("Vector Store Initialized.")
except Exception as e:
    print(f"Vector Store Init Failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Startup Check Complete.")
