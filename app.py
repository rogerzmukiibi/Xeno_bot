import uuid
import os
import gradio as gr
import pandas as pd
import torch
import numpy as np
from sentence_transformers import util
import google.generativeai as genai
import chromadb
from langchain_chroma import Chroma
import gspread
from google.oauth2.service_account import Credentials
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import json
from datetime import datetime
import re
from typing import Dict, List, Tuple
import time
from contextlib import contextmanager

import logging
import traceback
import sys

# ===== Configure Logging =====
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        return
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = log_exception
logging.info("App started successfully.")

# ===== Time Tracking Class =====
class PipelineTimer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timing data for a new request"""
        self.start_time = time.time()
        self.step_times = {}
        self.step_start = None
        self.current_step = None
    
    @contextmanager
    def time_step(self, step_name: str):
        """Context manager to time a specific step"""
        step_start = time.time()
        self.current_step = step_name
        try:
            yield
        finally:
            step_end = time.time()
            self.step_times[step_name] = round((step_end - step_start) * 1000, 2)  # Convert to milliseconds
            self.current_step = None
    
    def get_total_time(self):
        """Get total elapsed time since reset"""
        return round((time.time() - self.start_time) * 1000, 2)
    
    def get_timing_summary(self):
        """Get a summary of all timing data"""
        total_time = self.get_total_time()
        return {
            'total_time_ms': total_time,
            'step_times': self.step_times,
            'timestamp': datetime.now().isoformat()
        }

# Initialize global timer
timer = PipelineTimer()

# === Configuration ===
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
embedding_model = "models/embedding-001"
llm_model_name = "models/gemma-3-4b-it"
collection_name = "xeno_collection"

# === Google Sheets Setup ===
def get_google_sheets_credentials():
    credentials_json = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if not credentials_json:
        raise ValueError("GOOGLE_SHEETS_CREDENTIALS environment variable not set.")
    credentials_dict = json.loads(credentials_json)
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)
    return creds

client_gspread = gspread.authorize(get_google_sheets_credentials())

# Open the Google Sheet and get both sheets
spreadsheet = client_gspread.open("Response_Log")
response_sheet = spreadsheet.sheet1  # Main response log
try:
    timing_sheet = spreadsheet.worksheet("Timing_Log")
except:
    # Create timing sheet if it doesn't exist
    timing_sheet = spreadsheet.add_worksheet(title="Timing_Log", rows="1000", cols="15")
    # Add headers
    headers = [
        "Timestamp", "Session_ID", "Question", "Total_Time_MS",
        "Intent_Classification_MS", "Memory_Retrieval_MS", "RAG_Retrieval_MS", 
        "Embedding_Generation_MS", "Similarity_Calculation_MS", "Context_Processing_MS",
        "LLM_Generation_MS", "Memory_Update_MS", "Logging_MS", "Error_Step", "Notes"
    ]
    timing_sheet.append_row(headers)

def log_response(question, answer, source_ids, knowledge_pairs, session_id):
    """Original response logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    knowledge_question_1 = knowledge_pairs[0][0] if len(knowledge_pairs) > 0 else "N/A"
    knowledge_answer_1 = knowledge_pairs[0][1] if len(knowledge_pairs) > 0 else "N/A"
    knowledge_question_2 = knowledge_pairs[1][0] if len(knowledge_pairs) > 1 else "N/A"
    knowledge_answer_2 = knowledge_pairs[1][1] if len(knowledge_pairs) > 1 else "N/A"
    row = [
        timestamp, session_id, question, answer, source_ids,
        knowledge_question_1, knowledge_answer_1, knowledge_question_2, knowledge_answer_2
    ]
    try:
        response_sheet.append_row(row)
        print(f"Logged response: {question} | Source IDs: {source_ids}")
    except Exception as e:
        print(f"Failed to log to Google Sheet: {e}")
        with open("/tmp/response_log.txt", "a") as f:
            f.write(f"{timestamp},{question},{answer},{source_ids},{knowledge_question_1},{knowledge_answer_1},{knowledge_question_2},{knowledge_answer_2}\n")

def log_timing_data(question, session_id, timing_summary, error_step=None, notes=None):
    """Log timing data to the timing sheet"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_times = timing_summary['step_times']
    
    row = [
        timestamp,
        session_id,
        question[:100] + "..." if len(question) > 100 else question,  # Truncate long questions
        timing_summary['total_time_ms'],
        step_times.get('intent_classification', 0),
        step_times.get('memory_retrieval', 0),
        step_times.get('rag_retrieval', 0),
        step_times.get('embedding_generation', 0),
        step_times.get('similarity_calculation', 0),
        step_times.get('context_processing', 0),
        step_times.get('llm_generation', 0),
        step_times.get('memory_update', 0),
        step_times.get('response_logging', 0),
        error_step or "",
        notes or ""
    ]
    
    try:
        timing_sheet.append_row(row)
        print(f"Logged timing data: Total {timing_summary['total_time_ms']}ms")
    except Exception as e:
        print(f"Failed to log timing data: {e}")
        # Fallback to local file
        with open("/tmp/timing_log.txt", "a") as f:
            f.write(f"{timestamp},{session_id},{question},{timing_summary}\n")

# === LangGraph Memory Setup ===
conn = sqlite3.connect("xeno_memory.db", check_same_thread=False)
memory = SqliteSaver(conn=conn)

def update_memory(config, user_message, assistant_message):
    """Update memory with timing"""
    with timer.time_step("memory_update"):
        full_checkpoint = memory.get(config) or {}
        messages = full_checkpoint.get("channel_values", {}).get("messages", [])
        
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
        
        checkpoint_to_save = {
            "v": 1,
            "id": str(uuid.uuid4()),
            "ts": datetime.now().isoformat(),
            "channel_values": {"messages": messages},
            "channel_versions": {},
            "versions_seen": {},   
        }
        
        memory.put(config, checkpoint_to_save, {}, {})

def retrieve_memory(config):
    """Retrieve memory with timing"""
    with timer.time_step("memory_retrieval"):
        full_checkpoint = memory.get(config) or {}
        return full_checkpoint.get("channel_values", {}).get("messages", [])

# === Intent Classification System ===
class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
                    r'^(hi|hello|hey)[\s!.]*$',
                    r'\b(how are you|how do you do)\b'
                ],
                'responses': [
                    "Hello! I'm XENO Assistant. How can I help you with XENO financial services today?",
                    "Hi there! I'm here to assist you with any questions about XENO services. What can I help you with?",
                    "Good day! Welcome to XENO Support. How may I assist you today?"
                ]
            },
            'thanks': {
                'patterns': [
                    r'\b(thank you|thanks|thank u|thx|appreciate|grateful)\b',
                    r'^(thanks|thank you)[\s!.]*$',
                    r'\b(much appreciated|thanks a lot|thank you so much)\b'
                ],
                'responses': [
                    "You're welcome! Is there anything else I can help you with regarding XENO services?",
                    "Happy to help! Feel free to ask if you have any other questions about XENO.",
                    "Glad I could assist you! Let me know if you need help with anything else."
                ]
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|farewell|take care|have a good day)\b',
                    r'^(bye|goodbye)[\s!.]*$',
                    r'\b(talk to you later|see you later|until next time)\b'
                ],
                'responses': [
                    "Goodbye! Thank you for using XENO services. Have a great day!",
                    "Take care! Feel free to return anytime you need help with XENO services.",
                    "Have a wonderful day! Don't hesitate to reach out if you need assistance with XENO."
                ]
            }
        }
    
    def classify_intent(self, message: str) -> Tuple[str, str]:
        """Classify intent with timing"""
        message_lower = message.lower().strip()
        
        for intent_name, intent_data in self.intent_patterns.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    import random
                    response = random.choice(intent_data['responses'])
                    return intent_name, response
        
        return 'query', ''
    
    def is_simple_intent(self, intent: str) -> bool:
        simple_intents = ['greeting', 'thanks']
        return intent in simple_intents

intent_classifier = IntentClassifier()

# === Load and Clean Knowledge Base ===
df_kb = pd.read_json("XENO_Uganda_KnowledgeBase_Advisory.json")
df_kb.dropna(subset=['Content'], inplace=True)

def prepare_documents(data):
    documents, metadatas, ids = [], [], []
    for item in data:
        documents.append(f"Question: {item['Question']}\nAnswer: {item['Content']}")
        metadatas.append({
            "question": item["Question"],
            "content": item["Content"],
            "section": item.get("Section", ""),
            "source": item.get("Source", ""),
            "owner": item.get("Owner", ""),
            "tag": item.get("Tag", ""),
            "id": item["ID"]
        })
        ids.append(item["ID"])
    return documents, metadatas, ids

xeno_data_list = df_kb.to_dict('records')
documents, metadatas, ids = prepare_documents(xeno_data_list)

# === Setup ChromaDB ===
try:
    client = chromadb.PersistentClient(path="/tmp/xeno_db")
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Loaded existing ChromaDB collection: {collection_name}")
    except:
        print(f"Creating new ChromaDB collection: {collection_name}")
        collection = client.create_collection(name=collection_name)
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
except Exception as e:
    print(f"Failed to initialize ChromaDB: {e}")
    raise

vector_store = Chroma(client=client, collection_name=collection_name)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# === Prompt System ===
SYSTEM_PROMPT = """You are a friendly XENO Support Assistant, an AI-powered helpful and professional customer service representative.
Use only the information provided in the knowledge base context to answer user queries.
Do not hallucinate. If context doesn't contain relevant info, say so in a calm polite manner by saying I'm sorry, I can't assist with that.
Only use context that is clearly relevant to the user's question.
For greetings like "hi" or "hello", respond politely without using the context.
remember previous conversations."""

# === Context Processing ===
def process_context(results, cosine_scores, max_results=2):
    """Process context with timing"""
    with timer.time_step("context_processing"):
        sorted_indices = np.argsort(cosine_scores)[::-1][:max_results]
        formatted_context = ""
        source_ids = []
        knowledge_pairs = []
        for i, idx in enumerate(sorted_indices, 1):
            result = results[idx]
            score = cosine_scores[idx]
            question = result.metadata.get('question', 'N/A')
            answer = result.metadata.get('content', 'N/A')
            formatted_context += f"Knowledge Entry {i}:\n"
            formatted_context += f"Q: {question}\n"
            formatted_context += f"A: {answer}\n"
            formatted_context += "-" * 40 + "\n"
            source_ids.append(result.metadata.get('id', 'N/A'))
            knowledge_pairs.append((question, answer))
        return formatted_context, source_ids, knowledge_pairs

# === LLM Generation ===
def generate_xeno_response(context, question, chat_history):
    """Generate response with timing"""
    with timer.time_step("llm_generation"):
        model = genai.GenerativeModel(llm_model_name)
        formatted_history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        ) if chat_history else "None"
        
        prompt = f"{SYSTEM_PROMPT}\n### HISTORY ###\n{formatted_history}\n### CONTEXT ###\n{context}\n### QUESTION ###\n{question}"
        
        response = model.generate_content(prompt)
        return response.text.strip()

# === Main Interface Logic ===
def get_context_and_answer(message, history, session_id="default"):
    """Main pipeline with comprehensive timing"""
    # Reset timer for new request
    timer.reset()
    error_step = None
    notes = []
    
    try:
        config = {"configurable": {"thread_id": str(session_id), "checkpoint_ns": ""}}
        
        # Step 1: Intent Classification
        with timer.time_step("intent_classification"):
            intent, direct_response = intent_classifier.classify_intent(message)
        
        # Step 2: Memory Retrieval
        chat_history = retrieve_memory(config)
        
        answer = ""
        source_ids = "N/A"
        knowledge_pairs = []

        if intent != 'query':
            answer = direct_response
            notes.append(f"Simple intent: {intent}")
        else: 
            if len(message.strip()) < 3:
                answer = "I'd be happy to help! Could you please provide more details about what you'd like to know?"
                notes.append("Message too short")
            else:
                try:
                    # Step 3: RAG Retrieval
                    with timer.time_step("rag_retrieval"):
                        queried_results = retriever.invoke(message)
                    
                    # Step 4: Embedding Generation
                    with timer.time_step("embedding_generation"):
                        query_embedding = genai.embed_content(
                            model=embedding_model, 
                            content=message, 
                            task_type="retrieval_query"
                        )['embedding']
                        
                        doc_embeddings = [
                            genai.embed_content(
                                model=embedding_model, 
                                content=doc.page_content, 
                                task_type="retrieval_document"
                            )['embedding'] 
                            for doc in queried_results
                        ]
                    
                    # Step 5: Similarity Calculation
                    with timer.time_step("similarity_calculation"):
                        cosine_scores = util.cos_sim(
                            torch.tensor(query_embedding).float(), 
                            torch.tensor(doc_embeddings).float()
                        )[0].tolist()
                        max_score = max(cosine_scores)

                    if max_score < 0.4:
                        answer = "I'm sorry, I couldn't find specific information for your question. Could you try rephrasing it, or contact XENO support directly?"
                        notes.append(f"Low similarity score: {max_score:.3f}")
                    else:
                        # Step 6: Context Processing (timed within function)
                        context, source_ids_list, knowledge_pairs = process_context(queried_results, cosine_scores)
                        
                        # Step 7: LLM Generation (timed within function)
                        answer = generate_xeno_response(context, message, chat_history)
                        source_ids = ", ".join(source_ids_list)
                        notes.append(f"Max similarity: {max_score:.3f}")

                except Exception as e:
                    error_step = timer.current_step or "rag_processing"
                    print(f"Error during RAG processing: {e}")
                    answer = "I apologize, but I'm having a technical issue. Please try again shortly or contact XENO support."
                    notes.append(f"Error: {str(e)}")

        # Step 8: Memory Update (timed within function)
        update_memory(config, message, answer)
        
        # Step 9: Response Logging
        with timer.time_step("response_logging"):
            log_response(message, answer, source_ids, knowledge_pairs, session_id)
        
        # Log timing data
        timing_summary = timer.get_timing_summary()
        log_timing_data(
            message, 
            session_id, 
            timing_summary, 
            error_step=error_step,
            notes="; ".join(notes) if notes else None
        )
        
        return answer
        
    except Exception as e:
        error_step = timer.current_step or "main_pipeline"
        logging.error(f"Error in main pipeline: {e}")
        logging.error(traceback.format_exc())
        
        # Still log timing data even on error
        timing_summary = timer.get_timing_summary()
        log_timing_data(
            message, 
            session_id, 
            timing_summary, 
            error_step=error_step,
            notes=f"Pipeline error: {str(e)}"
        )
        
        return "I apologize, but I encountered an error processing your request. Please try again."

# === Enhanced Gradio UI ===
def respond(message, history, session_id):
    """Gradio's main response function"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    bot_response = get_context_and_answer(message, history, session_id)
    history.append([message, bot_response])
    
    return "", history

def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # ASKXENO
        **Welcome to XENO AI Support!**
        
        I can help you with questions about XENO financial services including:
        - Account management and setup
        - Transaction processes and fees
        - Platform features and troubleshooting
        - General service information
        
        *Simply type your question below to get started!*
        """)
        
        session_id_box = gr.Textbox(label="Session ID", value=str(uuid.uuid4()), interactive=True)
        
        chatbot = gr.Chatbot(
            label="XENO Assistant",
            bubble_full_width=False,
            height=500
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your question here...",
                scale=3,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        send_button.click(respond, [msg, chatbot, session_id_box], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, session_id_box], [msg, chatbot])
            
    return demo

if __name__ == "__main__":
    iface = create_interface()
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860, ssr_mode=False)