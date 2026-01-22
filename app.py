"""
XENO Bot - AI-powered customer service assistant
Main application file with Gradio interface
"""
import os
import uuid
import gradio as gr
import pandas as pd
import torch
import numpy as np
from sentence_transformers import util
from google import genai
import chromadb
from langchain_chroma import Chroma
import gspread
from google.oauth2.service_account import Credentials
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import json
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional, Any
import time
from contextlib import contextmanager
import threading
import logging
import traceback
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import existing modules
from src.utils import PipelineTimer
from src.config import SIMILARITY_THRESHOLD, SERVER_NAME, SERVER_PORT
from src.intent_classifier import IntentClassifier
from src.vector_store import generate_embeddings, calculate_similarity
from src.logger import log_response, log_timing_data
from src.neo4j_client import Neo4jClient  # <-- Import Neo4j client

# ===== NEW: Import Cognitive System Components =====
from src.cognitive.cognitive_memory import CognitiveMemory
from src.cognitive.reasoning import CognitiveReasoningEngine
from src.cognitive.retrieval import CognitiveRetriever
from src.graph.schema import KnowledgeGraphSchema
from src.graph.knowledge_ingest import KnowledgeGraphIngestor
# ===================================================

# Initialize components
timer = PipelineTimer()

# === Configuration ===
# Ensure API Key is set
if "GEMINI_API_KEY" not in os.environ:
    print("WARNING: GEMINI_API_KEY environment variable not found.")

# Initialize the client
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
embedding_model = "models/embedding-001"
llm_model_name = "models/gemma-3-4b-it"
collection_name = "xeno_collection"

# === Initialize Neo4j Client ===
neo4j_client = None
try:
    neo4j_client = Neo4jClient()
    print("✅ Neo4j client initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Neo4j client: {e}")

# === Initialize Cognitive System ===
cognitive_memory = None
cognitive_reasoning = None
cognitive_retriever = None
knowledge_ingestor = None

try:
    # Initialize Cognitive Memory
    cognitive_memory = CognitiveMemory(neo4j_client=neo4j_client)
    print("✅ Cognitive Memory initialized")
    
    # Initialize Cognitive Reasoning Engine
    cognitive_reasoning = CognitiveReasoningEngine(
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        cognitive_memory=cognitive_memory
    )
    print("✅ Cognitive Reasoning Engine initialized")
    
    # Initialize Cognitive Retriever
    cognitive_retriever = CognitiveRetriever(
        vector_store=None,  # Will be set later
        cognitive_memory=cognitive_memory,
        reasoning_engine=cognitive_reasoning
    )
    print("✅ Cognitive Retriever initialized")
    
    # Initialize Knowledge Graph Ingestor
    knowledge_ingestor = KnowledgeGraphIngestor(neo4j_client=neo4j_client)
    print("✅ Knowledge Graph Ingestor initialized")
    
except Exception as e:
    print(f"⚠️ Warning: Could not initialize cognitive system components: {e}")
    traceback.print_exc()

# === Google Sheets Setup ===
def get_google_sheets_credentials():
    """Get Google Sheets credentials from file path or environment variable"""
    # First try: Direct JSON in environment variable
    google_sheets_creds = os.environ.get("GOOGLE_SHEETS_CREDENTIALS")
    if google_sheets_creds:
        try:
            credentials_dict = json.loads(google_sheets_creds)
            print("✅ Google Sheets credentials loaded from environment variable")
            return Credentials.from_service_account_info(credentials_dict)
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse GOOGLE_SHEETS_CREDENTIALS JSON: {e}")
    
    # Second try: File path in environment variable
    credentials_path = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_PATH")
    if credentials_path and os.path.exists(credentials_path):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
            print(f"✅ Google Sheets credentials loaded from file: {credentials_path}")
            return creds
        except Exception as e:
            print(f"⚠️ Failed to load credentials from file {credentials_path}: {e}")
    
    # Third try: Default path
    default_path = "./credentials/google_sheets_credentials.json"
    if os.path.exists(default_path):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(default_path, scopes=scope)
            print(f"✅ Google Sheets credentials loaded from default path: {default_path}")
            return creds
        except Exception as e:
            print(f"⚠️ Failed to load credentials from default path: {e}")
    
    raise ValueError("Google Sheets credentials not found. Set GOOGLE_SHEETS_CREDENTIALS or GOOGLE_SHEETS_CREDENTIALS_PATH")

# Authenticate
try:
    client_gspread = gspread.authorize(get_google_sheets_credentials())
    spreadsheet = client_gspread.open("Response_Log")
    response_sheet = spreadsheet.sheet1
    print("✅ Google Sheets connection established")
except Exception as e:
    print(f"⚠️ Error connecting to Google Sheets: {e}")
    # Create dummy objects if connection fails to prevent app crash
    class DummySheet:
        def append_row(self, *args, **kwargs): 
            print(f"[DummySheet] Would log: {args}")
            pass
        def worksheet(self, *args): 
            return self
        def add_worksheet(self, *args, **kwargs): 
            return self
    spreadsheet = DummySheet()
    response_sheet = DummySheet()

# Setup Timing Sheet
try:
    timing_sheet = spreadsheet.worksheet("Timing_Log")
    print("✅ Timing_Log sheet found")
except:
    try:
        timing_sheet = spreadsheet.add_worksheet(title="Timing_Log", rows="1000", cols="15")
        headers = [
            "Timestamp", "Session_ID", "Question", "Total_Time_MS",
            "Intent_Classification_MS", "Memory_Retrieval_MS", "RAG_Retrieval_MS", 
            "Embedding_Generation_MS", "Similarity_Calculation_MS", "Context_Processing_MS",
            "LLM_Generation_MS", "Memory_Update_MS", "Logging_MS", "Error_Step", "Notes"
        ]
        timing_sheet.append_row(headers)
        print("✅ Timing_Log sheet created")
    except Exception as e:
        print(f"⚠️ Could not create Timing_Log sheet: {e}")
        timing_sheet = None

# Setup Feedback Sheet
try:
    feedback_sheet = spreadsheet.worksheet("Feedback_Log")
    print("✅ Feedback_Log sheet found")
except:
    try:
        feedback_sheet = spreadsheet.add_worksheet(title="Feedback_Log", rows="1000", cols="6")
        headers = ["Timestamp", "Session_ID", "User_Message", "Bot_Response", "Rating", "Flag_Reason"]
        feedback_sheet.append_row(headers)
        print("✅ Feedback_Log sheet created")
    except Exception as e:
        print(f"⚠️ Could not create Feedback_Log sheet: {e}")
        feedback_sheet = None

# === Logging Functions ===

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
        print(f"📊 Logged response: {question[:50]}... | Sources: {source_ids}")
    except Exception as e:
        print(f"⚠️ Failed to log to Google Sheet: {e}")
        # Fallback to local file
        with open("response_log_fallback.txt", "a") as f:
            f.write(f"{timestamp},{session_id},{question},{answer},{source_ids}\n")

def log_timing_data(question, session_id, timing_summary, error_step=None, notes=None):
    """Log timing data to the timing sheet"""
    if timing_sheet is None: 
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_times = timing_summary['step_times']
    
    row = [
        timestamp,
        session_id,
        question[:100] + "..." if len(question) > 100 else question,
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
        print(f"⏱️ Logged timing data: Total {timing_summary['total_time_ms']}ms")
    except Exception as e:
        print(f"⚠️ Failed to log timing data: {e}")

# === Feedback Functions ===

def _log_feedback_background(row):
    """Helper to run network request in background thread"""
    try:
        if feedback_sheet:
            feedback_sheet.append_row(row)
            print("✅ Feedback logged successfully.")
        else:
            print("⚠️ Feedback sheet not available.")
    except Exception as e:
        print(f"⚠️ Failed to log feedback: {e}")

def submit_feedback(rating, reason, history, session_id):
    """
    Handles user feedback submission.
    rating: 'Positive' or 'Negative'
    reason: User provided text
    history: Gradio chat history list
    """
    if not history or len(history) == 0:
        return "No conversation to rate yet."
    
    # Get the last interaction
    last_interaction = history[-1]
    
    # Safety check for history format
    if isinstance(last_interaction, list) and len(last_interaction) >= 2:
        user_msg = last_interaction[0]
        bot_msg = last_interaction[1]
    else:
        return "Error reading conversation history."
        
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare row data
    row = [timestamp, session_id, user_msg, bot_msg, rating, reason]
    
    # Run in thread to prevent UI blocking
    threading.Thread(target=_log_feedback_background, args=(row,)).start()
    
    return f"✅ Feedback received ({rating}). Thank you!"

# === LangGraph Memory Setup ===
conn = sqlite3.connect("xeno_memory.db", check_same_thread=False)
memory = SqliteSaver(conn=conn)

def create_session_config(session_id):
    return {"configurable": {"thread_id": session_id}}

def update_memory(config, user_message, assistant_message):
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
    with timer.time_step("memory_retrieval"):
        full_checkpoint = memory.get(config) or {}
        return full_checkpoint.get("channel_values", {}).get("messages", [])

# === Intent Classification System ===
intent_classifier = IntentClassifier()

# === Load and Clean Knowledge Base ===
try:
    df_kb = pd.read_json("XENO_Uganda_KnowledgeBase_Advisory.json")
    df_kb.dropna(subset=['Content'], inplace=True)
    print(f"✅ Loaded knowledge base with {len(df_kb)} entries")
    
    def prepare_documents(data):
        documents, metadatas, ids = [], [], []
        for item in data:
            documents.append(f"Question: {item['Question']}\nAnswer: {item['Content']}")
            metadatas.append({
                "question": item["Question"],
                "content": item["Content"],
                "id": str(item["ID"])
            })
            ids.append(str(item["ID"]))
        return documents, metadatas, ids

    xeno_data_list = df_kb.to_dict('records')
    documents, metadatas, ids = prepare_documents(xeno_data_list)
    
    # ===== NEW: Ingest into Knowledge Graph =====
    if knowledge_ingestor:
        try:
            print("🔄 Ingesting knowledge base into graph...")
            knowledge_ingestor.ingest_knowledge_base(xeno_data_list)
            print("✅ Knowledge graph ingestion completed")
        except Exception as e:
            print(f"⚠️ Could not ingest knowledge base into graph: {e}")
    # ============================================
    
except Exception as e:
    print(f"⚠️ Could not load JSON knowledge base: {e}")
    traceback.print_exc()
    documents, metadatas, ids = [], [], []

# === Setup ChromaDB ===
try:
    client = chromadb.PersistentClient(path="/tmp/xeno_db")
    try:
        collection = client.get_collection(name=collection_name)
        print(f"✅ Loaded existing ChromaDB collection: {collection_name}")
    except:
        print(f"🔄 Creating new ChromaDB collection: {collection_name}")
        collection = client.create_collection(name=collection_name)
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"✅ Added {len(documents)} documents to ChromaDB")
except Exception as e:
    print(f"❌ Failed to initialize ChromaDB: {e}")
    raise

vector_store = Chroma(client=client, collection_name=collection_name)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ===== NEW: Update Cognitive Retriever with Vector Store =====
if cognitive_retriever:
    cognitive_retriever.vector_store = vector_store
    print("✅ Cognitive Retriever updated with vector store")
# =============================================================

# === Prompt System ===
SYSTEM_PROMPT = """You are a friendly XENO Support Assistant, an AI-powered helpful and professional customer service representative.
Use only the information provided in the knowledge base context to answer user queries.
Do not hallucinate. If context doesn't contain relevant info, say so in a calm polite manner by saying I'm sorry, I can't assist with that.
Only use context that is clearly relevant to the user's question.
For greetings like "hi" or "hello", respond politely without using the context.
Remember previous conversations and use cognitive reasoning when appropriate."""

# === Enhanced Context Processing with Cognitive Reasoning ===
def process_context_with_cognition(results, cosine_scores, user_query: str, session_id: str, chat_history: List):
    """Enhanced context processing with cognitive reasoning"""
    with timer.time_step("context_processing"):
        sorted_indices = np.argsort(cosine_scores)[::-1][:4]  # Get top 4 for cognitive processing
        
        # Get base context
        formatted_context = ""
        source_ids = []
        knowledge_pairs = []
        notes = []  # For tracking notes
        
        for i, idx in enumerate(sorted_indices, 1):
            result = results[idx]
            score = cosine_scores[idx]
            question = result.metadata.get('question', 'N/A')
            answer = result.metadata.get('content', 'N/A')
            formatted_context += f"Knowledge Entry {i}:\n"
            formatted_context += f"Q: {question}\n"
            formatted_context += f"A: {answer}\n"
            formatted_context += f"Relevance Score: {score:.3f}\n"
            formatted_context += "-" * 40 + "\n"
            source_ids.append(str(result.metadata.get('id', 'N/A')))
            knowledge_pairs.append((question, answer))
        
        # ===== NEW: Apply Cognitive Reasoning =====
        if cognitive_reasoning and cognitive_memory:
            try:
                with timer.time_step("cognitive_reasoning"):
                    # Store conversation in cognitive memory
                    if chat_history:
                        last_turn = chat_history[-1] if len(chat_history) > 0 else None
                        if last_turn and len(last_turn) >= 2:
                            cognitive_memory.store_conversation_turn(
                                session_id=session_id,
                                user_message=last_turn[0] if isinstance(last_turn, list) else last_turn.get('user', ''),
                                assistant_message=last_turn[1] if isinstance(last_turn, list) else last_turn.get('assistant', ''),
                                metadata={"source": "conversation"}
                            )
                    
                    # Analyze user intent and extract entities
                    intent_analysis = cognitive_reasoning.analyze_user_intent(user_query)
                    
                    # Apply reasoning to enhance context
                    enhanced_context = cognitive_reasoning.enhance_response_with_reasoning(
                        user_query=user_query,
                        retrieved_context=formatted_context,
                        conversation_history=chat_history,
                        session_id=session_id
                    )
                    
                    # Store reasoning trace
                    cognitive_memory.store_reasoning_trace(
                        session_id=session_id,
                        user_query=user_query,
                        reasoning_steps=intent_analysis.get('reasoning_steps', []),
                        final_answer="",  # Will be filled after LLM generation
                        metadata={
                            "intent": intent_analysis.get('primary_intent', 'unknown'),
                            "entities": intent_analysis.get('entities', []),
                            "confidence": intent_analysis.get('confidence', 0.0)
                        }
                    )
                    
                    # Combine base context with enhanced reasoning
                    cognitive_enhancement = f"\n=== Cognitive Analysis ===\n"
                    cognitive_enhancement += f"Detected Intent: {intent_analysis.get('primary_intent', 'unknown')}\n"
                    cognitive_enhancement += f"Confidence: {intent_analysis.get('confidence', 0.0):.2f}\n"
                    
                    if intent_analysis.get('entities'):
                        cognitive_enhancement += f"Entities: {', '.join(intent_analysis.get('entities', []))}\n"
                    
                    cognitive_enhancement += f"\nReasoning Insights:\n{enhanced_context}\n"
                    cognitive_enhancement += "=" * 40 + "\n"
                    
                    formatted_context = cognitive_enhancement + formatted_context
                    
                    notes.append(f"Cognitive reasoning applied: {intent_analysis.get('primary_intent')}")
                    
            except Exception as e:
                print(f"⚠️ Cognitive reasoning failed: {e}")
                notes.append(f"Cognitive reasoning error: {str(e)}")
        # ==========================================
        
        return formatted_context, source_ids, knowledge_pairs, notes

# === Enhanced LLM Generation ===
def generate_xeno_response(context, question, chat_history, session_id):
    with timer.time_step("llm_generation"):
        model = genai.GenerativeModel(llm_model_name)
        
        # Format chat history
        formatted_history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        ) if chat_history else "None"
        
        # Enhanced prompt with cognitive awareness
        enhanced_prompt = f"""{SYSTEM_PROMPT}

=== COGNITIVE CONTEXT ===
You have access to cognitive reasoning about the user's intent and needs.
Use this to provide more personalized and context-aware responses.

=== CONVERSATION HISTORY ===
{formatted_history}

=== RETRIEVED KNOWLEDGE ===
{context}

=== USER QUESTION ===
{question}

=== INSTRUCTIONS ===
1. Use the retrieved knowledge as your primary source
2. Consider the conversation history for context
3. Apply cognitive insights to tailor your response
4. Be precise, helpful, and professional
5. If information is insufficient, politely say so

Response:"""
        
        response = model.generate_content(enhanced_prompt)
        answer = response.text.strip()
        
        # ===== NEW: Store final answer in cognitive memory =====
        if cognitive_memory:
            try:
                cognitive_memory.update_reasoning_trace_with_answer(
                    session_id=session_id,
                    user_query=question,
                    final_answer=answer
                )
            except Exception as e:
                print(f"⚠️ Could not update reasoning trace: {e}")
        # ======================================================
        
        return answer

# === Main Interface Logic with Cognitive Integration ===
def get_context_and_answer(message, history, session_id="default"):
    # Reset timer for new request
    timer.reset()
    error_step = None
    notes = []
    
    try:
        # Create session config
        config = create_session_config(session_id)
        
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
                    # Step 3: RAG Retrieval (using cognitive retriever if available)
                    with timer.time_step("rag_retrieval"):
                        if cognitive_retriever:
                            # Use cognitive retriever for enhanced retrieval
                            queried_results, retrieval_notes = cognitive_retriever.retrieve_with_reasoning(
                                query=message,
                                session_id=session_id,
                                conversation_history=chat_history
                            )
                            notes.extend(retrieval_notes)
                        else:
                            # Fallback to standard retriever
                            queried_results = retriever.invoke(message)
                            notes.append("Using standard retriever")
                    
                    # Step 4: Embedding Generation
                    query_embedding, doc_embeddings = generate_embeddings(message, queried_results)
                    
                    # Step 5: Similarity Calculation
                    with timer.time_step("similarity_calculation"):
                        cosine_scores = util.cos_sim(
                            torch.tensor(query_embedding).float(), 
                            torch.tensor(doc_embeddings).float()
                        )[0].tolist()
                        max_score = max(cosine_scores) if cosine_scores else 0

                    if max_score < SIMILARITY_THRESHOLD:
                        # ===== NEW: Try cognitive fallback =====
                        if cognitive_reasoning:
                            try:
                                notes.append(f"Low similarity score: {max_score:.3f}, attempting cognitive fallback")
                                
                                # Analyze why we might not have good matches
                                analysis = cognitive_reasoning.analyze_query_ambiguity(
                                    query=message,
                                    retrieved_results=queried_results,
                                    similarity_scores=cosine_scores
                                )
                                
                                if analysis.get('suggested_rewrites'):
                                    # Try with rewritten query
                                    rewritten_query = analysis['suggested_rewrites'][0]
                                    notes.append(f"Trying rewritten query: {rewritten_query}")
                                    
                                    # Retry with rewritten query
                                    if cognitive_retriever:
                                        queried_results, _ = cognitive_retriever.retrieve_with_reasoning(
                                            query=rewritten_query,
                                            session_id=session_id,
                                            conversation_history=chat_history
                                        )
                                    
                                    # Recalculate embeddings and scores
                                    query_embedding, doc_embeddings = generate_embeddings(rewritten_query, queried_results)
                                    cosine_scores = util.cos_sim(
                                        torch.tensor(query_embedding).float(), 
                                        torch.tensor(doc_embeddings).float()
                                    )[0].tolist()
                                    max_score = max(cosine_scores) if cosine_scores else 0
                                    
                                    notes.append(f"Rewritten query similarity: {max_score:.3f}")
                            except Exception as e:
                                notes.append(f"Cognitive fallback failed: {str(e)}")
                        # ========================================
                        
                        if max_score < SIMILARITY_THRESHOLD:
                            answer = "I'm sorry, I couldn't find specific information for your question. Could you try rephrasing it, or contact XENO support directly?"
                            notes.append(f"Final similarity too low: {max_score:.3f}")
                        else:
                            notes.append(f"Cognitive rewrite successful: {max_score:.3f}")
                    
                    if not answer:  # Proceed if we have good similarity or cognitive rewrite worked
                        # Step 6: Enhanced Context Processing with Cognitive Reasoning
                        context, source_ids_list, knowledge_pairs, processing_notes = process_context_with_cognition(
                            queried_results, 
                            cosine_scores,
                            user_query=message,
                            session_id=session_id,
                            chat_history=chat_history
                        )
                        notes.extend(processing_notes)
                        
                        # Step 7: Enhanced LLM Generation
                        answer = generate_xeno_response(context, message, chat_history, session_id)
                        source_ids = ", ".join(source_ids_list)
                        notes.append(f"Max similarity: {max_score:.3f}")

                except Exception as e:
                    error_step = timer.current_step or "rag_processing"
                    print(f"❌ Error during RAG processing: {e}")
                    traceback.print_exc()
                    answer = "I apologize, but I'm having a technical issue. Please try again shortly or contact XENO support."
                    notes.append(f"Error: {str(e)}")

        # Step 8: Memory Update
        update_memory(config, message, answer)
        
        # ===== NEW: Store in Cognitive Memory =====
        if cognitive_memory:
            try:
                cognitive_memory.store_conversation_turn(
                    session_id=session_id,
                    user_message=message,
                    assistant_message=answer,
                    metadata={
                        "intent": intent,
                        "sources": source_ids,
                        "notes": "; ".join(notes)
                    }
                )
            except Exception as e:
                print(f"⚠️ Could not store in cognitive memory: {e}")
        # ==========================================
        
        # Step 9: Response Logging
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
        logging.error(f"❌ Error in main pipeline: {e}")
        logging.error(traceback.format_exc())
        
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
def respond(message: str, history: List, session_id: str):
    """Gradio's main response function"""
    if not session_id:
        session_id = str(uuid.uuid4())
    
    bot_response = get_context_and_answer(message, history, session_id)
    history.append([message, bot_response])
    
    return "", history


def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 ASKXENO - Cognitive Assistant
        **Welcome to XENO AI Support!**
        
        I can help you with questions about XENO financial services including:
        - Account management and setup
        - Transaction processes and fees
        - Platform features and troubleshooting
        - General service information
        
        *Now with **cognitive reasoning** for more intelligent, personalized responses!*
        
        *Simply type your question below to get started!*
        """)
        
        # Hidden state for session
        session_id_box = gr.Textbox(label="Session ID", value=str(uuid.uuid4()), visible=False)
        
        chatbot = gr.Chatbot(
            label="XENO Assistant",
            bubble_full_width=False,
            height=450,
            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=XENO")
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your question here...",
                scale=4,
                container=False
            )
            send_button = gr.Button("Send", variant="primary", scale=1, size="lg")
        
        # ===== FEEDBACK SECTION =====
        with gr.Row():
            with gr.Accordion("📊 Rate this response / Flag Issue", open=False):
                with gr.Row():
                    thumbs_up = gr.Button("👍 Good Answer", variant="secondary")
                    thumbs_down = gr.Button("👎 Bad / Flag", variant="secondary")
                
                feedback_reason = gr.Textbox(
                    label="Reason (optional)", 
                    placeholder="E.g., Incorrect fees, hallucination, etc.",
                    lines=2
                )
                feedback_status = gr.Markdown("")

        # Feedback Event Listeners
        thumbs_up.click(
            fn=lambda h, s, r: submit_feedback("Positive", r if r else "Good", h, s),
            inputs=[chatbot, session_id_box, feedback_reason],
            outputs=[feedback_status]
        ).then(
            lambda: gr.update(value=""),
            outputs=[feedback_reason]
        )
        
        thumbs_down.click(
            fn=lambda r, h, s: submit_feedback("Negative", r, h, s),
            inputs=[feedback_reason, chatbot, session_id_box],
            outputs=[feedback_status]
        ).then(
            lambda: gr.update(value=""),
            outputs=[feedback_reason]
        )
        # =============================

        # Chat Event Listeners
        send_button.click(respond, [msg, chatbot, session_id_box], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, session_id_box], [msg, chatbot])
        
        # Add some helpful examples
        gr.Examples(
            examples=[
                "How do I open an account with XENO?",
                "What are the fees for sending money to Kenya?",
                "How do I reset my password?",
                "What is the minimum balance required?",
                "How long does international transfer take?"
            ],
            inputs=msg,
            label="Try these examples:"
        )
            
    return demo


if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting XENO Cognitive Bot...")
    print("="*50)
    
    # Print configuration summary
    print(f"\n📊 Configuration Summary:")
    print(f"   Gemini API: {'✅' if os.environ.get('GEMINI_API_KEY') else '❌'}")
    print(f"   Neo4j Connection: {'✅' if neo4j_client else '❌'}")
    print(f"   Cognitive System: {'✅' if cognitive_memory else '❌'}")
    print(f"   Knowledge Base: {len(documents)} entries")
    print(f"   Server: {SERVER_NAME}:{SERVER_PORT}")
    print("="*50 + "\n")
    
    iface = create_interface()
    iface.launch(
        share=False, 
        server_name=SERVER_NAME, 
        server_port=SERVER_PORT, 
        ssr_mode=False,
        show_error=True
    )