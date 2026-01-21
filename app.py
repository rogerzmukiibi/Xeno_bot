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
from typing import Dict, List, Tuple
import time
from contextlib import contextmanager
import logging
import traceback

# Import custom modules
from src.utils import PipelineTimer
from src.config import (
    SIMILARITY_THRESHOLD,
    SERVER_NAME,
    SERVER_PORT,
    SYSTEM_PROMPT
)
from src.memory import create_session_config, update_memory, retrieve_memory
from src.intent_classifier import IntentClassifier
from src.vector_store import (
    initialize_vector_store,
    generate_embeddings,
    calculate_similarity,
    process_context
)
from src.response_generator import generate_xeno_response
from src.logger import (
    log_response, 
    log_timing_data, 
    log_feedback
)
from src.knowledge_base import get_knowledge_base_data

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

# === Intent Classification System ===
intent_classifier = IntentClassifier()

# === Load and Clean Knowledge Base ===
documents, metadatas, ids = get_knowledge_base_data()

# === Setup ChromaDB ===
collection, vector_store, retriever = initialize_vector_store()

# === LLM Generation ===
def generate_xeno_response(context, question, chat_history):
    with timer.time_step("llm_generation"):
        formatted_history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        ) if chat_history else "None"
        
        prompt = f"{SYSTEM_PROMPT}\n### HISTORY ###\n{formatted_history}\n### CONTEXT ###\n{context}\n### QUESTION ###\n{question}"
        
        response = genai_client.models.generate_content(
            model=llm_model_name,
            contents={"text": prompt},
        )
        return response.text.strip()

# === Main Interface Logic ===
def get_context_and_answer(message, history, session_id="default"):
    # Reset timer for new request
    timer.reset()
    error_step = None
    notes = []
    
    try:
        # Create session memory config
        memory_config = create_session_config(session_id)
        
        # Step 1: Intent Classification
        intent, direct_response = intent_classifier.classify_intent(message)
        
        # Step 2: Memory Retrieval
        chat_history = retrieve_memory(memory_config)
        
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
                    query_embedding, doc_embeddings = generate_embeddings(
                        message, queried_results, timer
                    )
                    
                    # Step 5: Similarity Calculation
                    with timer.time_step("similarity_calculation"):
                        cosine_scores = util.cos_sim(
                            torch.tensor(query_embedding).float(), 
                            torch.tensor(doc_embeddings).float()
                        )[0].tolist()
                        max_score = max(cosine_scores) if cosine_scores else 0

                    if max_score < SIMILARITY_THRESHOLD:
                        answer = "I'm sorry, I couldn't find specific information for your question. Could you try rephrasing it, or contact XENO support directly?"
                        notes.append(f"Low similarity score: {max_score:.3f}")
                    else:
                        # Step 6: Context Processing
                        context, source_ids_list, knowledge_pairs = process_context(queried_results, cosine_scores)
                        
                        # Step 7: LLM Generation
                        answer = generate_xeno_response(context, message, chat_history)
                        source_ids = ", ".join(source_ids_list)
                        notes.append(f"Max similarity: {max_score:.3f}")

                except Exception as e:
                    error_step = timer.current_step or "rag_processing"
                    print(f"Error during RAG processing: {e}")
                    traceback.print_exc()
                    answer = "I apologize, but I'm having a technical issue. Please try again shortly or contact XENO support."
                    notes.append(f"Error: {str(e)}")

        # Step 8: Memory Update
        update_memory(memory_config, message, answer)
        
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
        logging.error(f"Error in main pipeline: {e}")
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
        # ASKXENO
        **Welcome to XENO AI Support!**
        
        I can help you with questions about XENO financial services including:
        - Account management and setup
        - Transaction processes and fees
        - Platform features and troubleshooting
        - General service information
        
        *Simply type your question below to get started!*
        """)
        
        # Hidden state for session
        session_id_box = gr.Textbox(label="Session ID", value=str(uuid.uuid4()), visible=False)
        
        chatbot = gr.Chatbot(
            label="XENO Assistant",
            bubble_full_width=False,
            height=450
        )
        
        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your question here...",
                scale=4,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)
        
        # ===== FEEDBACK SECTION =====
        with gr.Row():
            with gr.Accordion("Rate this response / Flag Issue", open=False):
                with gr.Row():
                    thumbs_up = gr.Button("👍 Good Answer")
                    thumbs_down = gr.Button("👎 Bad / Flag")
                
                feedback_reason = gr.Textbox(
                    label="Reason ", 
                    placeholder="E.g., Incorrect fees, hallucination,"
                )
                feedback_status = gr.Label(value="", label="Status", show_label=False)

        # Feedback Event Listeners
        # Logic: If Thumbs Up is clicked, send 'Positive'. If Textbox is empty, reason defaults to "Good".
        thumbs_up.click(
            fn=lambda h, s, r: log_feedback("Positive", r if r else "Good", h, s),
            inputs=[chatbot, session_id_box, feedback_reason],
            outputs=[feedback_status]
        )
        
        # Logic: If Thumbs Down is clicked, send 'Negative' with the content of the textbox.
        thumbs_down.click(
            fn=lambda r, h, s: log_feedback("Negative", r, h, s),
            inputs=[feedback_reason, chatbot, session_id_box],
            outputs=[feedback_status]
        )
        # =============================

        # Chat Event Listeners
        send_button.click(respond, [msg, chatbot, session_id_box], [msg, chatbot])
        msg.submit(respond, [msg, chatbot, session_id_box], [msg, chatbot])
            
    return demo


if __name__ == "__main__":
    iface = create_interface()
    iface.launch(
        share=False, 
        server_name=SERVER_NAME, 
        server_port=SERVER_PORT, 
        ssr_mode=False
    )