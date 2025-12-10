"""
XENO Bot - AI-powered customer service assistant
Main application file with Gradio interface
"""
import uuid
import gradio as gr
import logging
import traceback
from typing import List

# Import custom modules
from src.utils import PipelineTimer
from src.config import SIMILARITY_THRESHOLD, SERVER_NAME, SERVER_PORT
from src.memory import create_session_config, update_memory, retrieve_memory
from src.intent_classifier import IntentClassifier
from src.vector_store import (
    initialize_vector_store,
    generate_embeddings,
    calculate_similarity,
    process_context
)
from src.response_generator import generate_xeno_response
from src.logger import log_response, log_timing_data

# Initialize components
timer = PipelineTimer()
intent_classifier = IntentClassifier()

# Initialize vector store
print("Initializing vector store...")
collection, vector_store, retriever = initialize_vector_store()
print("Vector store initialized successfully!")


# === Main Interface Logic ===
def get_context_and_answer(message, history, session_id="default"):
    """Main pipeline with comprehensive timing"""
    # Reset timer for new request
    timer.reset()
    error_step = None
    notes = []
    
    try:
        # Create session config
        config = create_session_config(session_id)
        
        # Step 1: Intent Classification
        intent, direct_response = intent_classifier.classify_intent(message, timer)
        
        # Step 2: Memory Retrieval
        chat_history = retrieve_memory(config, timer)
        
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
                    cosine_scores = calculate_similarity(
                        query_embedding, doc_embeddings, timer
                    )
                    max_score = max(cosine_scores)

                    if max_score < SIMILARITY_THRESHOLD:
                        answer = "I'm sorry, I couldn't find specific information for your question. Could you try rephrasing it, or contact XENO support directly?"
                        notes.append(f"Low similarity score: {max_score:.3f}")
                    else:
                        # Step 6: Context Processing
                        context, source_ids_list, knowledge_pairs = process_context(
                            queried_results, cosine_scores, timer=timer
                        )
                        
                        # Step 7: LLM Generation
                        answer = generate_xeno_response(context, message, chat_history, timer)
                        source_ids = ", ".join(source_ids_list)
                        notes.append(f"Max similarity: {max_score:.3f}")

                except Exception as e:
                    error_step = timer.current_step or "rag_processing"
                    print(f"Error during RAG processing: {e}")
                    answer = "I apologize, but I'm having a technical issue. Please try again shortly or contact XENO support."
                    notes.append(f"Error: {str(e)}")

        # Step 8: Memory Update
        update_memory(config, message, answer, timer)
        
        # Step 9: Response Logging
        log_response(message, answer, source_ids, knowledge_pairs, session_id, timer)
        
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
        
        session_id_box = gr.Textbox(
            label="Session ID", 
            value=str(uuid.uuid4()), 
            interactive=True
        )
        
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
    iface.launch(
        share=False, 
        server_name=SERVER_NAME, 
        server_port=SERVER_PORT, 
        ssr_mode=False
    )