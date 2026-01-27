"""
XENO Bot - AI-powered customer service assistant
Main application file with Gradio interface
"""

import logging
import os
import traceback

from src.config import (COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL_NAME,
                        SERVER_NAME, SERVER_PORT, SIMILARITY_THRESHOLD)
from src.intent_classifier import IntentClassifier
from src.interface import create_interface
from src.knowledge_base import get_knowledge_base_data
from src.logger import log_response, log_timing_data
from src.memory import create_session_config, retrieve_memory, update_memory
from src.response_generator import generate_xeno_response
# Import custom modules
from src.utils import PipelineTimer
from src.vector_store import (generate_embeddings, initialize_vector_store,
                              process_context)

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


# === Core Orchestration Logic ===
def get_context_and_answer(
    message, history, session_id, intent_classifier, retriever
):
    """
    Core orchestration function that handles the RAG pipeline
    
    Args:
        message: User's message
        history: Chat history
        session_id: Session identifier
        intent_classifier: IntentClassifier instance
        retriever: Vector store retriever instance
    
    Returns:
        Generated answer string
    """
    # Create timer per session
    timer = PipelineTimer()
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

        if intent != "query":
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
                        import sentence_transformers.util as util
                        import torch
                        cosine_scores = util.cos_sim(
                            torch.tensor(query_embedding).float(),
                            torch.tensor(doc_embeddings).float(),
                        )[0].tolist()
                        max_score = max(cosine_scores) if cosine_scores else 0

                    if max_score < SIMILARITY_THRESHOLD:
                        answer = "I'm sorry, I couldn't find specific information for your question. Could you try rephrasing it, or contact XENO support directly?"
                        notes.append(f"Low similarity score: {max_score:.3f}")
                    else:
                        # Step 6: Context Processing
                        context, source_ids_list, knowledge_pairs = process_context(
                            queried_results, cosine_scores
                        )

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
            notes="; ".join(notes) if notes else None,
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
            notes=f"Pipeline error: {str(e)}",
        )

        return "I apologize, but I encountered an error processing your request. Please try again."


# === Main Interface Logic ===

if __name__ == "__main__":
    iface = create_interface(intent_classifier, retriever)
    iface.launch(
        share=False, server_name=SERVER_NAME, server_port=SERVER_PORT, ssr_mode=False
    )
