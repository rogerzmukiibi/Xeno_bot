import operator
import time
import logging
import json
from datetime import datetime
from typing import Annotated, List, TypedDict, Union, Dict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.vector_store import initialize_vector_store
from src.intent_classifier import IntentClassifier
import google.generativeai as genai
from src.config import LLM_MODEL_NAME, SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
# We initialize here to be used by the nodes
try:
    collection, vector_store, retriever = initialize_vector_store()
except Exception as e:
    logger.error(f"Error initializing vector store in agent_graph: {e}")
    retriever = None

intent_classifier = IntentClassifier()

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[List[BaseMessage], operator.add]
    question: str
    documents: List[str]
    context_metadata: List[Dict[str, Any]]
    generation: str
    intent: str
    steps: Annotated[List[str], operator.add]
    retry_count: int
    session_id: str

def log_event(event: str, data: Dict[str, Any]):
    """Logs a structured JSON event."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **data
    }
    logger.info(json.dumps(log_entry))

def execute_with_retry(func, *args, max_retries=3, delay=1, **kwargs):
    """Helper to execute a function with retries."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final failure in {func.__name__} after {max_retries} attempts: {e}")
                raise e
            logger.warning(f"Attempt {attempt + 1} failed in {func.__name__}: {e}. Retrying in {delay}s...")
            time.sleep(delay * (2 ** attempt)) # Exponential backoff

# === Nodes ===

def classify_input(state: AgentState):
    """Classifies the user input to decide the next step."""
    question = state["question"]
    session_id = state.get("session_id", "unknown")
    
    intent, response = intent_classifier.classify_intent(question)
    
    log_event("classify_input", {
        "session_id": session_id,
        "question": question,
        "intent": intent,
        "is_direct_response": intent != "query"
    })
    
    return {"intent": intent, "generation": response, "steps": ["classify_input"]}

def retrieve(state: AgentState):
    """Retrieves documents from the vector store."""
    question = state["question"]
    session_id = state.get("session_id", "unknown")
    
    try:
        if retriever:
            documents = retriever.invoke(question)
            # Extract content and metadata for context
            doc_texts = [f"Content: {doc.page_content}" for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            log_event("retrieve", {
                "session_id": session_id,
                "question": question,
                "docs_found": len(documents)
            })
        else:
            logger.warning("Retriever is not initialized.")
            doc_texts = []
            metadatas = []
            log_event("retrieve_error", {
                "session_id": session_id,
                "error": "Retriever not initialized"
            })
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        doc_texts = []
        metadatas = []
        log_event("retrieve_error", {
            "session_id": session_id,
            "error": str(e)
        })
        
    return {"documents": doc_texts, "context_metadata": metadatas, "steps": ["retrieve"]}

def grade_documents(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the question.
    If any document is relevant, we proceed to generate.
    """
    question = state["question"]
    documents = state["documents"]
    metadatas = state["context_metadata"]
    session_id = state.get("session_id", "unknown")
    
    # Simple LLM check for relevance
    model = genai.GenerativeModel(LLM_MODEL_NAME)
    
    relevant_docs = []
    relevant_metas = []
    
    # Check each doc
    for i, doc in enumerate(documents):
        prompt = f"""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {doc} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning useful to the question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        try:
            # Use retry logic for LLM call
            def call_grader():
                return model.generate_content(prompt)
                
            response = execute_with_retry(call_grader)
            score = response.text.strip().lower()
            if "yes" in score:
                relevant_docs.append(doc)
                if i < len(metadatas):
                    relevant_metas.append(metadatas[i])
        except Exception as e:
            logger.error(f"Error grading document {i}: {e}")
            # Fallback to keeping it if error
            relevant_docs.append(doc)
            if i < len(metadatas):
                relevant_metas.append(metadatas[i])
    
    log_event("grade_documents", {
        "session_id": session_id,
        "total_docs": len(documents),
        "relevant_docs": len(relevant_docs),
        "relevance_ratio": len(relevant_docs) / len(documents) if documents else 0
    })
            
    return {"documents": relevant_docs, "context_metadata": relevant_metas, "steps": ["grade_documents"]}

def generate(state: AgentState):
    """Generates a response using the LLM."""
    question = state["question"]
    documents = state["documents"]
    messages = state["messages"]
    session_id = state.get("session_id", "unknown")
    
    context = "\n\n".join(documents)
    
    model = genai.GenerativeModel(LLM_MODEL_NAME)
    
    # Format history
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])
    
    prompt = f"""{SYSTEM_PROMPT}
    
    ### HISTORY ###
    {history_str}
    
    ### CONTEXT ###
    {context}
    
    ### QUESTION ###
    {question}
    """
    
    try:
        def call_generator():
            return model.generate_content(prompt)
            
        response = execute_with_retry(call_generator)
        
        log_event("generate", {
            "session_id": session_id,
            "context_length": len(context),
            "response_length": len(response.text)
        })
        
        return {"generation": response.text, "steps": ["generate"]}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        log_event("generate_error", {
            "session_id": session_id,
            "error": str(e)
        })
        return {"generation": "I apologize, but I'm having trouble generating a response right now. Please try again later.", "steps": ["generate"]}

def transform_query(state: AgentState):
    """Transform the query to produce a better question."""
    question = state["question"]
    session_id = state.get("session_id", "unknown")
    model = genai.GenerativeModel(LLM_MODEL_NAME)
    
    prompt = f"""You are generating a better version of a user question for vector store retrieval. \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    {question} \n
    Formulate an improved question:"""
    
    try:
        def call_transform():
            return model.generate_content(prompt)
            
        response = execute_with_retry(call_transform)
        better_question = response.text.strip()
        
        log_event("transform_query", {
            "session_id": session_id,
            "original_question": question,
            "transformed_question": better_question
        })
        
    except Exception as e:
        logger.error(f"Error transforming query: {e}")
        better_question = question # Fallback to original question
        log_event("transform_query_error", {
            "session_id": session_id,
            "error": str(e)
        })
    
    return {"question": better_question, "steps": ["transform_query"]}

# === Conditional Edges ===

def decide_to_generate(state: AgentState):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    intent = state["intent"]
    
    if intent != "query":
        return "end" # Already have a response from classifier
        
    if not state["documents"]:
        # No relevant documents found, rewrite query
        return "transform_query"
    
    return "generate"

def route_intent(state: AgentState):
    """Routes based on intent."""
    intent = state["intent"]
    if intent == "query":
        return "retrieve"
    else:
        return "end"

# === Graph Construction ===

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("classify_input", classify_input)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Add edges
workflow.set_entry_point("classify_input")

workflow.add_conditional_edges(
    "classify_input",
    route_intent,
    {
        "retrieve": "retrieve",
        "end": END
    }
)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
        "end": END # Should not happen here usually, but safe fallback
    }
)

workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

# Compile
agent_app = workflow.compile()
