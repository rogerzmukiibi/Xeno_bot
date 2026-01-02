"""
Response Generation module for XENO Bot
Handles LLM response generation
"""
from google import genai
from typing import List, Dict
from src.config import LLM_MODEL_NAME, SYSTEM_PROMPT, client


def generate_xeno_response(context: str, question: str, chat_history: List[Dict[str, str]], timer=None) -> str:
    """
    Generate a response using the LLM
    
    Args:
        context: Formatted context from knowledge base
        question: User's question
        chat_history: List of previous messages
        timer: Optional timer object for tracking
    
    Returns:
        Generated response text
    """
    if timer:
        with timer.time_step("llm_generation"):
            return _generate_response_impl(context, question, chat_history)
    else:
        return _generate_response_impl(context, question, chat_history)


def _generate_response_impl(context: str, question: str, chat_history: List[Dict[str, str]]) -> str:
    """Internal implementation of response generation"""
    # Format chat history
    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
    ) if chat_history else "None"
    
    # Build prompt
    prompt = f"{SYSTEM_PROMPT}\n### HISTORY ###\n{formatted_history}\n### CONTEXT ###\n{context}\n### QUESTION ###\n{question}"
    
    # Generate response
    response = client.models.generate_content(
        model=LLM_MODEL_NAME,
        contents=prompt
    )
    
    return response.text.strip()


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Format chat history for display or logging
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
    
    Returns:
        Formatted string representation of chat history
    """
    if not messages:
        return "No previous conversation"
    
    formatted = []
    for msg in messages:
        role = msg.get('role', 'unknown').capitalize()
        content = msg.get('content', '')
        formatted.append(f"{role}: {content}")
    
    return "\n".join(formatted)
