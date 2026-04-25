"""
Response Generation module for XENO Bot
Handles LLM response generation
"""

from typing import Dict, List

from transformers import pipeline

from src.config import (HF_TOKEN, LLM_MAX_NEW_TOKENS, LLM_MODEL_NAME,
                        LLM_TEMPERATURE, SYSTEM_PROMPT)


_text_generator = None


def get_text_generator():
    """Lazily load and cache the local Transformers text-generation pipeline."""
    global _text_generator
    if _text_generator is None:
        _text_generator = pipeline(
            "text-generation",
            model=LLM_MODEL_NAME,
            token=HF_TOKEN,
            device_map="auto",
        )
    return _text_generator


def _call_local_llm(prompt: str) -> str:
    """Generate text locally using the Transformers pipeline."""
    generator = get_text_generator()

    outputs = generator(
        prompt,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        do_sample=True,
        temperature=LLM_TEMPERATURE,
        return_full_text=False,
    )

    text = outputs[0].get("generated_text", "") if outputs else ""

    if not text:
        raise ValueError("Local model returned an empty response")

    return text.strip()


def generate_xeno_response(
    context: str, question: str, chat_history: List[Dict[str, str]], timer=None
) -> str:
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


def _generate_response_impl(
    context: str, question: str, chat_history: List[Dict[str, str]]
) -> str:
    """Internal implementation of response generation"""
    # Format chat history
    formatted_history = (
        "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]
        )
        if chat_history
        else "None"
    )

    # Build prompt
    prompt = f"{SYSTEM_PROMPT}\n### HISTORY ###\n{formatted_history}\n### CONTEXT ###\n{context}\n### QUESTION ###\n{question}"

    return _call_local_llm(prompt)


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
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)
