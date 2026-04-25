import uuid
from typing import List

import gradio as gr


def respond(
    message: str, history: List, session_id: str, intent_classifier, retriever
):
    """
    Gradio's main response function
    
    Args:
        message: User's message
        history: Chat history
        session_id: Session identifier
        intent_classifier: IntentClassifier instance
        retriever: Vector store retriever instance
    
    Returns:
        Tuple of (empty string for input box, updated history)
    """
    # Import here to avoid circular imports
    from app import get_context_and_answer
    
    if not session_id:
        session_id = str(uuid.uuid4())

    bot_response = get_context_and_answer(
        message, history, session_id, intent_classifier, retriever
    )
    history.append([message, bot_response])

    return "", history


def create_interface(intent_classifier, retriever):
    """
    Create Gradio interface
    
    Args:
        intent_classifier: IntentClassifier instance
        retriever: Vector store retriever instance
    
    Returns:
        Gradio Blocks interface
    """
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
        session_id_box = gr.Textbox(
            label="Session ID", value=str(uuid.uuid4()), visible=False
        )

        chatbot = gr.Chatbot(
            label="XENO Assistant", bubble_full_width=False, height=450
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Type your question here...",
                scale=4,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        # Chat Event Listeners - Pass components to respond function
        send_button.click(
            lambda msg, chat, sid: respond(msg, chat, sid, intent_classifier, retriever),
            [msg, chatbot, session_id_box],
            [msg, chatbot],
        )
        msg.submit(
            lambda msg, chat, sid: respond(msg, chat, sid, intent_classifier, retriever),
            [msg, chatbot, session_id_box],
            [msg, chatbot],
        )

    return demo
