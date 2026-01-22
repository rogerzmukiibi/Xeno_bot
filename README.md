---
title: AskXeno
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---

# ASKXENO - AI-Powered XENO Support Assistant
## Overview
ASKXENO is an AI-powered customer support assistant designed to provide accurate and timely responses to queries about XENO financiall services. Built with a Retrieval-Augmented Generation (RAG) pipeline, it leverages a knowledge base, intent classification, and conversation memory to deliver professional and context-aware responses. The application is deployed as a Hugging Face Space and includes performance tracking for response time analysis.
Features

Natural Language Query Handling: Responds to user questions about XENO services, including account management, transactions, platform features, and general information.

Intent Classification: Detects simple intents (e.g., greetings, thanks, goodbyes) for quick, tailored responses without querying the knowledge base.
RAG Pipeline: Uses a ChromaDB vector store and Google Gemini embeddings to retrieve relevant information from a JSON-based XENO knowledge base.
Conversation Memory: Maintains chat history using LangGraph's SqliteSaver for context-aware responses.

Performance Tracking: Logs response times for each processing step (e.g., intent classification, retrieval, LLM generation) to a Google Sheet for analysis.

Gradio UI: Provides an interactive, user-friendly interface with session tracking and a chatbot-style conversation display.

Error Handling: Includes robust logging to handle and record errors, with fallback local file logging if Google Sheets fails.

## Tech Stack

**Python Libraries:**

gradio: For the web-based user interface.

pandas: For handling the JSON knowledge base.

sentence_transformers: For similarity calculations.

google.generativeai: For embeddings and LLM responses (Gemini API).

chromadb: For vector storage and retrieval.

langchain_chroma: For integrating ChromaDB with LangChain.

gspread: For logging to Google Sheets.

langgraph: For conversation memory management.

torch: For tensor operations in similarity calculations.


## Database:
SQLite for conversation memory (xeno_memory.db). [currently iys implemented in google sheets]
ChromaDB for persistent vector storage (/tmp/xeno_db).


## External Services:
Google Sheets for logging responses and timing data.
Google Gemini API for embeddings and text generation.



Setup and Installation
This project is designed to run in a Hugging Face Space. To set it up locally or in a similar environment, follow these steps:


Install Dependencies:Ensure Python 3.8+ is installed, then install the required packages:
pip install -r requirements.txt


Set Environment Variables:

GEMINI_API_KEY: Your Google Gemini API key.
GOOGLE_SHEETS_CREDENTIALS: JSON credentials for Google Sheets API access.Example:

export GEMINI_API_KEY="your-api-key"
export GOOGLE_SHEETS_CREDENTIALS='{"type": "service_account", ...}'


Prepare Knowledge Base:

Ensure XENO Uganda_KnowlegeBase_V1.json is in the project root.
The JSON file should contain entries with Question, Content, ID, and optional fields like Section, Source, Owner, and Tag.


Run the Application:
python app.py

The Gradio interface will be available at http://0.0.0.0:7860.


Usage

Access the Interface: Open the Hugging Face Space URL or local server URL in a browser.
Interact with the Chatbot:
Enter a session ID (or use the auto-generated UUID).
Type your question in the text box and click "Send" or press Enter.
The assistant responds based on the knowledge base or intent classification.


View Logs:
Response logs are stored in the "Response_Log" sheet of the configured Google Sheet.
Timing data (e.g., total time, step-wise breakdowns) is logged in the "Timing_Log" sheet.
Local fallback logs are saved in /tmp/response_log.txt and /tmp/timing_log.txt if Google Sheets fails.



Performance Tracking
The application logs detailed timing data for each request, including:

Intent classification
Memory retrieval
RAG retrieval
Embedding generation
Similarity calculation
Context processing
LLM generation
Memory update
Response logging

This data is useful for analyzing system performance and identifying bottlenecks. Check the "Timing_Log" sheet for detailed metrics.
Limitations

Knowledge Base Dependency: Responses are limited to the information in XENO Uganda_KnowlegeBase_V1.json. If no relevant information is found (similarity score < 0.4), the assistant politely declines to answer.
API Dependency: Requires a stable connection to the Google Gemini API and Google Sheets.
Hugging Face Space Constraints: Resource limitations (e.g., storage, compute) may affect performance in a shared environment.
Language Support: Currently optimized for English queries.