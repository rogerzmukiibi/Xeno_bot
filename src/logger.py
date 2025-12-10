"""
Logging module for XENO Bot
Handles Google Sheets logging for responses and timing data
"""
import json
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import gspread
from google.oauth2.service_account import Credentials
from src.config import (
    GOOGLE_SHEETS_CREDENTIALS_ENV,
    SPREADSHEET_NAME,
    RESPONSE_SHEET_INDEX,
    TIMING_SHEET_NAME
)


def get_google_sheets_credentials() -> Credentials:
    """
    Get Google Sheets credentials from environment variable
    
    Returns:
        Google Sheets credentials object
    """
    credentials_json = os.environ.get(GOOGLE_SHEETS_CREDENTIALS_ENV)
    if not credentials_json:
        raise ValueError(f"{GOOGLE_SHEETS_CREDENTIALS_ENV} environment variable not set.")
    
    credentials_dict = json.loads(credentials_json)
    scope = [
        "https://spreadsheets.google.com/feeds", 
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(credentials_dict, scopes=scope)
    
    return creds


def initialize_sheets():
    """
    Initialize Google Sheets client and get sheets
    
    Returns:
        Tuple of (response_sheet, timing_sheet)
    """
    client_gspread = gspread.authorize(get_google_sheets_credentials())
    spreadsheet = client_gspread.open(SPREADSHEET_NAME)
    
    # Get response sheet
    response_sheet = spreadsheet.get_worksheet(RESPONSE_SHEET_INDEX)
    
    # Get or create timing sheet
    try:
        timing_sheet = spreadsheet.worksheet(TIMING_SHEET_NAME)
    except:
        # Create timing sheet if it doesn't exist
        timing_sheet = spreadsheet.add_worksheet(title=TIMING_SHEET_NAME, rows="1000", cols="15")
        # Add headers
        headers = [
            "Timestamp", "Session_ID", "Question", "Total_Time_MS",
            "Intent_Classification_MS", "Memory_Retrieval_MS", "RAG_Retrieval_MS", 
            "Embedding_Generation_MS", "Similarity_Calculation_MS", "Context_Processing_MS",
            "LLM_Generation_MS", "Memory_Update_MS", "Logging_MS", "Error_Step", "Notes"
        ]
        timing_sheet.append_row(headers)
    
    return response_sheet, timing_sheet


# Initialize sheets
response_sheet, timing_sheet = initialize_sheets()


def log_response(question: str, answer: str, source_ids: str, 
                knowledge_pairs: List[Tuple[str, str]], session_id: str, timer=None):
    """
    Log response to Google Sheets
    
    Args:
        question: User's question
        answer: Generated answer
        source_ids: Source IDs used
        knowledge_pairs: Knowledge base Q&A pairs used
        session_id: Session identifier
        timer: Optional timer object for tracking
    """
    if timer:
        with timer.time_step("response_logging"):
            _log_response_impl(question, answer, source_ids, knowledge_pairs, session_id)
    else:
        _log_response_impl(question, answer, source_ids, knowledge_pairs, session_id)


def _log_response_impl(question: str, answer: str, source_ids: str, 
                       knowledge_pairs: List[Tuple[str, str]], session_id: str):
    """Internal implementation of response logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract knowledge pairs
    knowledge_question_1 = knowledge_pairs[0][0] if len(knowledge_pairs) > 0 else "N/A"
    knowledge_answer_1 = knowledge_pairs[0][1] if len(knowledge_pairs) > 0 else "N/A"
    knowledge_question_2 = knowledge_pairs[1][0] if len(knowledge_pairs) > 1 else "N/A"
    knowledge_answer_2 = knowledge_pairs[1][1] if len(knowledge_pairs) > 1 else "N/A"
    
    row = [
        timestamp, session_id, question, answer, source_ids,
        knowledge_question_1, knowledge_answer_1, 
        knowledge_question_2, knowledge_answer_2
    ]
    
    try:
        response_sheet.append_row(row)
        print(f"Logged response: {question} | Source IDs: {source_ids}")
    except Exception as e:
        print(f"Failed to log to Google Sheet: {e}")
        # Fallback to local file
        with open("/tmp/response_log.txt", "a") as f:
            f.write(f"{timestamp},{question},{answer},{source_ids},{knowledge_question_1},{knowledge_answer_1},{knowledge_question_2},{knowledge_answer_2}\n")


def log_timing_data(question: str, session_id: str, timing_summary: Dict, 
                    error_step: Optional[str] = None, notes: Optional[str] = None):
    """
    Log timing data to Google Sheets
    
    Args:
        question: User's question
        session_id: Session identifier
        timing_summary: Timing summary dictionary
        error_step: Step where error occurred (if any)
        notes: Additional notes
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_times = timing_summary['step_times']
    
    # Truncate long questions
    truncated_question = question[:100] + "..." if len(question) > 100 else question
    
    row = [
        timestamp,
        session_id,
        truncated_question,
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
