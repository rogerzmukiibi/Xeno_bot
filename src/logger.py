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

from config import (
    GOOGLE_SHEETS_CREDENTIALS_PATH,  # This is now a PATH, not env var name
    SPREADSHEET_NAME,
    RESPONSE_SHEET_INDEX,
    TIMING_SHEET_NAME
)

# ---------------------------------------------------------------------
# Credentials - FIXED VERSION
# ---------------------------------------------------------------------

def get_google_sheets_credentials() -> Optional[Credentials]:
    """Load Google Sheets credentials from file path"""
    try:
        # Check if path exists
        if not GOOGLE_SHEETS_CREDENTIALS_PATH:
            print("⚠️  GOOGLE_SHEETS_CREDENTIALS_PATH not configured")
            return None
        
        if not os.path.exists(GOOGLE_SHEETS_CREDENTIALS_PATH):
            print(f"⚠️  Google Sheets credentials file not found: {GOOGLE_SHEETS_CREDENTIALS_PATH}")
            return None
        
        # Load from file
        with open(GOOGLE_SHEETS_CREDENTIALS_PATH, 'r', encoding='utf-8') as f:
            credentials_dict = json.load(f)
        
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive"
        ]
        return Credentials.from_service_account_info(credentials_dict, scopes=scope)
        
    except Exception as e:
        print(f"⚠️  Failed to load Google Sheets credentials: {e}")
        return None


# ---------------------------------------------------------------------
# Sheet initialization - FIXED VERSION
# ---------------------------------------------------------------------

def initialize_sheets():
    """Initialize Google Sheets with graceful fallback"""
    try:
        creds = get_google_sheets_credentials()
        
        if not creds:
            print("⚠️  Google Sheets disabled - no credentials available")
            return None, None
        
        client = gspread.authorize(creds)
        spreadsheet = client.open(SPREADSHEET_NAME)

        response_sheet = spreadsheet.get_worksheet(RESPONSE_SHEET_INDEX)

        try:
            timing_sheet = spreadsheet.worksheet(TIMING_SHEET_NAME)
        except Exception:
            timing_sheet = spreadsheet.add_worksheet(
                title=TIMING_SHEET_NAME,
                rows="1000",
                cols="20"
            )
            headers = [
                "Timestamp", "Session_ID", "Question", "Total_Time_MS",
                "Intent_Classification_MS", "Memory_Retrieval_MS", "RAG_Retrieval_MS",
                "Embedding_Generation_MS", "Similarity_Calculation_MS",
                "Context_Processing_MS", "LLM_Generation_MS",
                "Memory_Update_MS", "Response_Logging_MS",
                "Error_Step", "Notes"
            ]
            timing_sheet.append_row(headers)

        print("✅ Google Sheets initialized successfully")
        return response_sheet, timing_sheet
        
    except Exception as e:
        print(f"⚠️  Google Sheets initialization failed: {e}")
        return None, None


# Initialize sheets with graceful fallback
try:
    response_sheet, timing_sheet = initialize_sheets()
    if not response_sheet:
        print("⚠️  Google Sheets logging disabled - using local fallback")
except Exception as e:
    print(f"⚠️  Failed to initialize Google Sheets: {e}")
    response_sheet, timing_sheet = None, None

# ---------------------------------------------------------------------
# Response logging - UPDATED with fallback
# ---------------------------------------------------------------------

def log_response(
    question: str,
    answer: str,
    source_ids: str,
    knowledge_pairs: List[Tuple[str, str]],
    session_id: str,
    user_node_id: Optional[str] = None,
    memory_node_id: Optional[str] = None,
    timer=None
):
    if timer:
        with timer.time_step("response_logging"):
            _log_response_impl(
                question,
                answer,
                source_ids,
                knowledge_pairs,
                session_id,
                user_node_id,
                memory_node_id
            )
    else:
        _log_response_impl(
            question,
            answer,
            source_ids,
            knowledge_pairs,
            session_id,
            user_node_id,
            memory_node_id
        )


def _log_response_impl(
    question: str,
    answer: str,
    source_ids: str,
    knowledge_pairs: List[Tuple[str, str]],
    session_id: str,
    user_node_id: Optional[str],
    memory_node_id: Optional[str]
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    kq1, ka1 = knowledge_pairs[0] if len(knowledge_pairs) > 0 else ("N/A", "N/A")
    kq2, ka2 = knowledge_pairs[1] if len(knowledge_pairs) > 1 else ("N/A", "N/A")

    row = [
        timestamp,
        session_id,
        question,
        answer,
        source_ids,
        kq1,
        ka1,
        kq2,
        ka2,
        user_node_id or "N/A",
        memory_node_id or "N/A"
    ]

    # Try Google Sheets first, fall back to local file
    if response_sheet:
        try:
            response_sheet.append_row(row)
            print(f"✅ Logged response to Google Sheets | Session: {session_id}")
            return
        except Exception as e:
            print(f"⚠️  Google Sheets logging failed, falling back to local: {e}")
    
    # Local fallback
    try:
        log_file = "response_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Timestamp,Session_ID,Question,Answer,Source_IDs,Knowledge_Q1,Knowledge_A1,Knowledge_Q2,Knowledge_A2,User_Node_ID,Memory_Node_ID\n")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(",".join(['"' + str(item).replace('"', '""') + '"' for item in row]) + "\n")
        
        print(f"📝 Logged response locally to {log_file} | Session: {session_id}")
    except Exception as e:
        print(f"❌ All logging failed: {e}")


# ---------------------------------------------------------------------
# Timing logging - UPDATED with fallback
# ---------------------------------------------------------------------

def log_timing_data(
    question: str,
    session_id: str,
    timing_summary: Dict,
    error_step: Optional[str] = None,
    notes: Optional[str] = None
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_times = timing_summary["step_times"]

    truncated_question = question[:100] + "..." if len(question) > 100 else question

    row = [
        timestamp,
        session_id,
        truncated_question,
        timing_summary["total_time_ms"],
        step_times.get("intent_classification", 0),
        step_times.get("memory_retrieval", 0),
        step_times.get("rag_retrieval", 0),
        step_times.get("embedding_generation", 0),
        step_times.get("similarity_calculation", 0),
        step_times.get("context_processing", 0),
        step_times.get("llm_generation", 0),
        step_times.get("memory_update", 0),
        step_times.get("response_logging", 0),
        error_step or "",
        notes or ""
    ]

    # Try Google Sheets first, fall back to local file
    if timing_sheet:
        try:
            timing_sheet.append_row(row)
            print(f"⏱️ Timing logged to Google Sheets | Total: {timing_summary['total_time_ms']}ms")
            return
        except Exception as e:
            print(f"⚠️  Google Sheets timing logging failed, falling back to local: {e}")
    
    # Local fallback
    try:
        log_file = "timing_log.csv"
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("Timestamp,Session_ID,Question,Total_Time_MS,Intent_Classification_MS,Memory_Retrieval_MS,RAG_Retrieval_MS,Embedding_Generation_MS,Similarity_Calculation_MS,Context_Processing_MS,LLM_Generation_MS,Memory_Update_MS,Response_Logging_MS,Error_Step,Notes\n")
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(",".join(['"' + str(item).replace('"', '""') + '"' for item in row]) + "\n")
        
        print(f"📝 Timing logged locally to {log_file}")
    except Exception as e:
        print(f"❌ All timing logging failed: {e}")