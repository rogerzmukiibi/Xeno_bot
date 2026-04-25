"""
Logging module for XENO Bot
Handles CSV logging for responses and timing data
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


CHAT_LOG_DIR = os.environ.get("CHAT_LOG_DIR", "chats")
RESPONSE_LOG_PATH = os.path.join(CHAT_LOG_DIR, "responses.csv")
TIMING_LOG_PATH = os.path.join(CHAT_LOG_DIR, "timing.csv")

RESPONSE_HEADERS = [
    "Timestamp",
    "Session_ID",
    "Question",
    "Answer",
    "Source_IDs",
    "Knowledge_Q1",
    "Knowledge_A1",
    "Knowledge_Q2",
    "Knowledge_A2",
]

TIMING_HEADERS = [
    "Timestamp",
    "Session_ID",
    "Question",
    "Total_Time_MS",
    "Intent_Classification_MS",
    "Memory_Retrieval_MS",
    "RAG_Retrieval_MS",
    "Embedding_Generation_MS",
    "Similarity_Calculation_MS",
    "Context_Processing_MS",
    "LLM_Generation_MS",
    "Memory_Update_MS",
    "Logging_MS",
    "Error_Step",
    "Notes",
]

def _append_csv_row(path: str, headers: List[str], row: List):
    """Create CSV with headers if needed, then append a row."""
    try:
        os.makedirs(CHAT_LOG_DIR, exist_ok=True)

        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"Failed to append CSV row to {path}: {e}")


def log_response(
    question: str,
    answer: str,
    source_ids: str,
    knowledge_pairs: List[Tuple[str, str]],
    session_id: str,
    timer=None,
):
    """
    Log response to CSV file

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
            _log_response_impl(
                question, answer, source_ids, knowledge_pairs, session_id
            )
    else:
        _log_response_impl(question, answer, source_ids, knowledge_pairs, session_id)


def _log_response_impl(
    question: str,
    answer: str,
    source_ids: str,
    knowledge_pairs: List[Tuple[str, str]],
    session_id: str,
):
    """Internal implementation of response logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract knowledge pairs
    knowledge_question_1 = knowledge_pairs[0][0] if len(knowledge_pairs) > 0 else "N/A"
    knowledge_answer_1 = knowledge_pairs[0][1] if len(knowledge_pairs) > 0 else "N/A"
    knowledge_question_2 = knowledge_pairs[1][0] if len(knowledge_pairs) > 1 else "N/A"
    knowledge_answer_2 = knowledge_pairs[1][1] if len(knowledge_pairs) > 1 else "N/A"

    row = [
        timestamp,
        session_id,
        question,
        answer,
        source_ids,
        knowledge_question_1,
        knowledge_answer_1,
        knowledge_question_2,
        knowledge_answer_2,
    ]

    try:
        _append_csv_row(RESPONSE_LOG_PATH, RESPONSE_HEADERS, row)
        print(f"Logged response: {question} | Source IDs: {source_ids}")
    except Exception as e:
        print(f"Failed to log response: {e}")


def log_timing_data(
    question: str,
    session_id: str,
    timing_summary: Dict,
    error_step: Optional[str] = None,
    notes: Optional[str] = None,
):
    """
    Log timing data to CSV file

    Args:
        question: User's question
        session_id: Session identifier
        timing_summary: Timing summary dictionary
        error_step: Step where error occurred (if any)
        notes: Additional notes
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_times = timing_summary["step_times"]

    # Truncate long questions
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
        notes or "",
    ]

    try:
        _append_csv_row(TIMING_LOG_PATH, TIMING_HEADERS, row)
        print(f"Logged timing data: Total {timing_summary['total_time_ms']}ms")
    except Exception as e:
        print(f"Failed to log timing data: {e}")


