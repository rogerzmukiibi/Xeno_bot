"""
Memory module for XENO Bot
Handles LangGraph memory operations using SQLite
"""

import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import SQLITE_DB_PATH

# === LangGraph Memory Setup ===
conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn=conn)


def update_memory(
    config: Dict[str, Any], user_message: str, assistant_message: str, timer=None
):
    """
    Update memory with new messages

    Args:
        config: Configuration dictionary with thread_id
        user_message: User's message
        assistant_message: Assistant's response
        timer: Optional timer object for tracking
    """
    if timer:
        with timer.time_step("memory_update"):
            _update_memory_impl(config, user_message, assistant_message)
    else:
        _update_memory_impl(config, user_message, assistant_message)


def _update_memory_impl(config, user_message: str, assistant_message: str):
    """Internal implementation of memory update"""
    full_checkpoint = memory.get(config) or {}
    messages = full_checkpoint.get("channel_values", {}).get("messages", [])

    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_message})

    checkpoint_to_save = {
        "v": 1,
        "id": str(uuid.uuid4()),
        "ts": datetime.now().isoformat(),
        "channel_values": {"messages": messages},
        "channel_versions": {},
        "versions_seen": {},
    }

    memory.put(config, checkpoint_to_save, {}, {})


def retrieve_memory(config: Dict[str, Any], timer=None) -> List[Dict[str, str]]:
    """
    Retrieve memory messages for a session

    Args:
        config: Configuration dictionary with thread_id
        timer: Optional timer object for tracking

    Returns:
        List of message dictionaries
    """
    if timer:
        with timer.time_step("memory_retrieval"):
            return _retrieve_memory_impl(config)
    else:
        return _retrieve_memory_impl(config)


def _retrieve_memory_impl(config) -> List[Dict[str, str]]:
    """Internal implementation of memory retrieval"""
    full_checkpoint = memory.get(config) or {}
    return full_checkpoint.get("channel_values", {}).get("messages", [])


def create_session_config(session_id: str = "default") -> Dict[str, Any]:
    """
    Create a configuration dictionary for a session

    Args:
        session_id: Unique session identifier

    Returns:
        Configuration dictionary
    """
    return {"configurable": {"thread_id": str(session_id), "checkpoint_ns": ""}}
