"""
Pytest configuration file
Sets up test environment and fixtures
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Default local model runtime endpoint for tests.
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client"""
    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        yield mock_client


@pytest.fixture
def mock_sqlite():
    """Mock SQLite connections for memory"""
    with patch("sqlite3.connect") as mock_connect:
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing"""
    doc1 = Mock()
    doc1.page_content = (
        "Question: How do I create an account?\nAnswer: Visit our website."
    )
    doc1.metadata = {
        "id": "KB001",
        "question": "How do I create an account?",
        "content": "Visit our website.",
        "section": "Account Management",
    }

    doc2 = Mock()
    doc2.page_content = "Question: What are the fees?\nAnswer: 1% per transaction."
    doc2.metadata = {
        "id": "KB002",
        "question": "What are the fees?",
        "content": "1% per transaction.",
        "section": "Fees",
    }

    return [doc1, doc2]


