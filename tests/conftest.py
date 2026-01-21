"""
Pytest configuration file
Sets up test environment and fixtures
"""

import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set mock environment variables before importing any modules
os.environ.setdefault("GEMINI_API_KEY", "test-api-key-12345")

# Mock Google Sheets credentials
mock_credentials = {
    "type": "service_account",
    "project_id": "test-project",
    "private_key_id": "test-key-id",
    "private_key": "-----BEGIN PRIVATE KEY-----\ntest\n-----END PRIVATE KEY-----\n",
    "client_email": "test@test.iam.gserviceaccount.com",
    "client_id": "12345",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test",
}
import json

os.environ.setdefault("GOOGLE_SHEETS_CREDENTIALS", json.dumps(mock_credentials))

# Mock google.oauth2 and gspread modules before src.logger imports them
mock_credentials_class = MagicMock()
mock_creds_instance = MagicMock()
mock_credentials_class.from_service_account_info = Mock(
    return_value=mock_creds_instance
)

mock_oauth2 = MagicMock()
mock_oauth2.service_account.Credentials = mock_credentials_class
sys.modules["google.oauth2"] = mock_oauth2
sys.modules["google.oauth2.service_account"] = mock_oauth2.service_account

mock_gspread = MagicMock()
mock_spreadsheet = MagicMock()
mock_worksheet = MagicMock()
mock_worksheet.append_row = Mock()
mock_spreadsheet.get_worksheet = Mock(return_value=mock_worksheet)
mock_spreadsheet.worksheet = Mock(return_value=mock_worksheet)
mock_spreadsheet.add_worksheet = Mock(return_value=mock_worksheet)
mock_client = MagicMock()
mock_client.open = Mock(return_value=mock_spreadsheet)
mock_gspread.authorize = Mock(return_value=mock_client)
sys.modules["gspread"] = mock_gspread


@pytest.fixture(autouse=True)
def mock_google_sheets():
    """Mock Google Sheets to avoid actual connections during testing"""
    with patch("src.logger.response_sheet") as mock_response, patch(
        "src.logger.timing_sheet"
    ) as mock_timing:
        mock_response.append_row = Mock()
        mock_timing.append_row = Mock()
        yield mock_response, mock_timing


@pytest.fixture
def mock_genai():
    """Mock Google Generative AI"""
    with patch("google.generativeai.configure") as mock_config, patch(
        "google.generativeai.GenerativeModel"
    ) as mock_model, patch("google.generativeai.embed_content") as mock_embed:
        yield {"configure": mock_config, "model": mock_model, "embed": mock_embed}


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
