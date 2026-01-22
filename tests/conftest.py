"""
Pytest configuration file - UPDATED VERSION
Sets up test environment and fixtures
"""
import os
import sys
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ==================== MOCK ENVIRONMENT VARIABLES ====================
# Set mock environment variables for testing
os.environ.update({
    'GEMINI_API_KEY': 'test-api-key-12345',
    'NEO4J_URI': 'bolt://localhost:7687',
    'NEO4J_USER': 'neo4j',
    'NEO4J_PASSWORD': 'test-password',
    'GOOGLE_SHEETS_CREDENTIALS_PATH': 'test_credentials.json'
})

# ==================== MOCK DOTENV ====================
# Mock dotenv.load_dotenv to avoid loading actual .env file
@pytest.fixture(autouse=True)
def mock_dotenv():
    """Mock dotenv to avoid loading actual .env file"""
    with patch('src.config.load_dotenv') as mock_load:
        mock_load.return_value = True
        yield mock_load

# ==================== MOCK GOOGLE SHEETS ====================
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
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test"
}

# Mock file operations for Google Sheets credentials
@pytest.fixture(autouse=True)
def mock_file_operations():
    """Mock file operations for Google Sheets credentials"""
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open(read_data=json.dumps(mock_credentials))):
        # Always return True for path exists
        mock_exists.return_value = True
        yield

# ==================== MOCK GOOGLE AUTH ====================
# Create mocks for Google authentication modules
mock_service_account = MagicMock()
mock_service_account.Credentials = Mock()
mock_service_account.Credentials.from_service_account_info = Mock()

mock_oauth2 = MagicMock()
mock_oauth2.service_account = mock_service_account

# Patch the modules before they're imported
sys.modules['google.oauth2'] = mock_oauth2
sys.modules['google.oauth2.service_account'] = mock_service_account

# ==================== MOCK GSPREAD ====================
mock_gspread = MagicMock()
sys.modules['gspread'] = mock_gspread

# ==================== FIXTURES ====================
@pytest.fixture(autouse=True)
def mock_google_sheets():
    """
    Mock Google Sheets to avoid actual connections during testing
    """
    with patch('google.oauth2.service_account.Credentials') as mock_creds_class:
        mock_creds_instance = Mock()
        mock_creds_class.from_service_account_info.return_value = mock_creds_instance
        
        with patch('gspread.authorize') as mock_authorize:
            mock_client = Mock()
            mock_spreadsheet = Mock()
            mock_worksheet = Mock()
            mock_worksheet.append_row = Mock()
            
            mock_spreadsheet.get_worksheet.return_value = mock_worksheet
            mock_spreadsheet.worksheet.return_value = mock_worksheet
            mock_client.open.return_value = mock_spreadsheet
            mock_authorize.return_value = mock_client
            
            # Patch logger module's sheet objects
            with patch('src.logger.response_sheet', new=mock_worksheet) as mock_response, \
                 patch('src.logger.timing_sheet', new=mock_worksheet) as mock_timing:
                
                yield mock_response, mock_timing


@pytest.fixture
def mock_genai():
    """Mock Google Generative AI"""
    with patch('google.generativeai.configure') as mock_config, \
         patch('google.generativeai.GenerativeModel') as mock_model, \
         patch('google.generativeai.embed_content') as mock_embed:
        yield {
            'configure': mock_config,
            'model': mock_model,
            'embed': mock_embed
        }


@pytest.fixture
def mock_chromadb():
    """Mock ChromaDB client"""
    with patch('chromadb.PersistentClient') as mock_client:
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        yield mock_client


@pytest.fixture
def mock_sqlite():
    """Mock SQLite connections for memory"""
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        yield mock_conn


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing"""
    doc1 = Mock()
    doc1.page_content = "Question: How do I create an account?\nAnswer: Visit our website."
    doc1.metadata = {
        'id': 'KB001',
        'question': 'How do I create an account?',
        'content': 'Visit our website.',
        'section': 'Account Management'
    }
    
    doc2 = Mock()
    doc2.page_content = "Question: What are the fees?\nAnswer: 1% per transaction."
    doc2.metadata = {
        'id': 'KB002',
        'question': 'What are the fees?',
        'content': '1% per transaction.',
        'section': 'Fees'
    }
    
    return [doc1, doc2]


@pytest.fixture(autouse=True)
def mock_knowledge_base_path():
    """Mock knowledge base path to always exist"""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        yield


# ==================== CLEANUP ====================
@pytest.fixture(autouse=True)
def cleanup_imports():
    """Clean up module imports between tests"""
    original_modules = dict(sys.modules)
    yield
    # Remove any modules that were imported during the test
    new_modules = set(sys.modules.keys()) - set(original_modules.keys())
    for module in new_modules:
        if module.startswith('src.'):
            del sys.modules[module]