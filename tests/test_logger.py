"""
Unit tests for logger module
Tests Google Sheets logging functionality with fallback
"""
import unittest
import os
import sys
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, mock_open

# Add the parent directory to sys.path to find src module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the config module before importing logger
config_mock = Mock()
config_mock.GOOGLE_SHEETS_CREDENTIALS_PATH = '/fake/path.json'
config_mock.SPREADSHEET_NAME = 'Test Spreadsheet'
config_mock.RESPONSE_SHEET_INDEX = 0
config_mock.TIMING_SHEET_NAME = 'Timing Data'

sys.modules['config'] = config_mock

# Now import the logger module
from src.logger import (
    get_google_sheets_credentials,
    initialize_sheets,
    log_response,
    log_timing_data,
    _log_response_impl
)


class TestLogger(unittest.TestCase):
    """Test cases for logger module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.question = "How do I create an account?"
        self.answer = "You can create an account by visiting our website."
        self.source_ids = "KB001, KB002"
        self.knowledge_pairs = [
            ("Question 1?", "Answer 1."),
            ("Question 2?", "Answer 2.")
        ]
        self.session_id = "test_session_123"
        self.user_node_id = "user_456"
        self.memory_node_id = "memory_789"
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up any created files
        for log_file in ['response_log.csv', 'timing_log.csv']:
            if os.path.exists(log_file):
                os.remove(log_file)
    
    def test_get_google_sheets_credentials_missing_path(self):
        """Test credentials loading with missing path"""
        with patch('src.logger.GOOGLE_SHEETS_CREDENTIALS_PATH', None):
            creds = get_google_sheets_credentials()
            self.assertIsNone(creds)
    
    @patch('os.path.exists')
    def test_get_google_sheets_credentials_file_not_found(self, mock_exists):
        """Test credentials loading when file doesn't exist"""
        with patch('src.logger.GOOGLE_SHEETS_CREDENTIALS_PATH', '/fake/path.json'):
            mock_exists.return_value = False
            creds = get_google_sheets_credentials()
            self.assertIsNone(creds)
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='{"test": "credentials"}')
    @patch('src.logger.Credentials.from_service_account_info')
    def test_get_google_sheets_credentials_success(self, mock_from_info, mock_file, mock_exists):
        """Test successful credentials loading"""
        with patch('src.logger.GOOGLE_SHEETS_CREDENTIALS_PATH', '/real/path.json'):
            mock_exists.return_value = True
            mock_from_info.return_value = Mock()
            
            creds = get_google_sheets_credentials()
            self.assertIsNotNone(creds)
            mock_file.assert_called_once_with('/real/path.json', 'r', encoding='utf-8')
    
    @patch('src.logger.get_google_sheets_credentials')
    @patch('src.logger.gspread.authorize')
    def test_initialize_sheets_success(self, mock_authorize, mock_get_creds):
        """Test successful sheets initialization"""
        mock_creds = Mock()
        mock_get_creds.return_value = mock_creds
        mock_client = Mock()
        mock_authorize.return_value = mock_client
        mock_spreadsheet = Mock()
        mock_client.open.return_value = mock_spreadsheet
        
        # Mock worksheet and add_worksheet
        mock_response_sheet = Mock()
        mock_timing_sheet = Mock()
        mock_spreadsheet.get_worksheet.return_value = mock_response_sheet
        mock_spreadsheet.worksheet.side_effect = Exception("Sheet not found")  # Force creation
        mock_spreadsheet.add_worksheet.return_value = mock_timing_sheet
        
        response_sheet, timing_sheet = initialize_sheets()
        
        self.assertIsNotNone(response_sheet)
        self.assertIsNotNone(timing_sheet)
        mock_spreadsheet.add_worksheet.assert_called_once()
    
    @patch('src.logger.get_google_sheets_credentials')
    def test_initialize_sheets_no_credentials(self, mock_get_creds):
        """Test sheets initialization without credentials"""
        mock_get_creds.return_value = None
        response_sheet, timing_sheet = initialize_sheets()
        self.assertIsNone(response_sheet)
        self.assertIsNone(timing_sheet)
    
    @patch('src.logger.response_sheet')
    def test_log_response_impl_with_all_fields(self, mock_sheet):
        """Test internal response logging implementation with all fields"""
        _log_response_impl(
            question=self.question,
            answer=self.answer,
            source_ids=self.source_ids,
            knowledge_pairs=self.knowledge_pairs,
            session_id=self.session_id,
            user_node_id=self.user_node_id,
            memory_node_id=self.memory_node_id
        )
        
        # Verify append_row was called
        mock_sheet.append_row.assert_called_once()
        
        # Check the row data
        row = mock_sheet.append_row.call_args[0][0]
        
        # Verify row structure (11 fields now with user_node_id and memory_node_id)
        self.assertEqual(len(row), 11)
        self.assertEqual(row[1], self.session_id)
        self.assertEqual(row[2], self.question)
        self.assertEqual(row[3], self.answer)
        self.assertEqual(row[4], self.source_ids)
        self.assertEqual(row[5], "Question 1?")
        self.assertEqual(row[6], "Answer 1.")
        self.assertEqual(row[7], "Question 2?")
        self.assertEqual(row[8], "Answer 2.")
        self.assertEqual(row[9], self.user_node_id)
        self.assertEqual(row[10], self.memory_node_id)
    
    @patch('src.logger.response_sheet')
    def test_log_response_impl_without_optional_ids(self, mock_sheet):
        """Test logging without optional node IDs"""
        _log_response_impl(
            question=self.question,
            answer=self.answer,
            source_ids=self.source_ids,
            knowledge_pairs=self.knowledge_pairs,
            session_id=self.session_id,
            user_node_id=None,
            memory_node_id=None
        )
        
        row = mock_sheet.append_row.call_args[0][0]
        self.assertEqual(row[9], "N/A")
        self.assertEqual(row[10], "N/A")
    
    @patch('src.logger.response_sheet')
    def test_log_response_with_timer(self, mock_sheet):
        """Test log_response with timer"""
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        log_response(
            question=self.question,
            answer=self.answer,
            source_ids=self.source_ids,
            knowledge_pairs=self.knowledge_pairs,
            session_id=self.session_id,
            user_node_id=self.user_node_id,
            memory_node_id=self.memory_node_id,
            timer=mock_timer
        )
        
        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("response_logging")
        mock_sheet.append_row.assert_called_once()
    
    @patch('src.logger.response_sheet')
    def test_log_response_empty_knowledge_pairs(self, mock_sheet):
        """Test logging with empty knowledge pairs"""
        _log_response_impl(
            question=self.question,
            answer=self.answer,
            source_ids=self.source_ids,
            knowledge_pairs=[],
            session_id=self.session_id,
            user_node_id=None,
            memory_node_id=None
        )
        
        row = mock_sheet.append_row.call_args[0][0]
        self.assertEqual(row[5], "N/A")
        self.assertEqual(row[6], "N/A")
        self.assertEqual(row[7], "N/A")
        self.assertEqual(row[8], "N/A")
    
    @patch('src.logger.response_sheet')
    def test_log_response_single_knowledge_pair(self, mock_sheet):
        """Test logging with single knowledge pair"""
        single_pair = [("Single question?", "Single answer.")]
        
        _log_response_impl(
            question=self.question,
            answer=self.answer,
            source_ids=self.source_ids,
            knowledge_pairs=single_pair,
            session_id=self.session_id,
            user_node_id=None,
            memory_node_id=None
        )
        
        row = mock_sheet.append_row.call_args[0][0]
        self.assertEqual(row[5], "Single question?")
        self.assertEqual(row[6], "Single answer.")
        self.assertEqual(row[7], "N/A")
        self.assertEqual(row[8], "N/A")
    
    @patch('src.logger.response_sheet')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_log_response_google_sheets_fallback(self, mock_exists, mock_file, mock_sheet):
        """Test fallback to CSV when Google Sheets fails"""
        # Make Google Sheets fail
        mock_sheet.append_row.side_effect = Exception("Connection error")
        mock_exists.return_value = False  # File doesn't exist initially
        
        _log_response_impl(
            question=self.question,
            answer=self.answer,
            source_ids=self.source_ids,
            knowledge_pairs=self.knowledge_pairs,
            session_id=self.session_id,
            user_node_id=None,
            memory_node_id=None
        )
        
        # Should write header first (since file doesn't exist)
        mock_file.assert_any_call('response_log.csv', 'w', encoding='utf-8')
        # Then append data
        mock_file.assert_any_call('response_log.csv', 'a', encoding='utf-8')
    
    @patch('src.logger.timing_sheet')
    def test_log_timing_data(self, mock_sheet):
        """Test timing data logging"""
        timing_summary = {
            "total_time_ms": 1500,
            "step_times": {
                "intent_classification": 50,
                "memory_retrieval": 100,
                "rag_retrieval": 200,
                "embedding_generation": 300,
                "similarity_calculation": 150,
                "context_processing": 100,
                "llm_generation": 500,
                "memory_update": 50,
                "response_logging": 50
            }
        }
        
        log_timing_data(
            question=self.question,
            session_id=self.session_id,
            timing_summary=timing_summary,
            error_step=None,
            notes="Test note"
        )
        
        # Verify append_row was called
        mock_sheet.append_row.assert_called_once()
        
        # Check row structure (15 fields)
        row = mock_sheet.append_row.call_args[0][0]
        self.assertEqual(len(row), 15)
        self.assertEqual(row[1], self.session_id)
        self.assertEqual(row[2], self.question)  # Not truncated
        self.assertEqual(row[3], 1500)  # total_time_ms
        self.assertEqual(row[4], 50)    # intent_classification
        self.assertEqual(row[14], "Test note")  # notes
    
    @patch('src.logger.timing_sheet')
    def test_log_timing_data_long_question(self, mock_sheet):
        """Test timing data logging with long question (truncation)"""
        long_question = "A" * 150  # 150 characters
        
        timing_summary = {
            "total_time_ms": 100,
            "step_times": {}
        }
        
        log_timing_data(
            question=long_question,
            session_id=self.session_id,
            timing_summary=timing_summary
        )
        
        row = mock_sheet.append_row.call_args[0][0]
        # Question should be truncated to 103 chars (100 + "...")
        self.assertEqual(len(row[2]), 103)
        self.assertTrue(row[2].endswith("..."))
    
    @patch('src.logger.timing_sheet')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_log_timing_data_google_sheets_fallback(self, mock_exists, mock_file, mock_sheet):
        """Test fallback to CSV for timing data when Google Sheets fails"""
        # Make Google Sheets fail
        mock_sheet.append_row.side_effect = Exception("Connection error")
        mock_exists.return_value = False  # File doesn't exist initially
        
        timing_summary = {
            "total_time_ms": 100,
            "step_times": {}
        }
        
        log_timing_data(
            question=self.question,
            session_id=self.session_id,
            timing_summary=timing_summary
        )
        
        # Should write header first (since file doesn't exist)
        mock_file.assert_any_call('timing_log.csv', 'w', encoding='utf-8')
        # Then append data
        mock_file.assert_any_call('timing_log.csv', 'a', encoding='utf-8')
    
    @patch('src.logger.timing_sheet')
    def test_log_timing_data_with_error_step(self, mock_sheet):
        """Test timing data logging with error step"""
        timing_summary = {
            "total_time_ms": 500,
            "step_times": {
                "intent_classification": 50
            }
        }
        
        log_timing_data(
            question=self.question,
            session_id=self.session_id,
            timing_summary=timing_summary,
            error_step="rag_retrieval",
            notes="Error occurred"
        )
        
        row = mock_sheet.append_row.call_args[0][0]
        self.assertEqual(row[13], "rag_retrieval")
        self.assertEqual(row[14], "Error occurred")
    
    @patch('src.logger.timing_sheet')
    def test_log_timing_data_missing_steps(self, mock_sheet):
        """Test timing data with missing step times"""
        timing_summary = {
            "total_time_ms": 100,
            "step_times": {
                "intent_classification": 100
                # Other steps missing
            }
        }
        
        log_timing_data(
            question=self.question,
            session_id=self.session_id,
            timing_summary=timing_summary
        )
        
        row = mock_sheet.append_row.call_args[0][0]
        # Missing steps should default to 0
        self.assertEqual(row[5], 0)  # memory_retrieval
        self.assertEqual(row[6], 0)  # rag_retrieval
    
    def test_log_response_local_fallback_creates_file(self):
        """Test that local fallback actually creates CSV file"""
        # Temporarily set response_sheet to None to force fallback
        original_response_sheet = sys.modules['src.logger'].response_sheet
        sys.modules['src.logger'].response_sheet = None
        
        try:
            # Remove file if it exists
            if os.path.exists('response_log.csv'):
                os.remove('response_log.csv')
            
            _log_response_impl(
                question=self.question,
                answer=self.answer,
                source_ids=self.source_ids,
                knowledge_pairs=self.knowledge_pairs,
                session_id=self.session_id,
                user_node_id=None,
                memory_node_id=None
            )
            
            # Check file was created
            self.assertTrue(os.path.exists('response_log.csv'))
            
            # Check file content
            with open('response_log.csv', 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn(self.question, content)
                self.assertIn(self.answer, content)
                self.assertIn(self.session_id, content)
                
        finally:
            # Restore original
            sys.modules['src.logger'].response_sheet = original_response_sheet
            
            # Clean up
            if os.path.exists('response_log.csv'):
                os.remove('response_log.csv')


if __name__ == '__main__':
    unittest.main()