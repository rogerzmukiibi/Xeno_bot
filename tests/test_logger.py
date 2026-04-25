"""
Unit tests for logger module
Tests CSV logging functionality
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from src.logger import _log_response_impl, log_response, log_timing_data


class TestLogger(unittest.TestCase):
    """Test cases for logger module"""

    def setUp(self):
        """Set up test fixtures"""
        self.question = "How do I create an account?"
        self.answer = "You can create an account by visiting our website."
        self.source_ids = "KB001, KB002"
        self.knowledge_pairs = [
            ("Question 1?", "Answer 1."),
            ("Question 2?", "Answer 2."),
        ]
        self.session_id = "test_session_123"

    @patch("src.logger._append_csv_row")
    def test_log_response_impl(self, mock_append):
        """Test internal response logging implementation"""
        _log_response_impl(
            self.question,
            self.answer,
            self.source_ids,
            self.knowledge_pairs,
            self.session_id,
        )

        # Verify CSV append was called
        mock_append.assert_called_once()

        # Check the row data
        call_args = mock_append.call_args
        row = call_args[0][2]

        # Verify row structure
        self.assertEqual(
            len(row), 9
        )  # timestamp, session_id, question, answer, source_ids, 4 knowledge fields
        self.assertEqual(row[1], self.session_id)
        self.assertEqual(row[2], self.question)
        self.assertEqual(row[3], self.answer)
        self.assertEqual(row[4], self.source_ids)
        self.assertEqual(row[5], "Question 1?")
        self.assertEqual(row[6], "Answer 1.")
        self.assertEqual(row[7], "Question 2?")
        self.assertEqual(row[8], "Answer 2.")

    @patch("src.logger._append_csv_row")
    def test_log_response_with_timer(self, mock_append):
        """Test log_response with timer"""
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()

        log_response(
            self.question,
            self.answer,
            self.source_ids,
            self.knowledge_pairs,
            self.session_id,
            timer=mock_timer,
        )

        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("response_logging")

    @patch("src.logger._append_csv_row")
    def test_log_response_empty_knowledge_pairs(self, mock_append):
        """Test logging with empty knowledge pairs"""
        _log_response_impl(
            self.question, self.answer, self.source_ids, [], self.session_id
        )

        # Should still work
        mock_append.assert_called_once()

        # Check that N/A is used for missing pairs
        row = mock_append.call_args[0][2]
        self.assertEqual(row[5], "N/A")
        self.assertEqual(row[6], "N/A")

    @patch("src.logger._append_csv_row")
    def test_log_response_single_knowledge_pair(self, mock_append):
        """Test logging with single knowledge pair"""
        single_pair = [("Single question?", "Single answer.")]

        _log_response_impl(
            self.question, self.answer, self.source_ids, single_pair, self.session_id
        )

        row = mock_append.call_args[0][2]

        # First pair should be present
        self.assertEqual(row[5], "Single question?")
        self.assertEqual(row[6], "Single answer.")

        # Second pair should be N/A
        self.assertEqual(row[7], "N/A")
        self.assertEqual(row[8], "N/A")

    @patch("src.logger._append_csv_row")
    def test_log_response_append_error_is_handled(self, mock_append):
        """Test that append errors do not break response logging"""
        mock_append.side_effect = Exception("Write error")

        _log_response_impl(
            self.question,
            self.answer,
            self.source_ids,
            self.knowledge_pairs,
            self.session_id,
        )

    @patch("src.logger._append_csv_row")
    def test_log_timing_data(self, mock_append):
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
                "response_logging": 50,
            },
        }

        log_timing_data(
            self.question,
            self.session_id,
            timing_summary,
            error_step=None,
            notes="Test note",
        )

        # Verify CSV append was called
        mock_append.assert_called_once()

        # Check row structure
        row = mock_append.call_args[0][2]

        # Should have 15 fields
        self.assertEqual(len(row), 15)
        self.assertEqual(row[1], self.session_id)
        self.assertEqual(row[3], 1500)  # total_time_ms
        self.assertEqual(row[4], 50)  # intent_classification
        self.assertEqual(row[5], 100)  # memory_retrieval
        self.assertEqual(row[14], "Test note")  # notes

    @patch("src.logger._append_csv_row")
    def test_log_timing_data_with_error(self, mock_append):
        """Test timing data logging with error"""
        timing_summary = {
            "total_time_ms": 500,
            "step_times": {"intent_classification": 50},
        }

        log_timing_data(
            self.question,
            self.session_id,
            timing_summary,
            error_step="rag_retrieval",
            notes="Error occurred",
        )

        row = mock_append.call_args[0][2]

        # Check error_step is logged
        self.assertEqual(row[13], "rag_retrieval")
        self.assertEqual(row[14], "Error occurred")

    @patch("src.logger._append_csv_row")
    def test_log_timing_data_missing_steps(self, mock_append):
        """Test timing data with missing step times"""
        timing_summary = {
            "total_time_ms": 100,
            "step_times": {
                "intent_classification": 100
                # Other steps missing
            },
        }

        log_timing_data(self.question, self.session_id, timing_summary)

        row = mock_append.call_args[0][2]

        # Missing steps should default to 0
        self.assertEqual(row[5], 0)  # memory_retrieval
        self.assertEqual(row[6], 0)  # rag_retrieval

    @patch("src.logger._append_csv_row")
    def test_log_timing_data_long_question(self, mock_append):
        """Test timing data logging with long question (truncation)"""
        long_question = "A" * 150  # 150 characters

        timing_summary = {"total_time_ms": 100, "step_times": {}}

        log_timing_data(long_question, self.session_id, timing_summary)

        row = mock_append.call_args[0][2]

        # Question should be truncated to 103 chars (100 + "...")
        self.assertEqual(len(row[2]), 103)
        self.assertTrue(row[2].endswith("..."))

    @patch("src.logger._append_csv_row")
    def test_log_timing_data_append_error_is_handled(self, mock_append):
        """Test that append errors do not break timing logging"""
        mock_append.side_effect = Exception("Write error")
        timing_summary = {"total_time_ms": 100, "step_times": {}}

        log_timing_data(self.question, self.session_id, timing_summary)

if __name__ == "__main__":
    unittest.main()
