"""
Unit tests for interface module
Tests Gradio interface functionality
"""

import unittest
import uuid
from unittest.mock import MagicMock, Mock, patch

from src.interface import create_interface, respond


class TestInterface(unittest.TestCase):
    """Test cases for interface module"""

    def setUp(self):
        """Set up test fixtures"""
        self.message = "How do I create an account?"
        self.history = [["Previous question", "Previous answer"]]
        self.session_id = str(uuid.uuid4())
        self.mock_intent_classifier = Mock()
        self.mock_retriever = Mock()

    @patch("app.get_context_and_answer")
    def test_respond_with_session_id(self, mock_get_answer):
        """Test respond function with existing session ID"""
        mock_get_answer.return_value = "You can create an account by visiting our website."

        result_msg, result_history = respond(
            self.message,
            self.history.copy(),
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Verify get_context_and_answer was called
        mock_get_answer.assert_called_once()
        call_args = mock_get_answer.call_args[0]
        self.assertEqual(call_args[0], self.message)
        self.assertEqual(call_args[2], self.session_id)

        # Check return values
        self.assertEqual(result_msg, "")
        self.assertEqual(len(result_history), 2)
        self.assertEqual(result_history[-1][0], self.message)
        self.assertEqual(
            result_history[-1][1],
            "You can create an account by visiting our website.",
        )

    @patch("app.get_context_and_answer")
    def test_respond_without_session_id(self, mock_get_answer):
        """Test respond function generates session ID when none provided"""
        mock_get_answer.return_value = "Response"

        result_msg, result_history = respond(
            self.message,
            [],
            None,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Should have called with a generated session ID
        self.assertEqual(mock_get_answer.call_count, 1)
        call_args = mock_get_answer.call_args[0]
        generated_session_id = call_args[2]

        # Verify it's a valid UUID
        try:
            uuid.UUID(generated_session_id)
            valid_uuid = True
        except ValueError:
            valid_uuid = False

        self.assertTrue(valid_uuid)

        # Check return values
        self.assertEqual(result_msg, "")
        self.assertEqual(len(result_history), 1)

    @patch("app.get_context_and_answer")
    def test_respond_with_empty_history(self, mock_get_answer):
        """Test respond function with empty history"""
        mock_get_answer.return_value = "Test response"

        result_msg, result_history = respond(
            "Test question",
            [],
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # History should have one entry
        self.assertEqual(len(result_history), 1)
        self.assertEqual(result_history[0][0], "Test question")
        self.assertEqual(result_history[0][1], "Test response")

    @patch("app.get_context_and_answer")
    def test_respond_preserves_existing_history(self, mock_get_answer):
        """Test respond function preserves existing chat history"""
        mock_get_answer.return_value = "New response"

        initial_history = [
            ["Question 1", "Answer 1"],
            ["Question 2", "Answer 2"],
        ]

        result_msg, result_history = respond(
            "Question 3",
            initial_history.copy(),
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Should have 3 entries now
        self.assertEqual(len(result_history), 3)
        self.assertEqual(result_history[0][0], "Question 1")
        self.assertEqual(result_history[1][0], "Question 2")
        self.assertEqual(result_history[2][0], "Question 3")

    def test_create_interface_returns_blocks(self):
        """Test create_interface returns Gradio Blocks interface"""
        result = create_interface(self.mock_intent_classifier, self.mock_retriever)
        
        # Should return a Gradio Blocks object
        import gradio as gr
        self.assertIsInstance(result, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
