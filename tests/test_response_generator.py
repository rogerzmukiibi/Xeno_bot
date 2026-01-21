"""
Unit tests for response_generator module
Tests LLM response generation functionality
"""

import unittest
from unittest.mock import patch, Mock, MagicMock
from src.response_generator import (
    generate_xeno_response,
    format_chat_history,
    _generate_response_impl,
)


class TestResponseGenerator(unittest.TestCase):
    """Test cases for response_generator module"""

    def setUp(self):
        """Set up test fixtures"""
        self.context = """Knowledge Entry 1:
Q: How do I create an account?
A: Visit our website and click Sign Up.
----------------------------------------"""

        self.question = "How can I create an account?"

        self.chat_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help you?"},
        ]

    def test_format_chat_history(self):
        """Test formatting chat history"""
        formatted = format_chat_history(self.chat_history)

        # Check format
        self.assertIn("User: Hello", formatted)
        self.assertIn("Assistant: Hi! How can I help you?", formatted)
        self.assertIn("\n", formatted)

    def test_format_chat_history_empty(self):
        """Test formatting empty chat history"""
        formatted = format_chat_history([])
        self.assertEqual(formatted, "No previous conversation")

    def test_format_chat_history_single_message(self):
        """Test formatting single message"""
        history = [{"role": "user", "content": "Hello"}]
        formatted = format_chat_history(history)
        self.assertEqual(formatted, "User: Hello")

    def test_format_chat_history_missing_fields(self):
        """Test formatting with missing fields"""
        history = [
            {"role": "user"},  # Missing content
            {"content": "Test"},  # Missing role
        ]
        formatted = format_chat_history(history)
        self.assertIn("User:", formatted)
        self.assertIn("Unknown:", formatted)

    @patch("src.response_generator.genai.GenerativeModel")
    def test_generate_response_impl(self, mock_model_class):
        """Test internal response generation implementation"""
        # Mock the model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "You can create an account by visiting our website."
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        response = _generate_response_impl(
            self.context, self.question, self.chat_history
        )

        # Verify model was initialized with correct model name
        mock_model_class.assert_called_once()

        # Verify generate_content was called
        mock_model.generate_content.assert_called_once()

        # Check response
        self.assertEqual(response, "You can create an account by visiting our website.")

    @patch("src.response_generator.genai.GenerativeModel")
    def test_generate_response_with_empty_history(self, mock_model_class):
        """Test generating response with empty history"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        response = _generate_response_impl(self.context, self.question, [])

        # Verify it still works
        self.assertEqual(response, "Test response")

        # Check that "None" was used for history in prompt
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]
        self.assertIn("None", prompt)

    @patch("src.response_generator.genai.GenerativeModel")
    def test_prompt_structure(self, mock_model_class):
        """Test that prompt includes all necessary components"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        _generate_response_impl(self.context, self.question, self.chat_history)

        # Get the prompt that was sent
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]

        # Verify prompt structure
        self.assertIn("HISTORY", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("QUESTION", prompt)
        self.assertIn(self.context, prompt)
        self.assertIn(self.question, prompt)

    @patch("src.response_generator.genai.GenerativeModel")
    def test_generate_xeno_response_with_timer(self, mock_model_class):
        """Test generate_xeno_response with timer"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()

        response = generate_xeno_response(
            self.context, self.question, self.chat_history, timer=mock_timer
        )

        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("llm_generation")

        # Verify response
        self.assertEqual(response, "Test response")

    @patch("src.response_generator.genai.GenerativeModel")
    def test_response_text_stripping(self, mock_model_class):
        """Test that response text is stripped of whitespace"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "  Test response with spaces  \n"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        response = _generate_response_impl(self.context, self.question, [])

        # Should be stripped
        self.assertEqual(response, "Test response with spaces")

    @patch("src.response_generator.genai.GenerativeModel")
    def test_system_prompt_inclusion(self, mock_model_class):
        """Test that system prompt is included in generated prompt"""
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Test"
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model

        _generate_response_impl(self.context, self.question, [])

        # Get the prompt
        call_args = mock_model.generate_content.call_args
        prompt = call_args[0][0]

        # Should contain system prompt text
        self.assertIn("XENO Support Assistant", prompt)


if __name__ == "__main__":
    unittest.main()
