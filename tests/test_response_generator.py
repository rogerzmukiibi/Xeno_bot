"""
Unit tests for response_generator module
Tests LLM response generation functionality
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from src.response_generator import (_generate_response_impl,
                                    format_chat_history,
                                    generate_xeno_response)


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

    @patch("src.response_generator.get_text_generator")
    def test_generate_response_impl(self, mock_get_generator):
        """Test internal response generation implementation"""
        mock_generator = Mock()
        mock_generator.return_value = [
            {"generated_text": "You can create an account by visiting our website."}
        ]
        mock_get_generator.return_value = mock_generator

        response = _generate_response_impl(
            self.context, self.question, self.chat_history
        )

        # Verify local transformers generator call
        mock_generator.assert_called_once()
        call_args = mock_generator.call_args[0]
        self.assertTrue(len(call_args) >= 1)

        # Check response
        self.assertEqual(response, "You can create an account by visiting our website.")

    @patch("src.response_generator.get_text_generator")
    def test_generate_response_with_empty_history(self, mock_get_generator):
        """Test generating response with empty history"""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "Test response"}]
        mock_get_generator.return_value = mock_generator

        response = _generate_response_impl(self.context, self.question, [])

        # Verify it still works
        self.assertEqual(response, "Test response")

        # Check that "None" was used for history in prompt
        prompt = mock_generator.call_args[0][0]
        self.assertIn("None", prompt)

    @patch("src.response_generator.get_text_generator")
    def test_prompt_structure(self, mock_get_generator):
        """Test that prompt includes all necessary components"""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "Test response"}]
        mock_get_generator.return_value = mock_generator

        _generate_response_impl(self.context, self.question, self.chat_history)

        # Get the prompt that was sent
        prompt = mock_generator.call_args[0][0]

        # Verify prompt structure
        self.assertIn("HISTORY", prompt)
        self.assertIn("CONTEXT", prompt)
        self.assertIn("QUESTION", prompt)
        self.assertIn(self.context, prompt)
        self.assertIn(self.question, prompt)

    @patch("src.response_generator.get_text_generator")
    def test_generate_xeno_response_with_timer(self, mock_get_generator):
        """Test generate_xeno_response with timer"""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "Test response"}]
        mock_get_generator.return_value = mock_generator

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

    @patch("src.response_generator.get_text_generator")
    def test_response_text_stripping(self, mock_get_generator):
        """Test that response text is stripped of whitespace"""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "  Test response with spaces  "}]
        mock_get_generator.return_value = mock_generator

        response = _generate_response_impl(self.context, self.question, [])

        # Response should be stripped
        self.assertEqual(response, "Test response with spaces")

    @patch("src.response_generator.get_text_generator")
    def test_system_prompt_inclusion(self, mock_get_generator):
        """Test that system prompt is included in generated prompt"""
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "Test"}]
        mock_get_generator.return_value = mock_generator

        _generate_response_impl(self.context, self.question, [])

        # Get the prompt
        prompt = mock_generator.call_args[0][0]

        # Should contain system prompt text
        self.assertIn("XENO Support Assistant", prompt)


if __name__ == "__main__":
    unittest.main()
