"""
Unit tests for app module
Tests main orchestration logic
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from app import get_context_and_answer


class TestApp(unittest.TestCase):
    """Test cases for app module"""

    def setUp(self):
        """Set up test fixtures"""
        self.message = "How do I create an account?"
        self.history = [["Previous question", "Previous answer"]]
        self.session_id = "test-session-123"
        self.mock_intent_classifier = Mock()
        self.mock_retriever = Mock()

    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_simple_intent(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
    ):
        """Test get_context_and_answer with simple intent (greeting)"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.return_value = []
        self.mock_intent_classifier.classify_intent.return_value = (
            "greeting",
            "Hello! How can I help you?",
        )

        # Call function
        answer = get_context_and_answer(
            "Hello",
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Verify intent was classified
        self.mock_intent_classifier.classify_intent.assert_called_once_with("Hello")

        # Should not use retriever for simple intent
        self.mock_retriever.invoke.assert_not_called()

        # Verify response
        self.assertEqual(answer, "Hello! How can I help you?")

        # Verify memory was updated
        mock_update_memory.assert_called_once()

        # Verify logging
        mock_log_response.assert_called_once()
        mock_log_timing.assert_called_once()

    @patch("app.generate_xeno_response")
    @patch("app.process_context")
    @patch("app.generate_embeddings")
    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_query_intent(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
        mock_generate_embeddings,
        mock_process_context,
        mock_generate_response,
    ):
        """Test get_context_and_answer with query intent"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.return_value = []
        self.mock_intent_classifier.classify_intent.return_value = ("query", None)

        # Mock retriever
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"id": "KB001", "question": "Q", "content": "A"}
        self.mock_retriever.invoke.return_value = [mock_doc]

        # Mock embeddings
        mock_generate_embeddings.return_value = (
            [0.1, 0.2, 0.3],  # query embedding
            [[0.2, 0.3, 0.4]],  # doc embeddings
        )

        # Mock context processing
        mock_process_context.return_value = (
            "Formatted context",
            ["KB001"],
            [("Q", "A")],
        )

        # Mock LLM response
        mock_generate_response.return_value = "Generated answer"

        # Call function
        answer = get_context_and_answer(
            self.message,
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Verify RAG pipeline was executed
        self.mock_retriever.invoke.assert_called_once_with(self.message)
        mock_generate_embeddings.assert_called_once()
        mock_process_context.assert_called_once()
        mock_generate_response.assert_called_once()

        # Verify response
        self.assertEqual(answer, "Generated answer")

        # Verify logging
        mock_log_response.assert_called_once()
        mock_log_timing.assert_called_once()

    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_short_message(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
    ):
        """Test get_context_and_answer with very short message"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.return_value = []
        self.mock_intent_classifier.classify_intent.return_value = ("query", None)

        # Call function with short message
        answer = get_context_and_answer(
            "Hi",
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Should return a request for more details
        self.assertIn("more details", answer)

        # Should not invoke retriever
        self.mock_retriever.invoke.assert_not_called()

    @patch("app.generate_embeddings")
    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_low_similarity(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
        mock_generate_embeddings,
    ):
        """Test get_context_and_answer with low similarity score"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.return_value = []
        self.mock_intent_classifier.classify_intent.return_value = ("query", None)

        # Mock retriever
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        self.mock_retriever.invoke.return_value = [mock_doc]

        # Mock embeddings with low similarity
        mock_generate_embeddings.return_value = (
            [0.1, 0.2, 0.3],
            [[1.0, 0.0, 0.0]],  # Will result in low cosine score
        )

        # Call function
        answer = get_context_and_answer(
            "Some random question",
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Should return "couldn't find" message
        self.assertIn("couldn't find", answer)

    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_rag_error(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
    ):
        """Test get_context_and_answer handles RAG errors gracefully"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.return_value = []
        self.mock_intent_classifier.classify_intent.return_value = ("query", None)

        # Mock retriever to raise exception
        self.mock_retriever.invoke.side_effect = Exception("Database error")

        # Call function
        answer = get_context_and_answer(
            self.message,
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Should return technical issue message
        self.assertIn("technical issue", answer)

        # Verify error was logged
        mock_log_timing.assert_called_once()
        call_kwargs = mock_log_timing.call_args[1]
        self.assertIsNotNone(call_kwargs.get("error_step"))

    @patch("app.log_timing_data")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_main_error(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_timing,
    ):
        """Test get_context_and_answer handles main pipeline errors"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.side_effect = Exception("Memory error")

        # Call function
        answer = get_context_and_answer(
            self.message,
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Should return error message
        self.assertIn("error", answer)

        # Verify error was logged
        mock_log_timing.assert_called_once()

    @patch("app.generate_xeno_response")
    @patch("app.process_context")
    @patch("app.generate_embeddings")
    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_with_chat_history(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
        mock_generate_embeddings,
        mock_process_context,
        mock_generate_response,
    ):
        """Test get_context_and_answer passes chat history to LLM"""
        # Setup mocks
        mock_session_config.return_value = {"session_id": self.session_id}
        chat_history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        mock_retrieve_memory.return_value = chat_history
        self.mock_intent_classifier.classify_intent.return_value = ("query", None)

        # Mock retriever
        mock_doc = Mock()
        mock_doc.page_content = "Test content"
        mock_doc.metadata = {"id": "KB001", "question": "Q", "content": "A"}
        self.mock_retriever.invoke.return_value = [mock_doc]

        # Mock embeddings
        mock_generate_embeddings.return_value = ([0.1, 0.2], [[0.9, 0.1]])

        # Mock context processing
        mock_process_context.return_value = ("Context", ["KB001"], [("Q", "A")])

        # Mock LLM response
        mock_generate_response.return_value = "Answer with context"

        # Call function
        answer = get_context_and_answer(
            self.message,
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Verify chat history was passed to LLM
        mock_generate_response.assert_called_once()
        call_args = mock_generate_response.call_args[0]
        self.assertEqual(call_args[2], chat_history)

    @patch("app.PipelineTimer")
    @patch("app.generate_xeno_response")
    @patch("app.process_context")
    @patch("app.generate_embeddings")
    @patch("app.log_timing_data")
    @patch("app.log_response")
    @patch("app.update_memory")
    @patch("app.retrieve_memory")
    @patch("app.create_session_config")
    def test_get_context_and_answer_timing(
        self,
        mock_session_config,
        mock_retrieve_memory,
        mock_update_memory,
        mock_log_response,
        mock_log_timing,
        mock_generate_embeddings,
        mock_process_context,
        mock_generate_response,
        mock_timer_class,
    ):
        """Test get_context_and_answer uses PipelineTimer correctly"""
        # Setup mocks
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        mock_timer.get_timing_summary.return_value = {"total": 1.5}
        mock_timer_class.return_value = mock_timer

        mock_session_config.return_value = {"session_id": self.session_id}
        mock_retrieve_memory.return_value = []
        self.mock_intent_classifier.classify_intent.return_value = ("query", None)

        # Mock retriever
        mock_doc = Mock()
        mock_doc.page_content = "Test"
        mock_doc.metadata = {"id": "KB001", "question": "Q", "content": "A"}
        self.mock_retriever.invoke.return_value = [mock_doc]

        # Mock embeddings
        mock_generate_embeddings.return_value = ([0.1], [[0.9]])
        mock_process_context.return_value = ("Context", ["KB001"], [("Q", "A")])
        mock_generate_response.return_value = "Answer"

        # Call function
        get_context_and_answer(
            self.message,
            self.history,
            self.session_id,
            self.mock_intent_classifier,
            self.mock_retriever,
        )

        # Verify timer was used
        mock_timer.reset.assert_called_once()
        mock_timer.get_timing_summary.assert_called()

        # Verify timing was logged
        mock_log_timing.assert_called_once()
        call_args = mock_log_timing.call_args[0]
        # Second positional argument is session_id, third is timing_summary
        self.assertIn("total", call_args[2])


if __name__ == "__main__":
    unittest.main()
