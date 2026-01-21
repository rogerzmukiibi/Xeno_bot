"""
Unit tests for memory module
Tests LangGraph memory operations
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from src.memory import (_retrieve_memory_impl, _update_memory_impl,
                        create_session_config, retrieve_memory, update_memory)


class TestMemory(unittest.TestCase):
    """Test cases for memory module"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "configurable": {"thread_id": "test_session_123", "checkpoint_ns": ""}
        }

    def test_create_session_config(self):
        """Test creating session config"""
        session_id = "test_session_456"
        config = create_session_config(session_id)

        # Check structure
        self.assertIn("configurable", config)
        self.assertEqual(config["configurable"]["thread_id"], session_id)
        self.assertEqual(config["configurable"]["checkpoint_ns"], "")

    def test_create_session_config_default(self):
        """Test creating session config with default ID"""
        config = create_session_config()

        # Check structure
        self.assertIn("configurable", config)
        self.assertEqual(config["configurable"]["thread_id"], "default")

    @patch("src.memory.memory")
    def test_update_memory_impl(self, mock_memory):
        """Test internal memory update implementation"""
        # Mock memory.get to return existing checkpoint
        mock_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"},
                ]
            }
        }
        mock_memory.get.return_value = mock_checkpoint

        user_message = "New question"
        assistant_message = "New answer"

        _update_memory_impl(self.test_config, user_message, assistant_message)

        # Verify memory.get was called
        mock_memory.get.assert_called_once_with(self.test_config)

        # Verify memory.put was called
        mock_memory.put.assert_called_once()

        # Check the checkpoint that was saved
        call_args = mock_memory.put.call_args
        saved_checkpoint = call_args[0][1]

        # Verify messages were appended
        messages = saved_checkpoint["channel_values"]["messages"]
        self.assertEqual(len(messages), 4)  # 2 existing + 2 new
        self.assertEqual(messages[-2]["role"], "user")
        self.assertEqual(messages[-2]["content"], user_message)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], assistant_message)

    @patch("src.memory.memory")
    def test_update_memory_empty_checkpoint(self, mock_memory):
        """Test updating memory with empty checkpoint"""
        # Mock memory.get to return None
        mock_memory.get.return_value = None

        user_message = "First question"
        assistant_message = "First answer"

        _update_memory_impl(self.test_config, user_message, assistant_message)

        # Verify memory.put was called
        mock_memory.put.assert_called_once()

        # Check the checkpoint
        call_args = mock_memory.put.call_args
        saved_checkpoint = call_args[0][1]
        messages = saved_checkpoint["channel_values"]["messages"]

        # Should have 2 messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")

    @patch("src.memory.memory")
    def test_update_memory_with_timer(self, mock_memory):
        """Test update_memory with timer"""
        mock_memory.get.return_value = {}
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()

        update_memory(self.test_config, "Test", "Answer", timer=mock_timer)

        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("memory_update")

    @patch("src.memory.memory")
    def test_retrieve_memory_impl(self, mock_memory):
        """Test internal memory retrieval implementation"""
        # Mock memory.get to return checkpoint with messages
        mock_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"},
                ]
            }
        }
        mock_memory.get.return_value = mock_checkpoint

        messages = _retrieve_memory_impl(self.test_config)

        # Verify memory.get was called
        mock_memory.get.assert_called_once_with(self.test_config)

        # Verify messages were retrieved
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["content"], "Question 1")

    @patch("src.memory.memory")
    def test_retrieve_memory_empty(self, mock_memory):
        """Test retrieving memory when empty"""
        # Mock memory.get to return None
        mock_memory.get.return_value = None

        messages = _retrieve_memory_impl(self.test_config)

        # Should return empty list
        self.assertEqual(messages, [])

    @patch("src.memory.memory")
    def test_retrieve_memory_with_timer(self, mock_memory):
        """Test retrieve_memory with timer"""
        mock_memory.get.return_value = {}
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()

        retrieve_memory(self.test_config, timer=mock_timer)

        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("memory_retrieval")

    @patch("src.memory.memory")
    def test_checkpoint_structure(self, mock_memory):
        """Test that checkpoint has correct structure"""
        mock_memory.get.return_value = None

        _update_memory_impl(self.test_config, "Test", "Answer")

        call_args = mock_memory.put.call_args
        checkpoint = call_args[0][1]

        # Verify checkpoint structure
        self.assertIn("v", checkpoint)
        self.assertIn("id", checkpoint)
        self.assertIn("ts", checkpoint)
        self.assertIn("channel_values", checkpoint)
        self.assertIn("channel_versions", checkpoint)
        self.assertIn("versions_seen", checkpoint)
        self.assertEqual(checkpoint["v"], 1)


if __name__ == "__main__":
    unittest.main()
