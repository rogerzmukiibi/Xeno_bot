"""
Unit tests for memory module
Tests LangGraph memory operations
"""
import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Add the parent directory to sys.path to find src module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the config module before importing memory
config_mock = Mock()
config_mock.SQLITE_DB_PATH = ':memory:'  # Use in-memory database for tests

sys.modules['config'] = config_mock

# Now import the memory module
from src.memory import (
    update_memory,
    retrieve_memory,
    create_session_config
)


class TestMemory(unittest.TestCase):
    """Test cases for memory module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_config = {
            "configurable": {
                "thread_id": "test_session_123",
                "checkpoint_ns": ""
            }
        }
        self.user_message = "How do I create an account?"
        self.assistant_message = "You can create an account by visiting our website."
    
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
    
    @patch('src.memory.memory')
    def test_update_memory_with_existing_checkpoint(self, mock_memory):
        """Test updating memory with existing checkpoint"""
        # Mock memory.get to return existing checkpoint
        mock_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Previous question"},
                    {"role": "assistant", "content": "Previous answer"}
                ]
            }
        }
        mock_memory.get.return_value = mock_checkpoint
        
        update_memory(self.test_config, self.user_message, self.assistant_message)
        
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
        self.assertEqual(messages[-2]["content"], self.user_message)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(messages[-1]["content"], self.assistant_message)
    
    @patch('src.memory.memory')
    def test_update_memory_empty_checkpoint(self, mock_memory):
        """Test updating memory with empty checkpoint"""
        # Mock memory.get to return None
        mock_memory.get.return_value = None
        
        update_memory(self.test_config, self.user_message, self.assistant_message)
        
        # Verify memory.put was called
        mock_memory.put.assert_called_once()
        
        # Check the checkpoint
        call_args = mock_memory.put.call_args
        saved_checkpoint = call_args[0][1]
        messages = saved_checkpoint["channel_values"]["messages"]
        
        # Should have 2 messages
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], self.user_message)
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], self.assistant_message)
    
    @patch('src.memory.memory')
    def test_update_memory_with_timer(self, mock_memory):
        """Test update_memory with timer"""
        mock_memory.get.return_value = {}
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        update_memory(self.test_config, self.user_message, self.assistant_message, timer=mock_timer)
        
        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("memory_update")
        mock_memory.put.assert_called_once()
    
    @patch('src.memory.memory')
    def test_update_memory_without_timer(self, mock_memory):
        """Test update_memory without timer"""
        mock_memory.get.return_value = {}
        
        update_memory(self.test_config, self.user_message, self.assistant_message)
        
        # Should still work without timer
        mock_memory.put.assert_called_once()
    
    @patch('src.memory.memory')
    def test_retrieve_memory(self, mock_memory):
        """Test memory retrieval"""
        # Mock memory.get to return checkpoint with messages
        mock_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"}
                ]
            }
        }
        mock_memory.get.return_value = mock_checkpoint
        
        messages = retrieve_memory(self.test_config)
        
        # Verify memory.get was called
        mock_memory.get.assert_called_once_with(self.test_config)
        
        # Verify messages were retrieved
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["content"], "Question 1")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["content"], "Answer 1")
        self.assertEqual(messages[1]["role"], "assistant")
    
    @patch('src.memory.memory')
    def test_retrieve_memory_empty(self, mock_memory):
        """Test retrieving memory when empty"""
        # Mock memory.get to return None
        mock_memory.get.return_value = None
        
        messages = retrieve_memory(self.test_config)
        
        # Should return empty list
        self.assertEqual(messages, [])
    
    @patch('src.memory.memory')
    def test_retrieve_memory_with_timer(self, mock_memory):
        """Test retrieve_memory with timer"""
        mock_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"}
                ]
            }
        }
        mock_memory.get.return_value = mock_checkpoint
        
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        messages = retrieve_memory(self.test_config, timer=mock_timer)
        
        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("memory_retrieval")
        
        # Verify messages were retrieved
        self.assertEqual(len(messages), 2)
    
    @patch('src.memory.memory')
    def test_retrieve_memory_without_timer(self, mock_memory):
        """Test retrieve_memory without timer"""
        mock_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Test question"},
                    {"role": "assistant", "content": "Test answer"}
                ]
            }
        }
        mock_memory.get.return_value = mock_checkpoint
        
        messages = retrieve_memory(self.test_config)
        
        # Should still work without timer
        self.assertEqual(len(messages), 2)
    
    @patch('src.memory.memory')
    def test_checkpoint_structure(self, mock_memory):
        """Test that checkpoint has correct structure"""
        mock_memory.get.return_value = None
        
        update_memory(self.test_config, self.user_message, self.assistant_message)
        
        call_args = mock_memory.put.call_args
        checkpoint = call_args[0][1]
        
        # Verify checkpoint structure
        self.assertIn("v", checkpoint)
        self.assertIn("id", checkpoint)
        self.assertIn("ts", checkpoint)
        self.assertIn("channel_values", checkpoint)
        self.assertIn("channel_versions", checkpoint)
        self.assertIn("versions_seen", checkpoint)
        
        # Verify specific values
        self.assertEqual(checkpoint["v"], 1)
        self.assertIsInstance(checkpoint["id"], str)  # Should be a UUID string
        self.assertIsInstance(checkpoint["ts"], str)  # Should be an ISO timestamp
        
        # Verify messages are in channel_values
        self.assertIn("messages", checkpoint["channel_values"])
        messages = checkpoint["channel_values"]["messages"]
        self.assertEqual(len(messages), 2)
    
    @patch('src.memory.memory')
    def test_full_conversation_flow(self, mock_memory):
        """Test full conversation flow with multiple updates and retrievals"""
        # Start with empty memory
        mock_memory.get.return_value = None
        
        # First interaction
        update_memory(self.test_config, "Question 1", "Answer 1")
        
        # Mock get to return first checkpoint
        first_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"}
                ]
            }
        }
        mock_memory.get.return_value = first_checkpoint
        
        # Retrieve first messages
        messages1 = retrieve_memory(self.test_config)
        self.assertEqual(len(messages1), 2)
        
        # Second interaction
        update_memory(self.test_config, "Question 2", "Answer 2")
        
        # Mock get to return updated checkpoint
        second_checkpoint = {
            "channel_values": {
                "messages": [
                    {"role": "user", "content": "Question 1"},
                    {"role": "assistant", "content": "Answer 1"},
                    {"role": "user", "content": "Question 2"},
                    {"role": "assistant", "content": "Answer 2"}
                ]
            }
        }
        mock_memory.get.return_value = second_checkpoint
        
        # Retrieve all messages
        messages2 = retrieve_memory(self.test_config)
        self.assertEqual(len(messages2), 4)
        self.assertEqual(messages2[-1]["content"], "Answer 2")


if __name__ == '__main__':
    unittest.main()