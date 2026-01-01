"""
Simple test file for cognitive_memory.py that doesn't use conftest.py fixtures
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch, MagicMock
from src.cognitive.cognitive_memory import CognitiveMemory


class TestCognitiveMemorySimple:
    """Test suite for CognitiveMemory class without global fixtures"""
    
    def test_init_creates_driver_with_config(self):
        """Test that CognitiveMemory initializes driver with correct config"""
        with patch('src.cognitive.cognitive_memory.GraphDatabase') as mock_db:
            mock_driver = Mock()
            mock_db.driver.return_value = mock_driver
            
            with patch('src.cognitive.cognitive_memory.NEO4J_URI', 'test_uri'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_USER', 'test_user'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_PASSWORD', 'test_pass'):
                
                cm = CognitiveMemory()
                
                mock_db.driver.assert_called_once_with(
                    'test_uri',
                    auth=('test_user', 'test_pass')
                )
                assert cm.driver == mock_driver

    def test_close_closes_driver(self):
        """Test close method closes the driver"""
        with patch('src.cognitive.cognitive_memory.GraphDatabase') as mock_db:
            mock_driver = Mock()
            mock_db.driver.return_value = mock_driver
            
            with patch('src.cognitive.cognitive_memory.NEO4J_URI', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_USER', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_PASSWORD', 'test'):
                
                cm = CognitiveMemory()
                cm.close()
                mock_driver.close.assert_called_once()

    def test_upsert_user_creates_new_user(self):
        """Test upsert_user creates new user when not exists"""
        session_id = "test_session_123"
        
        with patch('src.cognitive.cognitive_memory.GraphDatabase') as mock_db:
            mock_driver = Mock()
            
            # Create a MagicMock which handles __enter__ and __exit__ better
            mock_session = MagicMock()
            mock_result = Mock()
            
            # Create a context manager mock for session()
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = Mock(return_value=mock_session)
            mock_context_manager.__exit__ = Mock(return_value=None)
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value = mock_context_manager
            mock_session.run.return_value = mock_result
            
            with patch('src.cognitive.cognitive_memory.NEO4J_URI', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_USER', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_PASSWORD', 'test'):
                
                cm = CognitiveMemory()
                cm.upsert_user(session_id)
                
                mock_session.run.assert_called_once()
                call_args = mock_session.run.call_args
                assert call_args[1]['session_id'] == session_id

    def test_create_memory_creates_node_and_links_to_user(self):
        """Test create_memory creates memory node and links to user"""
        session_id = "test_session_123"
        user_question = "What is AI?"
        bot_answer = "AI is artificial intelligence."
        source_ids = ["src1", "src2"]
        intent = "question"
        confidence = 0.95
        
        with patch('src.cognitive.cognitive_memory.GraphDatabase') as mock_db:
            mock_driver = Mock()
            
            # Create a MagicMock which handles __enter__ and __exit__ better
            mock_session = MagicMock()
            mock_result = Mock()
            
            # Create a context manager mock for session()
            mock_context_manager = MagicMock()
            mock_context_manager.__enter__ = Mock(return_value=mock_session)
            mock_context_manager.__exit__ = Mock(return_value=None)
            
            mock_db.driver.return_value = mock_driver
            mock_driver.session.return_value = mock_context_manager
            mock_session.run.return_value = mock_result
            
            with patch('src.cognitive.cognitive_memory.NEO4J_URI', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_USER', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_PASSWORD', 'test'):
                
                cm = CognitiveMemory()
                cm.create_memory(
                    session_id=session_id,
                    user_question=user_question,
                    bot_answer=bot_answer,
                    source_ids=source_ids,
                    intent=intent,
                    confidence=confidence
                )
                
                mock_session.run.assert_called_once()
                params = mock_session.run.call_args[1]
                assert params['session_id'] == session_id
                assert params['question'] == user_question
                assert params['answer'] == bot_answer
                assert params['source_ids'] == source_ids
                assert params['intent'] == intent
                assert params['confidence'] == confidence

    def test_write_memory_calls_all_methods(self):
        """Test write_memory calls all three methods in sequence"""
        session_id = "test_session_123"
        question = "What is ML?"
        answer = "ML is machine learning."
        source_ids = ["src1"]
        intent = "question"
        confidence = 0.9
        
        with patch('src.cognitive.cognitive_memory.GraphDatabase') as mock_db:
            mock_driver = Mock()
            mock_db.driver.return_value = mock_driver
            
            with patch('src.cognitive.cognitive_memory.NEO4J_URI', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_USER', 'test'), \
                 patch('src.cognitive.cognitive_memory.NEO4J_PASSWORD', 'test'):
                
                cm = CognitiveMemory()
                
                # Mock the three methods
                with patch.object(cm, 'upsert_user') as mock_upsert, \
                     patch.object(cm, 'create_memory') as mock_create, \
                     patch.object(cm, 'link_memory_to_knowledge') as mock_link:
                    
                    cm.write_memory(
                        session_id=session_id,
                        question=question,
                        answer=answer,
                        source_ids=source_ids,
                        intent=intent,
                        confidence=confidence
                    )
                    
                    mock_upsert.assert_called_once_with(session_id)
                    mock_create.assert_called_once_with(
                        session_id=session_id,
                        user_question=question,
                        bot_answer=answer,
                        source_ids=source_ids,
                        intent=intent,
                        confidence=confidence
                    )
                    mock_link.assert_called_once_with(session_id, source_ids)


if __name__ == "__main__":
    # Simple test runner
    print("Running Cognitive Memory Tests (direct execution)...")
    print("=" * 60)
    
    test_instance = TestCognitiveMemorySimple()
    test_methods = [
        'test_init_creates_driver_with_config',
        'test_close_closes_driver', 
        'test_upsert_user_creates_new_user',
        'test_create_memory_creates_node_and_links_to_user',
        'test_write_memory_calls_all_methods'
    ]
    
    for method_name in test_methods:
        print(f"\nRunning: {method_name}")
        print("-" * 40)
        try:
            test_method = getattr(test_instance, method_name)
            test_method()
            print(f"✓ PASSED: {method_name}")
        except Exception as e:
            print(f"✗ FAILED: {method_name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test run completed!")