"""
Simple test file for knowledge_ingest.py - FIXED VERSION
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json
from src.graph.knowledge_ingest import KnowledgeGraphIngestor


class TestKnowledgeGraphIngestor:
    """Test suite for KnowledgeGraphIngestor class"""
    
    def test_init_creates_neo4j_client(self):
        """Test that KnowledgeGraphIngestor initializes with Neo4j client"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            mock_client.assert_called_once()
            assert ingestor.client == mock_client_instance

    def test_extract_id_with_id_field(self):
        """Test _extract_id with lowercase 'id' field"""
        ingestor = KnowledgeGraphIngestor()
        
        test_item = {"id": "KB001", "Question": "What is AI?"}
        result = ingestor._extract_id(test_item)
        
        assert result == "KB001"

    def test_extract_id_with_ID_field(self):
        """Test _extract_id with uppercase 'ID' field"""
        ingestor = KnowledgeGraphIngestor()
        
        test_item = {"ID": "KB002", "Question": "What is ML?"}
        result = ingestor._extract_id(test_item)
        
        assert result == "KB002"

    def test_extract_id_prefers_ID_over_id(self):
        """Test _extract_id prefers 'ID' over 'id' when both exist"""
        ingestor = KnowledgeGraphIngestor()
        
        test_item = {"ID": "PREFER_THIS", "id": "NOT_THIS", "Question": "Test"}
        result = ingestor._extract_id(test_item)
        
        assert result == "PREFER_THIS"

    def test_extract_id_returns_none_when_missing(self):
        """Test _extract_id returns None when no ID field exists"""
        ingestor = KnowledgeGraphIngestor()
        
        test_item = {"Question": "No ID here"}
        result = ingestor._extract_id(test_item)
        
        assert result is None

    def test_load_knowledge_base_success(self):
        """Test load_knowledge_base loads JSON data"""
        test_data = [
            {"id": "KB001", "Question": "Q1", "Content": "A1"},
            {"id": "KB002", "Question": "Q2", "Content": "A2"}
        ]
        
        with patch('src.graph.knowledge_ingest.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=json.dumps(test_data))), \
             patch('src.graph.knowledge_ingest.KNOWLEDGE_BASE_PATH', '/fake/path'):
            
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            ingestor = KnowledgeGraphIngestor()
            result = ingestor.load_knowledge_base()
            
            assert result == test_data
            mock_path_instance.exists.assert_called_once()

    def test_load_knowledge_base_file_not_found(self):
        """Test load_knowledge_base raises FileNotFoundError when file doesn't exist"""
        with patch('src.graph.knowledge_ingest.Path') as mock_path, \
             patch('src.graph.knowledge_ingest.KNOWLEDGE_BASE_PATH', '/fake/path'):
            
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            try:
                ingestor.load_knowledge_base()
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError as e:
                assert "Knowledge base not found" in str(e)

    def test_clear_existing_knowledge(self):
        """Test clear_existing_knowledge calls run_query with correct query"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            ingestor = KnowledgeGraphIngestor()
            ingestor.clear_existing_knowledge()
            
            # Check that run_query was called with the correct query
            mock_client_instance.run_query.assert_called_once()
            call_args = mock_client_instance.run_query.call_args
            
            # The query should be the first argument
            query = call_args[0][0]
            assert "MATCH (q:Question)" in query
            assert "DETACH DELETE q" in query

    def test_ingest_question_success(self):
        """Test ingest_question creates correct Neo4j query"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Track the parameters passed to run_query
            captured_params = {}
            
            def capture_run_query(query, params=None):
                captured_params['query'] = query
                captured_params['params'] = params
                return Mock()  # Return a mock result
            
            mock_client_instance.run_query.side_effect = capture_run_query
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "id": "KB001",
                "Question": "What is AI?",
                "Content": "AI is Artificial Intelligence",
                "Source": "Test Source",
                "Section": "Technology",
                "Tag": "AI"
            }
            
            ingestor.ingest_question(test_item)
            
            # Check that run_query was called
            mock_client_instance.run_query.assert_called_once()
            
            # Check the captured parameters
            assert captured_params['params'] is not None
            params = captured_params['params']
            
            assert params["id"] == "KB001"
            assert params["question"] == "What is AI?"
            assert params["content"] == "AI is Artificial Intelligence"
            assert params["source"] == "Test Source"
            assert params["section"] == "Technology"
            assert params["tag"] == "AI"

    def test_ingest_question_with_ID_field(self):
        """Test ingest_question works with uppercase ID field"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Track the parameters passed to run_query
            captured_params = {}
            
            def capture_run_query(query, params=None):
                captured_params['query'] = query
                captured_params['params'] = params
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_run_query
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "ID": "KB002",  # Uppercase ID
                "Question": "What is ML?",
                "Content": "ML is Machine Learning"
                # Missing optional fields should use defaults
            }
            
            ingestor.ingest_question(test_item)
            
            # Check the captured parameters
            assert captured_params['params'] is not None
            params = captured_params['params']
            
            assert params["id"] == "KB002"
            assert params["question"] == "What is ML?"
            assert params["content"] == "ML is Machine Learning"
            assert params["source"] == "KnowledgeBase"  # Default
            assert params["section"] == "General"  # Default
            assert params["tag"] == "General"  # Default

    def test_ingest_question_missing_id_raises_error(self):
        """Test ingest_question raises ValueError when ID is missing"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "Question": "No ID here",
                "Content": "Some content"
            }
            
            try:
                ingestor.ingest_question(test_item)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "Missing ID / id field" in str(e)
            
            mock_client_instance.run_query.assert_not_called()

    def test_ingest_question_strips_whitespace(self):
        """Test ingest_question strips whitespace from text fields"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Track the parameters passed to run_query
            captured_params = {}
            
            def capture_run_query(query, params=None):
                captured_params['query'] = query
                captured_params['params'] = params
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_run_query
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "id": "KB003",
                "Question": "  What is Python?  ",
                "Content": "  Python is a language  ",
                "Source": "  Test Source  ",
                "Section": "  Programming  ",
                "Tag": "  Python  "
            }
            
            ingestor.ingest_question(test_item)
            
            # Check the captured parameters
            assert captured_params['params'] is not None
            params = captured_params['params']
            
            # question and content should be stripped (have .strip() in code)
            assert params["question"] == "What is Python?"  # Stripped
            assert params["content"] == "Python is a language"  # Stripped
            
            # source, section, tag should NOT be stripped (no .strip() in code)
            assert params["source"] == "  Test Source  "  # NOT stripped
            assert params["section"] == "  Programming  "  # NOT stripped
            assert params["tag"] == "  Python  "  # NOT stripped

    def test_ingest_all_success(self):
        """Test ingest_all processes all items"""
        test_data = [
            {"id": "KB001", "Question": "Q1", "Content": "A1"},
            {"id": "KB002", "Question": "Q2", "Content": "A2"},
            {"id": "KB003", "Question": "Q3", "Content": "A3"}
        ]
        
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client, \
             patch('src.graph.knowledge_ingest.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=json.dumps(test_data))), \
             patch('src.graph.knowledge_ingest.KNOWLEDGE_BASE_PATH', '/fake/path'):
            
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            # Mock the clear_existing_knowledge method
            with patch.object(ingestor, 'clear_existing_knowledge') as mock_clear:
                ingestor.ingest_all(wipe_existing=True)
                
                mock_clear.assert_called_once()
                
                # Should have called run_query 3 times (once per item)
                assert mock_client_instance.run_query.call_count == 3

    def test_ingest_all_without_wipe_existing(self):
        """Test ingest_all doesn't clear existing when wipe_existing=False"""
        test_data = [{"id": "KB001", "Question": "Q1", "Content": "A1"}]
        
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client, \
             patch('src.graph.knowledge_ingest.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=json.dumps(test_data))), \
             patch('src.graph.knowledge_ingest.KNOWLEDGE_BASE_PATH', '/fake/path'):
            
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            # Mock the clear_existing_knowledge method
            with patch.object(ingestor, 'clear_existing_knowledge') as mock_clear:
                ingestor.ingest_all(wipe_existing=False)
                
                # Should NOT have called clear_existing_knowledge
                mock_clear.assert_not_called()
                
                # Should have called run_query for the item
                mock_client_instance.run_query.assert_called_once()

    def test_ingest_all_with_failures(self):
        """Test ingest_all handles failed items gracefully"""
        test_data = [
            {"id": "KB001", "Question": "Q1", "Content": "A1"},  # Should succeed
            {"Question": "Q2", "Content": "A2"},  # Missing ID - should fail
            {"id": "KB003", "Question": "Q3", "Content": "A3"}   # Should succeed
        ]
        
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client, \
             patch('src.graph.knowledge_ingest.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=json.dumps(test_data))), \
             patch('src.graph.knowledge_ingest.KNOWLEDGE_BASE_PATH', '/fake/path'):
            
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            # Count successful calls
            successful_calls = 0
            
            def count_run_query(query, params=None):
                nonlocal successful_calls
                if params and 'id' in params:
                    successful_calls += 1
                return Mock()
            
            mock_client_instance.run_query.side_effect = count_run_query
            
            with patch.object(ingestor, 'clear_existing_knowledge'):
                ingestor.ingest_all(wipe_existing=True)
                
                # Should have had 2 successful calls
                assert successful_calls == 2

    def test_ingest_all_empty_data(self):
        """Test ingest_all handles empty knowledge base"""
        test_data = []
        
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client, \
             patch('src.graph.knowledge_ingest.Path') as mock_path, \
             patch('builtins.open', mock_open(read_data=json.dumps(test_data))), \
             patch('src.graph.knowledge_ingest.KNOWLEDGE_BASE_PATH', '/fake/path'):
            
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            
            ingestor = KnowledgeGraphIngestor()
            
            with patch.object(ingestor, 'clear_existing_knowledge'):
                ingestor.ingest_all(wipe_existing=True)
                
                # Should not call run_query for any items
                mock_client_instance.run_query.assert_not_called()

    def test_close_method(self):
        """Test close method closes the client"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            ingestor = KnowledgeGraphIngestor()
            ingestor.close()
            
            mock_client_instance.close.assert_called_once()

    def test_ingest_question_with_minimal_data(self):
        """Test ingest_question works with minimal data (only required fields)"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Track the parameters passed to run_query
            captured_params = {}
            
            def capture_run_query(query, params=None):
                captured_params['query'] = query
                captured_params['params'] = params
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_run_query
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "id": "MINIMAL001",
                "Question": "Minimal Q",
                "Content": "Minimal A"
                # No Source, Section, Tag
            }
            
            ingestor.ingest_question(test_item)
            
            # Check the captured parameters
            assert captured_params['params'] is not None
            params = captured_params['params']
            
            assert params["id"] == "MINIMAL001"
            assert params["question"] == "Minimal Q"
            assert params["content"] == "Minimal A"
            assert params["source"] == "KnowledgeBase"  # Default
            assert params["section"] == "General"  # Default
            assert params["tag"] == "General"  # Default

    def test_ingest_question_with_empty_strings(self):
        """Test ingest_question handles empty string fields"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Track the parameters passed to run_query
            captured_params = {}
            
            def capture_run_query(query, params=None):
                captured_params['query'] = query
                captured_params['params'] = params
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_run_query
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "id": "EMPTY001",
                "Question": "",
                "Content": "",
                "Source": "",
                "Section": "",
                "Tag": ""
            }
            
            ingestor.ingest_question(test_item)
            
            # Check the captured parameters
            assert captured_params['params'] is not None
            params = captured_params['params']
            
            assert params["id"] == "EMPTY001"
            # question and content: empty string -> .strip() -> empty string
            assert params["question"] == ""  # Empty string after .strip()
            assert params["content"] == ""  # Empty string after .strip()
            
            # IMPORTANT: When key exists with empty string value, 
            # item.get("Source", "KnowledgeBase") returns "" (empty string)
            # NOT "KnowledgeBase" (default is only used when key doesn't exist)
            assert params["source"] == ""  # Empty string (key exists)
            assert params["section"] == ""  # Empty string (key exists)
            assert params["tag"] == ""  # Empty string (key exists)

    def test_ingest_question_with_special_characters(self):
        """Test ingest_question handles special characters in fields"""
        with patch('src.graph.knowledge_ingest.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Track the parameters passed to run_query
            captured_params = {}
            
            def capture_run_query(query, params=None):
                captured_params['query'] = query
                captured_params['params'] = params
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_run_query
            
            ingestor = KnowledgeGraphIngestor()
            
            test_item = {
                "id": "SPECIAL001",
                "Question": "What's the difference between AI & ML?",
                "Content": "AI (Artificial Intelligence) ≠ ML (Machine Learning)",
                "Source": "Source: https://example.com",
                "Section": "Tech/AI",
                "Tag": "AI/ML"
            }
            
            ingestor.ingest_question(test_item)
            
            # Check the captured parameters
            assert captured_params['params'] is not None
            params = captured_params['params']
            
            assert params["id"] == "SPECIAL001"
            assert "What's" in params["question"]
            assert "≠" in params["content"]
            assert "https://" in params["source"]
            assert "Tech/AI" == params["section"]
            assert "AI/ML" == params["tag"]


if __name__ == "__main__":
    # Simple test runner
    print("Running Knowledge Graph Ingestor Tests...")
    print("=" * 60)
    
    test_instance = TestKnowledgeGraphIngestor()
    
    # Get all test methods
    test_methods = []
    for method_name in dir(test_instance):
        if method_name.startswith('test_'):
            test_methods.append(method_name)
    
    test_methods.sort()  # Run in alphabetical order
    
    passed = 0
    failed = 0
    
    for method_name in test_methods:
        print(f"\nRunning: {method_name}")
        print("-" * 40)
        try:
            getattr(test_instance, method_name)()
            print(f"✓ PASSED: {method_name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {method_name}")
            print(f"  Assertion Error: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"✗ FAILED: {method_name}")
            print(f"  Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")
    
    # Exit with appropriate code
    import sys
    sys.exit(1 if failed > 0 else 0)