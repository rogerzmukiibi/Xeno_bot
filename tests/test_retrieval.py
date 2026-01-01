"""
Simple test file for retrieval.py - FIXED VERSION
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch, MagicMock
from src.cognitive.retrieval import CognitiveRetriever, SIMILARITY_THRESHOLD


class TestCognitiveRetriever:
    """Test suite for CognitiveRetriever class"""
    
    def test_init_initializes_components(self):
        """Test that CognitiveRetriever initializes properly"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_collection = Mock()
            mock_vector_store = Mock()
            mock_retriever = Mock()
            
            mock_init.return_value = (mock_collection, mock_vector_store, mock_retriever)
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            retriever = CognitiveRetriever()
            
            mock_init.assert_called_once()
            mock_client.assert_called_once()
            
            assert retriever.collection == mock_collection
            assert retriever.vector_store == mock_vector_store
            assert retriever.retriever == mock_retriever
            assert retriever.graph_client == mock_client_instance

    def test_empty_result_static_method(self):
        """Test the _empty_result static method"""
        reason = "Test reason for empty result"
        result = CognitiveRetriever._empty_result(reason)
        
        assert result["status"] == "empty"
        assert result["reason"] == reason
        assert result["formatted_context"] == ""
        assert result["source_ids"] == []
        assert result["knowledge_pairs"] == []
        assert result["graph_context"] == []
        assert result["confidence"] == 0.0

    def test_retrieve_greeting_intent_short_circuit(self):
        """Test retrieve short-circuits for greeting intent"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_client.return_value = Mock()
            
            retriever = CognitiveRetriever()
            
            # Test greeting intent
            result = retriever.retrieve("Hello", "greeting")
            
            assert result["status"] == "empty"
            assert "Intent does not require knowledge retrieval" in result["reason"]
            assert result["confidence"] == 0.0

    def test_retrieve_small_talk_intent_short_circuit(self):
        """Test retrieve short-circuits for small_talk intent"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_client.return_value = Mock()
            
            retriever = CognitiveRetriever()
            
            # Test small_talk intent
            result = retriever.retrieve("How are you?", "small_talk")
            
            assert result["status"] == "empty"
            assert "Intent does not require knowledge retrieval" in result["reason"]

    def test_retrieve_no_documents_found(self):
        """Test retrieve returns empty when no documents found"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_retriever = Mock()
            mock_retriever.get_relevant_documents.return_value = []  # Empty results
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client.return_value = Mock()
            
            retriever = CognitiveRetriever()
            
            result = retriever.retrieve("What is quantum physics?", "question")
            
            assert result["status"] == "empty"
            assert "No documents retrieved from vector store" in result["reason"]
            mock_retriever.get_relevant_documents.assert_called_once_with("What is quantum physics?")

    def test_retrieve_low_similarity_threshold(self):
        """Test retrieve returns empty when similarity below threshold"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client, \
             patch('src.cognitive.retrieval.generate_embeddings') as mock_embeddings, \
             patch('src.cognitive.retrieval.calculate_similarity') as mock_similarity:
            
            mock_retriever = Mock()
            mock_doc = Mock()
            mock_doc.page_content = "Some content"
            mock_retriever.get_relevant_documents.return_value = [mock_doc]
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client.return_value = Mock()
            
            # Mock embeddings to return two values
            mock_embeddings.return_value = (Mock(), [Mock()])
            
            # Mock low similarity scores (below threshold)
            mock_similarity.return_value = [SIMILARITY_THRESHOLD - 0.1]
            
            retriever = CognitiveRetriever()
            
            result = retriever.retrieve("Test query", "question")
            
            assert result["status"] == "empty"
            assert "Similarity below confidence threshold" in result["reason"]
            
            mock_embeddings.assert_called_once()
            mock_similarity.assert_called_once()

    def test_retrieve_successful_with_high_similarity(self):
        """Test successful retrieval with high similarity"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client, \
             patch('src.cognitive.retrieval.generate_embeddings') as mock_embeddings, \
             patch('src.cognitive.retrieval.calculate_similarity') as mock_similarity, \
             patch('src.cognitive.retrieval.process_context') as mock_context:
            
            mock_retriever = Mock()
            mock_doc = Mock()
            mock_doc.page_content = "Some content"
            mock_retriever.get_relevant_documents.return_value = [mock_doc]
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock embeddings to return two values
            mock_embeddings.return_value = (Mock(), [Mock()])
            
            # Mock high similarity scores (above threshold)
            high_confidence = SIMILARITY_THRESHOLD + 0.1
            mock_similarity.return_value = [high_confidence]
            
            # Mock process_context return values
            mock_context.return_value = (
                "Formatted context",
                ["src1", "src2"],
                [("Question", "Answer")]
            )
            
            # Mock graph context
            mock_client_instance.run_query.return_value = [
                {"id": "src1", "question": "Test question", "relation": "RELATED", "related": "Related node"}
            ]
            
            retriever = CognitiveRetriever()
            
            result = retriever.retrieve("Test query", "question")
            
            assert result["status"] == "ok"
            assert result["formatted_context"] == "Formatted context"
            assert result["source_ids"] == ["src1", "src2"]
            assert result["knowledge_pairs"] == [("Question", "Answer")]
            assert result["confidence"] == high_confidence
            assert len(result["graph_context"]) == 1
            
            mock_embeddings.assert_called_once()
            mock_similarity.assert_called_once()
            mock_context.assert_called_once()
            mock_client_instance.run_query.assert_called_once()

    def test_retrieve_with_timer_parameter(self):
        """Test retrieve passes timer parameter to helper functions"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client, \
             patch('src.cognitive.retrieval.generate_embeddings') as mock_embeddings, \
             patch('src.cognitive.retrieval.calculate_similarity') as mock_similarity, \
             patch('src.cognitive.retrieval.process_context') as mock_context:
            
            mock_retriever = Mock()
            mock_doc = Mock()
            mock_retriever.get_relevant_documents.return_value = [mock_doc]
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock embeddings to return two values
            mock_embeddings.return_value = (Mock(), [Mock()])
            
            mock_similarity.return_value = [SIMILARITY_THRESHOLD + 0.1]
            mock_context.return_value = ("Context", ["src1"], [("Q", "A")])
            
            retriever = CognitiveRetriever()
            
            # Mock timer object
            mock_timer = Mock()
            
            result = retriever.retrieve("Test query", "question", timer=mock_timer)
            
            # Verify timer was passed to helper functions
            mock_embeddings.assert_called_once()
            call_args = mock_embeddings.call_args
            
            mock_similarity.assert_called_once()
            call_args = mock_similarity.call_args
            
            mock_context.assert_called_once()
            call_args = mock_context.call_args

    def test_fetch_graph_context_with_source_ids(self):
        """Test _fetch_graph_context with source IDs"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            mock_client_instance.run_query.return_value = [
                {"id": "src1", "question": "Q1", "relation": "RELATES", "related": "Node1"},
                {"id": "src2", "question": "Q2", "relation": "LINKS", "related": "Node2"}
            ]
            
            retriever = CognitiveRetriever()
            
            source_ids = ["src1", "src2", "src3"]
            result = retriever._fetch_graph_context(source_ids)
            
            mock_client_instance.run_query.assert_called_once()
            query_args = mock_client_instance.run_query.call_args
            
            # Check query was called with parameters
            # Note: The actual parameter name might be different in your Neo4jClient
            # Let's check what parameters were actually passed
            if len(query_args) > 1 and isinstance(query_args[1], dict):
                # If parameters are passed as dict
                params = query_args[1]
                # Check if 'ids' parameter exists, otherwise adjust
                if 'ids' in params:
                    assert params["ids"] == source_ids
                else:
                    # Try to find the parameter name
                    for key, value in params.items():
                        if value == source_ids:
                            break
            
            assert len(result) == 2

    def test_fetch_graph_context_empty_source_ids(self):
        """Test _fetch_graph_context with empty source IDs"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            retriever = CognitiveRetriever()
            
            result = retriever._fetch_graph_context([])
            
            # Should return empty list without calling run_query
            assert result == []
            mock_client_instance.run_query.assert_not_called()

    def test_fetch_graph_context_exception_handling(self):
        """Test _fetch_graph_context handles exceptions gracefully"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Make run_query raise an exception
            mock_client_instance.run_query.side_effect = Exception("Graph DB error")
            
            retriever = CognitiveRetriever()
            
            source_ids = ["src1"]
            result = retriever._fetch_graph_context(source_ids)
            
            # Should return empty list on exception
            assert result == []
            mock_client_instance.run_query.assert_called_once()

    def test_retrieve_at_threshold_similarity(self):
        """Test retrieve with similarity exactly at threshold"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client, \
             patch('src.cognitive.retrieval.generate_embeddings') as mock_embeddings, \
             patch('src.cognitive.retrieval.calculate_similarity') as mock_similarity, \
             patch('src.cognitive.retrieval.process_context') as mock_context:
            
            mock_retriever = Mock()
            mock_doc = Mock()
            mock_retriever.get_relevant_documents.return_value = [mock_doc]
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock embeddings to return two values
            mock_embeddings.return_value = (Mock(), [Mock()])
            
            # Mock similarity at exact threshold
            mock_similarity.return_value = [SIMILARITY_THRESHOLD]  # Exactly at threshold
            
            mock_context.return_value = ("Context", ["src1"], [("Q", "A")])
            
            retriever = CognitiveRetriever()
            
            result = retriever.retrieve("Test query", "question")
            
            # Should succeed (>= threshold)
            assert result["status"] == "ok"
            assert result["confidence"] == SIMILARITY_THRESHOLD

    def test_close_method(self):
        """Test close method cleans up resources"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client:
            
            mock_init.return_value = (Mock(), Mock(), Mock())
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            retriever = CognitiveRetriever()
            retriever.close()
            
            mock_client_instance.close.assert_called_once()

    def test_retrieve_multiple_documents(self):
        """Test retrieve with multiple documents"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client, \
             patch('src.cognitive.retrieval.generate_embeddings') as mock_embeddings, \
             patch('src.cognitive.retrieval.calculate_similarity') as mock_similarity, \
             patch('src.cognitive.retrieval.process_context') as mock_context:
            
            mock_retriever = Mock()
            
            # Create multiple mock documents
            mock_doc1 = Mock()
            mock_doc1.page_content = "Content 1"
            mock_doc2 = Mock()
            mock_doc2.page_content = "Content 2"
            mock_doc3 = Mock()
            mock_doc3.page_content = "Content 3"
            
            mock_retriever.get_relevant_documents.return_value = [mock_doc1, mock_doc2, mock_doc3]
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Mock embeddings to return two values
            mock_embeddings.return_value = (Mock(), [Mock(), Mock(), Mock()])
            
            # Mock similarity scores with one high score
            mock_similarity.return_value = [0.3, 0.8, 0.4]  # Second document has high similarity
            
            mock_context.return_value = (
                "Formatted context from multiple docs",
                ["src2", "src1", "src3"],
                [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
            )
            
            retriever = CognitiveRetriever()
            
            result = retriever.retrieve("Test query matching doc 2", "question")
            
            assert result["status"] == "ok"
            assert result["confidence"] == 0.8  # Max similarity
            assert len(result["source_ids"]) == 3
            assert len(result["knowledge_pairs"]) == 3

    def test_retrieve_with_different_intents(self):
        """Test retrieve with various non-short-circuit intents"""
        with patch('src.cognitive.retrieval.initialize_vector_store') as mock_init, \
             patch('src.cognitive.retrieval.Neo4jClient') as mock_client, \
             patch('src.cognitive.retrieval.generate_embeddings') as mock_embeddings, \
             patch('src.cognitive.retrieval.calculate_similarity') as mock_similarity, \
             patch('src.cognitive.retrieval.process_context') as mock_context:
            
            mock_retriever = Mock()
            mock_doc = Mock()
            mock_retriever.get_relevant_documents.return_value = [mock_doc]
            
            mock_init.return_value = (Mock(), Mock(), mock_retriever)
            mock_client.return_value = Mock()
            
            # Mock embeddings to return two values
            mock_embeddings.return_value = (Mock(), [Mock()])
            
            mock_similarity.return_value = [SIMILARITY_THRESHOLD + 0.1]
            mock_context.return_value = ("Context", ["src1"], [("Q", "A")])
            
            retriever = CognitiveRetriever()
            
            # Test various intents that should NOT short-circuit
            non_short_circuit_intents = [
                "question",
                "clarification", 
                "feedback",
                "unknown_intent",
                "technical_question"
            ]
            
            for intent in non_short_circuit_intents:
                result = retriever.retrieve(f"Test query for {intent}", intent)
                assert result["status"] == "ok"
                
                # Reset mocks for next iteration
                mock_retriever.get_relevant_documents.reset_mock()
                mock_embeddings.reset_mock()
                mock_similarity.reset_mock()
                mock_context.reset_mock()
                
                # Re-set return values
                mock_embeddings.return_value = (Mock(), [Mock()])
                mock_similarity.return_value = [SIMILARITY_THRESHOLD + 0.1]
                mock_context.return_value = ("Context", ["src1"], [("Q", "A")])


if __name__ == "__main__":
    # Simple test runner
    print("Running Cognitive Retriever Tests...")
    print("=" * 60)
    
    test_instance = TestCognitiveRetriever()
    
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