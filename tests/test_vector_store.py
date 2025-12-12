"""
Unit tests for vector_store module
Tests ChromaDB vector store operations
"""
import unittest
import numpy as np
import torch
from unittest.mock import patch, Mock, MagicMock
from src.vector_store import (
    generate_embeddings,
    calculate_similarity,
    process_context,
    _generate_embeddings_impl,
    _calculate_similarity_impl,
    _process_context_impl
)


class TestVectorStore(unittest.TestCase):
    """Test cases for vector_store module"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock document
        self.mock_doc = Mock()
        self.mock_doc.page_content = "Test document content"
        self.mock_doc.metadata = {
            'id': 'KB001',
            'question': 'Test question?',
            'content': 'Test answer.',
            'section': 'Test'
        }
        
        self.mock_documents = [self.mock_doc]
    
    @patch('src.vector_store.genai.embed_content')
    def test_generate_embeddings_impl(self, mock_embed):
        """Test internal embedding generation implementation"""
        # Mock embeddings
        mock_embed.side_effect = [
            {'embedding': [0.1, 0.2, 0.3]},  # Query embedding
            {'embedding': [0.2, 0.3, 0.4]}   # Doc embedding
        ]
        
        query = "Test query"
        query_emb, doc_embs = _generate_embeddings_impl(query, self.mock_documents)
        
        # Verify embed_content was called correctly
        self.assertEqual(mock_embed.call_count, 2)
        
        # Check query embedding call
        first_call = mock_embed.call_args_list[0]
        self.assertEqual(first_call[1]['content'], query)
        self.assertEqual(first_call[1]['task_type'], 'retrieval_query')
        
        # Check doc embedding call
        second_call = mock_embed.call_args_list[1]
        self.assertEqual(second_call[1]['content'], self.mock_doc.page_content)
        self.assertEqual(second_call[1]['task_type'], 'retrieval_document')
        
        # Verify embeddings
        self.assertEqual(query_emb, [0.1, 0.2, 0.3])
        self.assertEqual(len(doc_embs), 1)
        self.assertEqual(doc_embs[0], [0.2, 0.3, 0.4])
    
    @patch('src.vector_store.genai.embed_content')
    def test_generate_embeddings_with_timer(self, mock_embed):
        """Test embedding generation with timer"""
        mock_embed.side_effect = [
            {'embedding': [0.1, 0.2, 0.3]},
            {'embedding': [0.2, 0.3, 0.4]}
        ]
        
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        generate_embeddings("Test", self.mock_documents, timer=mock_timer)
        
        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("embedding_generation")
    
    @patch('src.vector_store.genai.embed_content')
    def test_generate_embeddings_multiple_docs(self, mock_embed):
        """Test embedding generation with multiple documents"""
        # Create multiple mock documents
        mock_doc2 = Mock()
        mock_doc2.page_content = "Second document"
        docs = [self.mock_doc, mock_doc2]
        
        # Mock embeddings
        mock_embed.side_effect = [
            {'embedding': [0.1, 0.2, 0.3]},  # Query
            {'embedding': [0.2, 0.3, 0.4]},  # Doc 1
            {'embedding': [0.3, 0.4, 0.5]}   # Doc 2
        ]
        
        query_emb, doc_embs = _generate_embeddings_impl("Test", docs)
        
        # Should have 2 doc embeddings
        self.assertEqual(len(doc_embs), 2)
        self.assertEqual(mock_embed.call_count, 3)
    
    def test_calculate_similarity_impl(self):
        """Test internal similarity calculation implementation"""
        query_embedding = [1.0, 0.0, 0.0]
        doc_embeddings = [
            [1.0, 0.0, 0.0],  # Same as query - score should be ~1.0
            [0.0, 1.0, 0.0],  # Orthogonal - score should be ~0.0
            [0.5, 0.5, 0.0]   # Partial similarity
        ]
        
        scores = _calculate_similarity_impl(query_embedding, doc_embeddings)
        
        # Check scores
        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(scores[0], 1.0, places=5)
        self.assertAlmostEqual(scores[1], 0.0, places=5)
        self.assertGreater(scores[2], 0.0)
        self.assertLess(scores[2], 1.0)
    
    def test_calculate_similarity_with_timer(self):
        """Test similarity calculation with timer"""
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        query_emb = [1.0, 0.0, 0.0]
        doc_embs = [[1.0, 0.0, 0.0]]
        
        calculate_similarity(query_emb, doc_embs, timer=mock_timer)
        
        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("similarity_calculation")
    
    def test_process_context_impl(self):
        """Test internal context processing implementation"""
        # Create mock results with metadata
        results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.metadata = {
                'id': f'KB00{i+1}',
                'question': f'Question {i+1}?',
                'content': f'Answer {i+1}.'
            }
            results.append(mock_result)
        
        # Cosine scores (sorted: 0.9, 0.7, 0.5)
        cosine_scores = [0.7, 0.5, 0.9]
        
        context, source_ids, knowledge_pairs = _process_context_impl(
            results, cosine_scores, max_results=2
        )
        
        # Should return top 2 results
        self.assertEqual(len(source_ids), 2)
        self.assertEqual(len(knowledge_pairs), 2)
        
        # Check that highest score (0.9, index 2) is first
        self.assertEqual(source_ids[0], 'KB003')
        self.assertEqual(knowledge_pairs[0][0], 'Question 3?')
        
        # Check formatted context
        self.assertIn("Knowledge Entry 1:", context)
        self.assertIn("Knowledge Entry 2:", context)
        self.assertIn("Q: Question 3?", context)
        self.assertIn("A: Answer 3.", context)
    
    def test_process_context_with_timer(self):
        """Test context processing with timer"""
        mock_result = Mock()
        mock_result.metadata = {'id': 'KB001', 'question': 'Q?', 'content': 'A.'}
        
        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        process_context([mock_result], [0.9], timer=mock_timer)
        
        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("context_processing")
    
    def test_process_context_max_results(self):
        """Test that max_results parameter limits output"""
        # Create 5 mock results
        results = []
        for i in range(5):
            mock_result = Mock()
            mock_result.metadata = {
                'id': f'KB00{i}',
                'question': f'Q{i}?',
                'content': f'A{i}.'
            }
            results.append(mock_result)
        
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        # Request only 3 results
        context, source_ids, knowledge_pairs = _process_context_impl(
            results, scores, max_results=3
        )
        
        # Should only return 3
        self.assertEqual(len(source_ids), 3)
        self.assertEqual(len(knowledge_pairs), 3)
    
    def test_process_context_formatting(self):
        """Test context formatting details"""
        mock_result = Mock()
        mock_result.metadata = {
            'id': 'KB001',
            'question': 'Test question?',
            'content': 'Test answer.'
        }
        
        context, _, _ = _process_context_impl([mock_result], [0.9], max_results=1)
        
        # Check formatting
        self.assertIn("Knowledge Entry 1:", context)
        self.assertIn("Q: Test question?", context)
        self.assertIn("A: Test answer.", context)
        self.assertIn("-" * 40, context)
    
    def test_process_context_missing_metadata(self):
        """Test context processing with missing metadata fields"""
        mock_result = Mock()
        mock_result.metadata = {}  # No metadata
        
        context, source_ids, knowledge_pairs = _process_context_impl(
            [mock_result], [0.9], max_results=1
        )
        
        # Should handle missing fields with N/A
        self.assertIn("N/A", context)
        self.assertEqual(source_ids[0], "N/A")


if __name__ == '__main__':
    unittest.main()
