"""
Unit tests for vector_store module
Tests ChromaDB vector store operations
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

from src.vector_store import (_calculate_similarity_impl,
                              _generate_embeddings_impl, _process_context_impl,
                              calculate_similarity, generate_embeddings,
                              process_context)


class TestVectorStore(unittest.TestCase):
    """Test cases for vector_store module"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock document
        self.mock_doc = Mock()
        self.mock_doc.page_content = "Test document content"
        self.mock_doc.metadata = {
            "id": "KB001",
            "question": "Test question?",
            "content": "Test answer.",
            "section": "Test",
        }

        self.mock_documents = [self.mock_doc]

    @patch("src.vector_store.genai_client")
    def test_generate_embeddings_impl(self, mock_genai_client):
        """Test internal embedding generation implementation"""
        # Mock embeddings for query and document
        mock_query_embedding = Mock()
        mock_query_embedding.values = [0.1, 0.2, 0.3]
        mock_doc_embedding = Mock()
        mock_doc_embedding.values = [0.2, 0.3, 0.4]
        
        # Setup side effect for multiple calls
        call_count = [0]
        def embed_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_response = Mock()
            if call_count[0] == 1:
                mock_response.embeddings = [mock_query_embedding]
            else:
                mock_response.embeddings = [mock_doc_embedding]
            return mock_response
        
        mock_genai_client.models.embed_content.side_effect = embed_side_effect

        query = "Test query"
        query_emb, doc_embs = _generate_embeddings_impl(query, self.mock_documents)

        # Verify embed_content was called correctly
        self.assertEqual(mock_genai_client.models.embed_content.call_count, 2)

        # Verify embeddings
        self.assertEqual(query_emb, [0.1, 0.2, 0.3])
        self.assertEqual(len(doc_embs), 1)
        self.assertEqual(doc_embs[0], [0.2, 0.3, 0.4])

    @patch("src.vector_store.genai_client")
    def test_generate_embeddings_with_timer(self, mock_genai_client):
        """Test embedding generation with timer"""
        # Mock embeddings
        mock_embedding = Mock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.embeddings = [mock_embedding]
        mock_genai_client.models.embed_content.return_value = mock_response

        mock_timer = Mock()
        mock_timer.time_step = MagicMock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()

        generate_embeddings("Test", self.mock_documents, timer=mock_timer)

        # Verify timer was used
        mock_timer.time_step.assert_called_once_with("embedding_generation")

    @patch("src.vector_store.genai_client")
    def test_generate_embeddings_multiple_docs(self, mock_genai_client):
        """Test embedding generation with multiple documents"""
        # Create multiple mock documents
        mock_doc2 = Mock()
        mock_doc2.page_content = "Second document"
        docs = [self.mock_doc, mock_doc2]

        # Mock embeddings
        mock_query_emb = Mock()
        mock_query_emb.values = [0.1, 0.2, 0.3]
        mock_doc1_emb = Mock()
        mock_doc1_emb.values = [0.2, 0.3, 0.4]
        mock_doc2_emb = Mock()
        mock_doc2_emb.values = [0.3, 0.4, 0.5]
        
        # First call for query, second call for both docs
        call_count = [0]
        def embed_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_response = Mock()
            if call_count[0] == 1:
                mock_response.embeddings = [mock_query_emb]
            else:
                mock_response.embeddings = [mock_doc1_emb, mock_doc2_emb]
            return mock_response
        
        mock_genai_client.models.embed_content.side_effect = embed_side_effect

        query_emb, doc_embs = _generate_embeddings_impl("Test", docs)

        # Should have 2 doc embeddings
        self.assertEqual(len(doc_embs), 2)
        self.assertEqual(mock_genai_client.models.embed_content.call_count, 2)

    def test_calculate_similarity_impl(self):
        """Test internal similarity calculation implementation"""
        query_embedding = [1.0, 0.0, 0.0]
        doc_embeddings = [
            [1.0, 0.0, 0.0],  # Same as query - score should be ~1.0
            [0.0, 1.0, 0.0],  # Orthogonal - score should be ~0.0
            [0.5, 0.5, 0.0],  # Partial similarity
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
                "id": f"KB00{i+1}",
                "question": f"Question {i+1}?",
                "content": f"Answer {i+1}.",
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
        self.assertEqual(source_ids[0], "KB003")
        self.assertEqual(knowledge_pairs[0][0], "Question 3?")

        # Check formatted context
        self.assertIn("Knowledge Entry 1:", context)
        self.assertIn("Knowledge Entry 2:", context)
        self.assertIn("Q: Question 3?", context)
        self.assertIn("A: Answer 3.", context)

    def test_process_context_with_timer(self):
        """Test context processing with timer"""
        mock_result = Mock()
        mock_result.metadata = {"id": "KB001", "question": "Q?", "content": "A."}

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
                "id": f"KB00{i}",
                "question": f"Q{i}?",
                "content": f"A{i}.",
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
            "id": "KB001",
            "question": "Test question?",
            "content": "Test answer.",
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

    @patch("src.vector_store.get_knowledge_base_data")
    @patch("src.vector_store.chromadb.PersistentClient")
    @patch("src.vector_store.Chroma")
    def test_initialize_vector_store_new_collection(
        self, mock_chroma_class, mock_client_class, mock_get_kb
    ):
        """Test initializing vector store with new collection"""
        # Mock knowledge base data
        mock_get_kb.return_value = (
            ["doc1", "doc2"],
            [{"id": "1"}, {"id": "2"}],
            ["id1", "id2"],
        )

        # Mock ChromaDB client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate collection doesn't exist (raises exception)
        mock_client.get_collection.side_effect = Exception("Collection not found")
        
        # Mock create_collection
        mock_collection = Mock()
        mock_client.create_collection.return_value = mock_collection

        # Mock Chroma vector store
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma_class.return_value = mock_vector_store

        # Call function
        from src.vector_store import initialize_vector_store

        collection, vector_store, retriever = initialize_vector_store()

        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        mock_collection.add.assert_called_once()

        # Verify vector store and retriever
        self.assertEqual(vector_store, mock_vector_store)
        self.assertEqual(retriever, mock_retriever)

    @patch("src.vector_store.get_knowledge_base_data")
    @patch("src.vector_store.chromadb.PersistentClient")
    @patch("src.vector_store.Chroma")
    def test_initialize_vector_store_existing_collection(
        self, mock_chroma_class, mock_client_class, mock_get_kb
    ):
        """Test initializing vector store with existing collection"""
        # Mock knowledge base data
        mock_get_kb.return_value = (
            ["doc1", "doc2"],
            [{"id": "1"}, {"id": "2"}],
            ["id1", "id2"],
        )

        # Mock ChromaDB client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate collection exists
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection

        # Mock Chroma vector store
        mock_vector_store = Mock()
        mock_retriever = Mock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        mock_chroma_class.return_value = mock_vector_store

        # Call function
        from src.vector_store import initialize_vector_store

        collection, vector_store, retriever = initialize_vector_store()

        # Verify existing collection was loaded (not created)
        mock_client.get_collection.assert_called_once()
        mock_client.create_collection.assert_not_called()

        # Verify vector store and retriever
        self.assertEqual(collection, mock_collection)
        self.assertEqual(vector_store, mock_vector_store)
        self.assertEqual(retriever, mock_retriever)

    @patch("src.vector_store.get_knowledge_base_data")
    @patch("src.vector_store.chromadb.PersistentClient")
    def test_initialize_vector_store_failure(self, mock_client_class, mock_get_kb):
        """Test initialize_vector_store handles errors properly"""
        # Mock knowledge base data
        mock_get_kb.return_value = (["doc1"], [{"id": "1"}], ["id1"])

        # Mock client to raise exception
        mock_client_class.side_effect = Exception("Database connection failed")

        # Call function and expect exception
        from src.vector_store import initialize_vector_store

        with self.assertRaises(Exception) as context:
            initialize_vector_store()

        self.assertIn("Database connection failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
