"""
Unit tests for knowledge_base module
Tests knowledge base loading and preparation
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.knowledge_base import (get_knowledge_base_data, load_knowledge_base,
                                prepare_documents)


class TestKnowledgeBase(unittest.TestCase):
    """Test cases for knowledge_base module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample knowledge base data
        self.sample_data = [
            {
                "ID": "KB001",
                "Question": "How do I create an account?",
                "Content": "You can create an account by visiting our website.",
                "Section": "Account Management",
                "Source": "Website",
                "Owner": "Support Team",
                "Tag": "account",
            },
            {
                "ID": "KB002",
                "Question": "What are the fees?",
                "Content": "Our transaction fees are 1% per transaction.",
                "Section": "Fees",
                "Source": "Documentation",
                "Owner": "Finance Team",
                "Tag": "fees",
            },
        ]

        # Create temporary JSON file
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        json.dump(self.sample_data, self.temp_file)
        self.temp_file.close()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_load_knowledge_base(self):
        """Test loading knowledge base from JSON file"""
        df = load_knowledge_base(self.temp_file.name)

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("ID", df.columns)
        self.assertIn("Question", df.columns)
        self.assertIn("Content", df.columns)

    def test_load_knowledge_base_drops_null_content(self):
        """Test that rows with null Content are dropped"""
        data_with_null = self.sample_data + [
            {
                "ID": "KB003",
                "Question": "Test question?",
                "Content": None,
                "Section": "Test",
            }
        ]

        temp_file_null = tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        )
        json.dump(data_with_null, temp_file_null)
        temp_file_null.close()

        try:
            df = load_knowledge_base(temp_file_null.name)
            # Should only have 2 rows (null Content row dropped)
            self.assertEqual(len(df), 2)
        finally:
            os.unlink(temp_file_null.name)

    def test_prepare_documents(self):
        """Test preparing documents for vector store"""
        documents, metadatas, ids = prepare_documents(self.sample_data)

        # Check lengths match
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(metadatas), 2)
        self.assertEqual(len(ids), 2)

        # Check document format
        self.assertIn("Question:", documents[0])
        self.assertIn("Answer:", documents[0])
        self.assertIn("How do I create an account?", documents[0])

        # Check metadata structure
        self.assertEqual(metadatas[0]["id"], "KB001")
        self.assertEqual(metadatas[0]["question"], "How do I create an account?")
        self.assertEqual(metadatas[0]["section"], "Account Management")

        # Check IDs
        self.assertEqual(ids[0], "KB001")
        self.assertEqual(ids[1], "KB002")

    def test_prepare_documents_with_missing_fields(self):
        """Test preparing documents with missing optional fields"""
        data_minimal = [
            {"ID": "KB001", "Question": "Test question?", "Content": "Test answer."}
        ]

        documents, metadatas, ids = prepare_documents(data_minimal)

        # Should still work with defaults
        self.assertEqual(len(documents), 1)
        self.assertEqual(metadatas[0]["section"], "")
        self.assertEqual(metadatas[0]["source"], "")
        self.assertEqual(metadatas[0]["owner"], "")
        self.assertEqual(metadatas[0]["tag"], "")

    @patch("src.knowledge_base.load_knowledge_base")
    def test_get_knowledge_base_data(self, mock_load):
        """Test get_knowledge_base_data function"""
        # Mock the load_knowledge_base function
        mock_df = pd.DataFrame(self.sample_data)
        mock_load.return_value = mock_df

        documents, metadatas, ids = get_knowledge_base_data()

        # Verify load was called
        mock_load.assert_called_once()

        # Verify output
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(metadatas), 2)
        self.assertEqual(len(ids), 2)

    def test_document_text_format(self):
        """Test that document text is properly formatted"""
        documents, _, _ = prepare_documents(self.sample_data)

        # Check first document format
        expected_format = "Question: How do I create an account?\nAnswer: You can create an account by visiting our website."
        self.assertEqual(documents[0], expected_format)

    def test_empty_knowledge_base(self):
        """Test handling of empty knowledge base"""
        empty_data = []
        documents, metadatas, ids = prepare_documents(empty_data)

        self.assertEqual(len(documents), 0)
        self.assertEqual(len(metadatas), 0)
        self.assertEqual(len(ids), 0)

    def test_metadata_completeness(self):
        """Test that all metadata fields are present"""
        _, metadatas, _ = prepare_documents(self.sample_data)

        required_fields = [
            "question",
            "content",
            "section",
            "source",
            "owner",
            "tag",
            "id",
        ]
        for metadata in metadatas:
            for field in required_fields:
                self.assertIn(field, metadata)

    @patch("src.knowledge_base.load_knowledge_base")
    def test_get_knowledge_base_data_with_exception(self, mock_load):
        """Test get_knowledge_base_data handles exceptions"""
        # Make load_knowledge_base raise an exception
        mock_load.side_effect = Exception("File not found")

        # Should raise the exception
        with self.assertRaises(Exception) as context:
            get_knowledge_base_data()

        self.assertIn("File not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
