"""
Unit tests for neo4j_client module
Tests Neo4j database connection and query execution
"""
import unittest
import sys
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Add the parent directory to sys.path to find src module
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestNeo4jClient(unittest.TestCase):
    """Test cases for neo4j_client module"""
    
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def tearDown(self):
        """Clean up test fixtures"""
        pass
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_initialization(self, mock_driver_constructor):
        """Test Neo4jClient initialization"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mock
        mock_driver = Mock()
        mock_driver_constructor.return_value = mock_driver
        
        # Create client
        client = Neo4jClient()
        
        # Verify driver was created with correct parameters
        mock_driver_constructor.assert_called_once_with(
            'bolt://localhost:7687',
            auth=('neo4j', 'password')
        )
        
        # Verify driver is stored
        self.assertEqual(client.driver, mock_driver)
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_close(self, mock_driver_constructor):
        """Test closing the Neo4j connection"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mock
        mock_driver = Mock()
        mock_driver_constructor.return_value = mock_driver
        
        # Create client and close it
        client = Neo4jClient()
        client.close()
        
        # Verify driver.close() was called
        mock_driver.close.assert_called_once()
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_close_without_driver(self, mock_driver_constructor):
        """Test closing when driver is None"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        mock_driver = Mock()
        mock_driver_constructor.return_value = mock_driver
        
        client = Neo4jClient()
        client.driver = None  # Simulate driver being None
        client.close()
        
        # Should not raise an error
        mock_driver.close.assert_not_called()
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_run_query_with_parameters(self, mock_driver_constructor):
        """Test running a query with parameters"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mocks
        mock_driver = Mock()
        mock_session = MagicMock()  # Use MagicMock for session
        mock_result = MagicMock()   # Use MagicMock for result
        
        mock_driver_constructor.return_value = mock_driver
        
        # Mock the session context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_driver.session.return_value = context_manager
        
        mock_session.run.return_value = mock_result
        
        # Mock result records
        mock_record1 = Mock()
        mock_record1.data.return_value = {'id': 1, 'name': 'Alice'}
        mock_record2 = Mock()
        mock_record2.data.return_value = {'id': 2, 'name': 'Bob'}
        mock_result.__iter__.return_value = iter([mock_record1, mock_record2])
        
        # Create client and run query
        client = Neo4jClient()
        query = "MATCH (p:Person) WHERE p.age > $age RETURN p"
        parameters = {'age': 30}
        results = client.run_query(query, parameters)
        
        # Verify session.run was called with correct query and parameters
        mock_session.run.assert_called_once_with(query, parameters)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0], {'id': 1, 'name': 'Alice'})
        self.assertEqual(results[1], {'id': 2, 'name': 'Bob'})
        
        # Verify context manager was used
        mock_driver.session.assert_called_once()
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_run_query_without_parameters(self, mock_driver_constructor):
        """Test running a query without parameters"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mocks
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_driver_constructor.return_value = mock_driver
        
        # Mock the session context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_driver.session.return_value = context_manager
        
        mock_session.run.return_value = mock_result
        
        # Mock empty result
        mock_result.__iter__.return_value = iter([])
        
        # Create client and run query
        client = Neo4jClient()
        query = "MATCH (n) RETURN n LIMIT 10"
        results = client.run_query(query)
        
        # Verify session.run was called with empty dict as parameters
        mock_session.run.assert_called_once_with(query, {})
        
        # Verify empty results
        self.assertEqual(results, [])
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_run_query_empty_result(self, mock_driver_constructor):
        """Test running a query that returns no results"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mocks
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_driver_constructor.return_value = mock_driver
        
        # Mock the session context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_driver.session.return_value = context_manager
        
        mock_session.run.return_value = mock_result
        
        # Mock empty result
        mock_result.__iter__.return_value = iter([])
        
        # Create client and run query
        client = Neo4jClient()
        results = client.run_query("MATCH (n) WHERE n.nonexistent = true RETURN n")
        
        # Verify empty results
        self.assertEqual(results, [])
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_run_query_single_result(self, mock_driver_constructor):
        """Test running a query that returns a single result"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mocks
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_driver_constructor.return_value = mock_driver
        
        # Mock the session context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_driver.session.return_value = context_manager
        
        mock_session.run.return_value = mock_result
        
        # Mock single result
        mock_record = Mock()
        mock_record.data.return_value = {'count': 42}
        mock_result.__iter__.return_value = iter([mock_record])
        
        # Create client and run query
        client = Neo4jClient()
        results = client.run_query("MATCH (n) RETURN count(n) as count")
        
        # Verify single result
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], {'count': 42})
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_run_query_complex_parameters(self, mock_driver_constructor):
        """Test running a query with complex parameters"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mocks
        mock_driver = Mock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        mock_driver_constructor.return_value = mock_driver
        
        # Mock the session context manager
        context_manager = MagicMock()
        context_manager.__enter__.return_value = mock_session
        context_manager.__exit__.return_value = None
        mock_driver.session.return_value = context_manager
        
        mock_session.run.return_value = mock_result
        
        # Mock empty result
        mock_result.__iter__.return_value = iter([])
        
        # Create client and run query with complex parameters
        client = Neo4jClient()
        query = """
            MATCH (p:Person)
            WHERE p.name = $name AND p.age IN $ages
            RETURN p
        """
        parameters = {
            'name': 'John Doe',
            'ages': [25, 30, 35],
            'active': True,
            'score': 95.5
        }
        results = client.run_query(query, parameters)
        
        # Verify session.run was called with correct complex parameters
        mock_session.run.assert_called_once_with(query, parameters)
    
    @patch('src.neo4j_client.GraphDatabase.driver')
    @patch('src.neo4j_client.NEO4J_URI', 'bolt://localhost:7687')
    @patch('src.neo4j_client.NEO4J_USER', 'neo4j')
    @patch('src.neo4j_client.NEO4J_PASSWORD', 'password')
    def test_multiple_queries(self, mock_driver_constructor):
        """Test running multiple queries"""
        # Import here to ensure patches are applied
        from src.neo4j_client import Neo4jClient
        
        # Setup mocks
        mock_driver = Mock()
        mock_driver_constructor.return_value = mock_driver
        
        # Create client
        client = Neo4jClient()
        
        # First query setup
        mock_session1 = MagicMock()
        mock_result1 = MagicMock()
        context_manager1 = MagicMock()
        context_manager1.__enter__.return_value = mock_session1
        context_manager1.__exit__.return_value = None
        
        # Second query setup
        mock_session2 = MagicMock()
        mock_result2 = MagicMock()
        context_manager2 = MagicMock()
        context_manager2.__enter__.return_value = mock_session2
        context_manager2.__exit__.return_value = None
        
        # Make driver.session return different context managers on each call
        mock_driver.session.side_effect = [context_manager1, context_manager2]
        
        # Mock results for different queries
        mock_record1 = Mock()
        mock_record1.data.return_value = {'id': 1}
        mock_result1.__iter__.return_value = iter([mock_record1])
        mock_session1.run.return_value = mock_result1
        
        mock_record2 = Mock()
        mock_record2.data.return_value = {'id': 2}
        mock_result2.__iter__.return_value = iter([mock_record2])
        mock_session2.run.return_value = mock_result2
        
        # First query
        results1 = client.run_query("QUERY 1")
        self.assertEqual(results1, [{'id': 1}])
        
        # Second query
        results2 = client.run_query("QUERY 2")
        self.assertEqual(results2, [{'id': 2}])
        
        # Verify both queries were executed
        self.assertEqual(mock_driver.session.call_count, 2)
        self.assertEqual(mock_session1.run.call_count, 1)
        self.assertEqual(mock_session2.run.call_count, 1)


if __name__ == '__main__':
    unittest.main()