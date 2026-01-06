"""
Simple test file for schema.py - FIXED VERSION
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import Mock, patch
import re
from src.graph.schema import XenoGraphSchema


class TestXenoGraphSchema:
    """Test suite for XenoGraphSchema class"""
    
    def test_init_creates_neo4j_client(self):
        """Test that XenoGraphSchema initializes with Neo4j client"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            
            mock_client.assert_called_once()
            assert schema.client == mock_client_instance

    def test_create_constraints(self):
        """Test create_constraints executes all constraint queries"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            schema.create_constraints()
            
            # Should have called run_query 4 times (one for each constraint)
            assert mock_client_instance.run_query.call_count == 4
            
            # Get all the queries that were called
            calls = mock_client_instance.run_query.call_args_list
            
            # Check each query contains expected patterns
            queries = [call[0][0] for call in calls]
            
            # Check constraint queries
            assert any("CREATE CONSTRAINT kb_question_id" in q for q in queries)
            assert any("CREATE CONSTRAINT section_name" in q for q in queries)
            assert any("CREATE CONSTRAINT tag_name" in q for q in queries)
            assert any("CREATE CONSTRAINT user_session" in q for q in queries)
            
            # Check they all have IF NOT EXISTS
            for q in queries:
                assert "IF NOT EXISTS" in q

    def test_create_indexes(self):
        """Test create_indexes executes all index queries"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            schema.create_indexes()
            
            # Should have called run_query 3 times (one for each index)
            assert mock_client_instance.run_query.call_count == 3
            
            # Get all the queries that were called
            calls = mock_client_instance.run_query.call_args_list
            queries = [call[0][0] for call in calls]
            
            # Check index queries
            assert any("CREATE INDEX question_text" in q for q in queries)
            assert any("CREATE INDEX question_embedding" in q for q in queries)
            assert any("CREATE INDEX memory_timestamp" in q for q in queries)
            
            # Check they all have IF NOT EXISTS
            for q in queries:
                assert "IF NOT EXISTS" in q

    def test_initialize_calls_both_methods(self):
        """Test initialize calls both create_constraints and create_indexes"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            
            # Mock the two methods
            with patch.object(schema, 'create_constraints') as mock_constraints, \
                 patch.object(schema, 'create_indexes') as mock_indexes:
                
                schema.initialize()
                
                # Both methods should be called
                mock_constraints.assert_called_once()
                mock_indexes.assert_called_once()

    def test_initialize_prints_messages(self):
        """Test initialize prints appropriate messages"""
        with patch('src.graph.schema.Neo4jClient') as mock_client, \
             patch('builtins.print') as mock_print:
            
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            
            # Mock the two methods
            with patch.object(schema, 'create_constraints'), \
                 patch.object(schema, 'create_indexes'):
                
                schema.initialize()
                
                # Check that print was called with expected messages
                print_calls = [call[0][0] for call in mock_print.call_args_list]
                
                assert any("🔧 Creating Neo4j schema..." in str(call) for call in print_calls)
                assert any("✅ Neo4j schema initialized successfully." in str(call) for call in print_calls)

    def test_close_closes_client(self):
        """Test close method closes the client"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            schema.close()
            
            mock_client_instance.close.assert_called_once()

    def test_constraint_queries_content(self):
        """Test the actual content of constraint queries"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            captured_queries = []
            
            def capture_query(query):
                captured_queries.append(query)
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_query
            
            schema = XenoGraphSchema()
            schema.create_constraints()
            
            # Check we have 4 queries
            assert len(captured_queries) == 4
            
            # Check each query in detail
            question_constraint = captured_queries[0]
            assert "CREATE CONSTRAINT kb_question_id IF NOT EXISTS" in question_constraint
            assert "FOR (q:Question)" in question_constraint
            assert "REQUIRE q.id IS UNIQUE" in question_constraint
            
            section_constraint = captured_queries[1]
            assert "CREATE CONSTRAINT section_name IF NOT EXISTS" in section_constraint
            assert "FOR (s:Section)" in section_constraint
            assert "REQUIRE s.name IS UNIQUE" in section_constraint
            
            tag_constraint = captured_queries[2]
            assert "CREATE CONSTRAINT tag_name IF NOT EXISTS" in tag_constraint
            assert "FOR (t:Tag)" in tag_constraint
            assert "REQUIRE t.name IS UNIQUE" in tag_constraint
            
            user_constraint = captured_queries[3]
            assert "CREATE CONSTRAINT user_session IF NOT EXISTS" in user_constraint
            assert "FOR (u:User)" in user_constraint
            assert "REQUIRE u.session_id IS UNIQUE" in user_constraint

    def test_index_queries_content(self):
        """Test the actual content of index queries"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            captured_queries = []
            
            def capture_query(query):
                captured_queries.append(query)
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_query
            
            schema = XenoGraphSchema()
            schema.create_indexes()
            
            # Check we have 3 queries
            assert len(captured_queries) == 3
            
            # Check each query in detail
            text_index = captured_queries[0]
            assert "CREATE INDEX question_text IF NOT EXISTS" in text_index
            assert "FOR (q:Question)" in text_index
            assert "ON (q.text)" in text_index
            
            embedding_index = captured_queries[1]
            assert "CREATE INDEX question_embedding IF NOT EXISTS" in embedding_index
            assert "FOR (q:Question)" in embedding_index
            assert "ON (q.embedding)" in embedding_index
            
            timestamp_index = captured_queries[2]
            assert "CREATE INDEX memory_timestamp IF NOT EXISTS" in timestamp_index
            assert "FOR (m:Memory)" in timestamp_index
            assert "ON (m.timestamp)" in timestamp_index

    def test_create_constraints_handles_exceptions(self):
        """Test create_constraints raises exceptions when queries fail"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Make the second call raise an exception
            call_count = 0
            
            def side_effect_query(query):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second query fails
                    raise Exception("Constraint already exists")
                return Mock()
            
            mock_client_instance.run_query.side_effect = side_effect_query
            
            schema = XenoGraphSchema()
            
            try:
                schema.create_constraints()
                # Should raise exception
                assert False, "Should have raised an exception when query failed"
            except Exception as e:
                # Should raise the exception
                assert "Constraint already exists" in str(e)
                # Should have only tried 2 queries before failing
                assert call_count == 2

    def test_create_indexes_handles_exceptions(self):
        """Test create_indexes raises exceptions when queries fail"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            # Make the second call raise an exception
            call_count = 0
            
            def side_effect_query(query):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second query fails
                    raise Exception("Index already exists")
                return Mock()
            
            mock_client_instance.run_query.side_effect = side_effect_query
            
            schema = XenoGraphSchema()
            
            try:
                schema.create_indexes()
                # Should raise exception
                assert False, "Should have raised an exception when query failed"
            except Exception as e:
                # Should raise the exception
                assert "Index already exists" in str(e)
                # Should have only tried 2 queries before failing
                assert call_count == 2

    def test_initialize_order(self):
        """Test that initialize calls constraints before indexes"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            
            call_order = []
            
            def record_constraints():
                call_order.append('constraints')
            
            def record_indexes():
                call_order.append('indexes')
            
            with patch.object(schema, 'create_constraints', side_effect=record_constraints), \
                 patch.object(schema, 'create_indexes', side_effect=record_indexes):
                
                schema.initialize()
                
                # Constraints should be called before indexes
                assert call_order == ['constraints', 'indexes']

    def test_schema_initialization_complete_flow(self):
        """Test complete schema initialization flow"""
        with patch('src.graph.schema.Neo4jClient') as mock_client, \
             patch('builtins.print') as mock_print:
            
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            schema = XenoGraphSchema()
            
            # Count run_query calls
            constraint_calls = 0
            index_calls = 0
            
            def count_run_query(query):
                nonlocal constraint_calls, index_calls
                if "CONSTRAINT" in query:
                    constraint_calls += 1
                elif "INDEX" in query:
                    index_calls += 1
                return Mock()
            
            mock_client_instance.run_query.side_effect = count_run_query
            
            schema.initialize()
            
            # Should have created 4 constraints and 3 indexes
            assert constraint_calls == 4
            assert index_calls == 3
            
            # Check print messages
            print_calls = [call[0][0] for call in mock_print.call_args_list]
            assert "🔧 Creating Neo4j schema..." in print_calls[0]
            assert "✅ Neo4j schema initialized successfully." in print_calls[1]

    def test_constraint_for_missing_labels(self):
        """Test that constraints are created for all expected node labels"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            captured_queries = []
            
            def capture_query(query):
                captured_queries.append(query)
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_query
            
            schema = XenoGraphSchema()
            schema.create_constraints()
            
            # Extract node labels from constraint queries
            node_labels = []
            for query in captured_queries:
                # Find the FOR (x:Label) pattern
                match = re.search(r'FOR\s*\(\w+:(\w+)\)', query)
                if match:
                    node_labels.append(match.group(1))
            
            # Should have constraints for all these labels
            expected_labels = ['Question', 'Section', 'Tag', 'User']
            assert set(node_labels) == set(expected_labels)

    def test_index_for_missing_labels(self):
        """Test that indexes are created for all expected node labels"""
        with patch('src.graph.schema.Neo4jClient') as mock_client:
            mock_client_instance = Mock()
            mock_client.return_value = mock_client_instance
            
            captured_queries = []
            
            def capture_query(query):
                captured_queries.append(query)
                return Mock()
            
            mock_client_instance.run_query.side_effect = capture_query
            
            schema = XenoGraphSchema()
            schema.create_indexes()
            
            # Extract node labels from index queries
            node_labels = []
            for query in captured_queries:
                # Find the FOR (x:Label) pattern
                match = re.search(r'FOR\s*\(\w+:(\w+)\)', query)
                if match:
                    node_labels.append(match.group(1))
            
            # Should have indexes for these labels
            expected_labels = ['Question', 'Question', 'Memory']  # Question appears twice
            assert node_labels == expected_labels


if __name__ == "__main__":
    # Simple test runner
    print("Running Xeno Graph Schema Tests...")
    print("=" * 60)
    
    test_instance = TestXenoGraphSchema()
    
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