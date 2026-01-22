"""
Simple test file for config.py - SIMPLIFIED VERSION
Just test the constants, not the runtime behavior
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class TestConfigConstants:
    """Test suite for config.py constants (not runtime loading)"""
    
    def test_ai_model_configuration(self):
        """Test AI model configuration constants"""
        from src.config import EMBEDDING_MODEL, LLM_MODEL_NAME
        assert EMBEDDING_MODEL == "models/embedding-001"
        assert LLM_MODEL_NAME == "models/gemini-1.5-flash"
    
    def test_database_configuration(self):
        """Test database configuration constants"""
        from src.config import COLLECTION_NAME, CHROMA_DB_PATH, SQLITE_DB_PATH
        assert COLLECTION_NAME == "xeno_collection"
        assert CHROMA_DB_PATH == "/tmp/xeno_db"
        assert SQLITE_DB_PATH == "xeno_memory.db"
    
    def test_rag_configuration(self):
        """Test RAG configuration constants"""
        from src.config import RAG_TOP_K, RAG_MAX_RESULTS, SIMILARITY_THRESHOLD
        assert RAG_TOP_K == 4
        assert RAG_MAX_RESULTS == 2
        assert SIMILARITY_THRESHOLD == 0.4
    
    def test_server_configuration(self):
        """Test server configuration constants"""
        from src.config import SERVER_NAME, SERVER_PORT
        assert SERVER_NAME == "0.0.0.0"
        assert SERVER_PORT == 7860
    
    def test_system_prompt(self):
        """Test system prompt configuration"""
        from src.config import SYSTEM_PROMPT
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0
        assert "XENO Support Assistant" in SYSTEM_PROMPT
    
    def test_google_sheets_settings(self):
        """Test Google Sheets settings constants"""
        from src.config import SPREADSHEET_NAME, RESPONSE_SHEET_INDEX, TIMING_SHEET_NAME
        assert SPREADSHEET_NAME == "Response_Log"
        assert RESPONSE_SHEET_INDEX == 0
        assert TIMING_SHEET_NAME == "Timing_Log"
    
    def test_exported_variables(self):
        """Test that all expected variables are exported"""
        from src.config import __all__
        
        expected_exports = [
            # API Keys
            "GEMINI_API_KEY",
            
            # Neo4j
            "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_CONFIGURED",
            
            # Google Sheets
            "GOOGLE_SHEETS_CREDENTIALS", "GOOGLE_SHEETS_CREDENTIALS_PATH",
            "SPREADSHEET_NAME", "RESPONSE_SHEET_INDEX", "TIMING_SHEET_NAME",
            
            # Models
            "EMBEDDING_MODEL", "LLM_MODEL_NAME",
            
            # Databases
            "COLLECTION_NAME", "CHROMA_DB_PATH", "SQLITE_DB_PATH",
            
            # Knowledge Base
            "KNOWLEDGE_BASE_PATH",
            
            # RAG
            "RAG_TOP_K", "RAG_MAX_RESULTS", "SIMILARITY_THRESHOLD",
            
            # Server
            "SERVER_NAME", "SERVER_PORT",
            
            # Prompts
            "SYSTEM_PROMPT",
        ]
        
        # Check all expected exports are in __all__
        for export in expected_exports:
            assert export in __all__
    
    def test_log_status_function(self):
        """Test the log_status helper function"""
        from src.config import log_status
        
        # Test with mocked print
        import io
        import sys
        from unittest.mock import patch
        
        captured_output = io.StringIO()
        with patch('sys.stdout', new=captured_output):
            log_status("Test Item", True, "Success message")
            log_status("Test Item", False, "Failure message")
        
        output = captured_output.getvalue()
        assert "✅ Test Item: Success message" in output
        assert "❌ Test Item: Failure message" in output


if __name__ == "__main__":
    # Simple test runner
    print("Running Config Constants Tests...")
    print("=" * 60)
    
    test_instance = TestConfigConstants()
    
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