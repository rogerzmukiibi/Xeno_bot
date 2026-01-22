"""
Simple test file for reasoning.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cognitive.reasoning import CognitiveReasoner


class TestCognitiveReasoner:
    """Test suite for CognitiveReasoner class"""
    
    def test_init_creates_instance(self):
        """Test that CognitiveReasoner initializes properly"""
        reasoner = CognitiveReasoner()
        assert reasoner is not None
        assert isinstance(reasoner, CognitiveReasoner)

    def test_reason_greeting_intent_responds_directly(self):
        """Test reason returns respond_directly for greeting intent"""
        reasoner = CognitiveReasoner()
        
        query = "Hello"
        intent = "greeting"
        retrieval_result = {
            "status": "empty",
            "reason": "No knowledge found",
            "confidence": 0.0,
            "formatted_context": "",
            "source_ids": [],
            "knowledge_pairs": []
        }
        
        result = reasoner.reason(query, intent, retrieval_result)
        
        assert result["action"] == "respond_directly"
        assert result["message"] == "Intent-based response"
        assert result["confidence"] == 1.0
        assert result["context"] == ""
        assert result["source_ids"] == []
        assert result["knowledge_pairs"] == []

    def test_reason_small_talk_intent_responds_directly(self):
        """Test reason returns respond_directly for small_talk intent"""
        reasoner = CognitiveReasoner()
        
        query = "How are you?"
        intent = "small_talk"
        retrieval_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.8,
            "formatted_context": "Some context",
            "source_ids": ["src1"],
            "knowledge_pairs": [("question", "answer")]
        }
        
        result = reasoner.reason(query, intent, retrieval_result)
        
        assert result["action"] == "respond_directly"
        assert result["message"] == "Intent-based response"
        assert result["confidence"] == 1.0

    def test_reason_empty_retrieval_asks_for_clarification(self):
        """Test reason returns ask_for_clarification for empty retrieval"""
        reasoner = CognitiveReasoner()
        
        query = "What is quantum physics?"
        intent = "question"
        retrieval_result = {
            "status": "empty",
            "reason": "No knowledge found for this query",
            "confidence": 0.0,
            "formatted_context": "",
            "source_ids": [],
            "knowledge_pairs": []
        }
        
        result = reasoner.reason(query, intent, retrieval_result)
        
        assert result["action"] == "ask_for_clarification"
        assert result["message"] == "No knowledge found for this query"
        assert result["confidence"] == 0.0
        assert "No knowledge" in result["message"]

    def test_reason_low_confidence_declines_answer(self):
        """Test reason returns decline_answer for low confidence"""
        reasoner = CognitiveReasoner()
        
        query = "What is AI?"
        intent = "question"
        retrieval_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.3,  # Below 0.5 threshold
            "formatted_context": "AI is artificial intelligence",
            "source_ids": ["src1"],
            "knowledge_pairs": [("What is AI?", "Artificial Intelligence")]
        }
        
        result = reasoner.reason(query, intent, retrieval_result)
        
        assert result["action"] == "decline_answer"
        assert result["message"] == "Insufficient confidence in retrieved knowledge."
        assert result["confidence"] == 0.3

    def test_reason_high_confidence_generates_answer(self):
        """Test reason returns generate_answer for high confidence"""
        reasoner = CognitiveReasoner()
        
        query = "What is Python?"
        intent = "question"
        retrieval_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.9,  # Above 0.5 threshold
            "formatted_context": "Python is a programming language",
            "source_ids": ["src1", "src2"],
            "knowledge_pairs": [
                ("What is Python?", "A programming language"),
                ("Python features", "Easy to learn")
            ]
        }
        
        result = reasoner.reason(query, intent, retrieval_result)
        
        assert result["action"] == "generate_answer"
        assert result["message"] == "Sufficient knowledge retrieved."
        assert result["confidence"] == 0.9
        assert result["context"] == "Python is a programming language"
        assert result["source_ids"] == ["src1", "src2"]
        assert len(result["knowledge_pairs"]) == 2

    def test_reason_exact_threshold_generates_answer(self):
        """Test reason handles exact confidence threshold (0.5)"""
        reasoner = CognitiveReasoner()
        
        query = "Test query"
        intent = "question"
        retrieval_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.5,  # Exactly at threshold
            "formatted_context": "Test context",
            "source_ids": ["src1"],
            "knowledge_pairs": [("Q", "A")]
        }
        
        result = reasoner.reason(query, intent, retrieval_result)
        
        # 0.5 should be accepted (>= 0.5)
        assert result["action"] == "generate_answer"
        assert result["confidence"] == 0.5

    def test_reason_edge_cases(self):
        """Test reason handles various edge cases"""
        reasoner = CognitiveReasoner()
        
        # Test with minimal retrieval result
        minimal_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.8,
            "formatted_context": "",
            "source_ids": [],
            "knowledge_pairs": []
        }
        
        result = reasoner.reason("test", "question", minimal_result)
        assert result["action"] == "generate_answer"
        assert result["source_ids"] == []
        assert result["knowledge_pairs"] == []

    def test_decision_static_method(self):
        """Test the _decision static method directly"""
        result = CognitiveReasoner._decision(
            action="test_action",
            message="Test message",
            confidence=0.75,
            context="Test context",
            source_ids=["id1", "id2"],
            knowledge_pairs=[("q1", "a1"), ("q2", "a2")]
        )
        
        assert result["action"] == "test_action"
        assert result["message"] == "Test message"
        assert result["confidence"] == 0.75
        assert result["context"] == "Test context"
        assert result["source_ids"] == ["id1", "id2"]
        assert result["knowledge_pairs"] == [("q1", "a1"), ("q2", "a2")]

    def test_decision_static_method_defaults(self):
        """Test _decision static method with default parameters"""
        result = CognitiveReasoner._decision(
            action="default_test",
            message="Default test",
            confidence=0.5
        )
        
        assert result["action"] == "default_test"
        assert result["message"] == "Default test"
        assert result["confidence"] == 0.5
        assert result["context"] == ""  # Default
        assert result["source_ids"] == []  # Default
        assert result["knowledge_pairs"] == []  # Default

    def test_reason_priority_order(self):
        """Test that reasoning follows correct priority order"""
        reasoner = CognitiveReasoner()
        
        # Test 1: Intent should win even with good retrieval
        retrieval_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.9,
            "formatted_context": "Good context",
            "source_ids": ["src1"],
            "knowledge_pairs": [("Q", "A")]
        }
        
        result = reasoner.reason("Hi there", "greeting", retrieval_result)
        assert result["action"] == "respond_directly"  # Intent takes priority
        
        # Test 2: Empty retrieval should win over low confidence check
        empty_result = {
            "status": "empty",
            "reason": "No data",
            "confidence": 0.1,  # Also low confidence
            "formatted_context": "",
            "source_ids": [],
            "knowledge_pairs": []
        }
        
        result = reasoner.reason("Test", "question", empty_result)
        assert result["action"] == "ask_for_clarification"  # Empty takes priority

    def test_reason_with_different_intents(self):
        """Test reason with various intent types"""
        reasoner = CognitiveReasoner()
        
        test_cases = [
            ("greeting", "respond_directly"),
            ("small_talk", "respond_directly"),
            ("question", "generate_answer"),  # With good retrieval
            ("clarification", "generate_answer"),
            ("feedback", "generate_answer")
        ]
        
        retrieval_result = {
            "status": "success",
            "reason": "",
            "confidence": 0.8,
            "formatted_context": "Context",
            "source_ids": ["src1"],
            "knowledge_pairs": [("Q", "A")]
        }
        
        for intent, expected_action in test_cases:
            result = reasoner.reason("Test query", intent, retrieval_result)
            
            if intent in ["greeting", "small_talk"]:
                assert result["action"] == "respond_directly"
            else:
                assert result["action"] == "generate_answer"

    def test_reason_confidence_ranges(self):
        """Test reason with various confidence values"""
        reasoner = CognitiveReasoner()
        
        confidence_test_cases = [
            (0.0, "ask_for_clarification"),  # Empty status
            (0.1, "decline_answer"),  # Low confidence
            (0.3, "decline_answer"),
            (0.49, "decline_answer"),
            (0.5, "generate_answer"),  # At threshold
            (0.51, "generate_answer"),
            (0.8, "generate_answer"),
            (1.0, "generate_answer")
        ]
        
        for confidence, expected_action in confidence_test_cases:
            # For non-empty, non-intent cases
            retrieval_result = {
                "status": "success",
                "reason": "",
                "confidence": confidence,
                "formatted_context": f"Context with confidence {confidence}",
                "source_ids": ["src1"],
                "knowledge_pairs": [("Question", f"Answer for {confidence}")]
            }
            
            result = reasoner.reason("Test", "question", retrieval_result)
            
            if confidence < 0.5:
                assert result["action"] == "decline_answer"
            else:
                assert result["action"] == "generate_answer"


if __name__ == "__main__":
    # Simple test runner
    print("Running Cognitive Reasoner Tests...")
    print("=" * 60)
    
    test_instance = TestCognitiveReasoner()
    
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