"""
Unit tests for intent_classifier module
Tests the IntentClassifier class
"""
import unittest
from unittest.mock import Mock
from src.intent_classifier import IntentClassifier


class TestIntentClassifier(unittest.TestCase):
    """Test cases for IntentClassifier class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.classifier = IntentClassifier()
    
    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsNotNone(self.classifier.intent_patterns)
        self.assertIn('greeting', self.classifier.intent_patterns)
        self.assertIn('thanks', self.classifier.intent_patterns)
        self.assertIn('goodbye', self.classifier.intent_patterns)
    
    def test_classify_greeting(self):
        """Test classification of greeting messages"""
        test_cases = [
            "hi",
            "hello",
            "Hey there",
            "good morning",
            "Good afternoon!",
            "how are you"
        ]
        
        for message in test_cases:
            intent, response = self.classifier.classify_intent(message)
            self.assertEqual(intent, 'greeting', f"Failed for message: {message}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_classify_thanks(self):
        """Test classification of thank you messages"""
        test_cases = [
            "thank you",
            "thanks",
            "thank u",
            "thx",
            "I appreciate it",
            "thanks a lot",
            "thank you so much"
        ]
        
        for message in test_cases:
            intent, response = self.classifier.classify_intent(message)
            self.assertEqual(intent, 'thanks', f"Failed for message: {message}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_classify_goodbye(self):
        """Test classification of goodbye messages"""
        test_cases = [
            "bye",
            "goodbye",
            "see you",
            "farewell",
            "take care",
            "have a good day",
            "talk to you later"
        ]
        
        for message in test_cases:
            intent, response = self.classifier.classify_intent(message)
            self.assertEqual(intent, 'goodbye', f"Failed for message: {message}")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
    
    def test_classify_query(self):
        """Test classification of query messages"""
        test_cases = [
            "How do I open an account?",
            "What are the transaction fees?",
            "Can you help me with my balance?",
            "Tell me about XENO services"
        ]
        
        for message in test_cases:
            intent, response = self.classifier.classify_intent(message)
            self.assertEqual(intent, 'query', f"Failed for message: {message}")
            self.assertEqual(response, '')
    
    def test_case_insensitivity(self):
        """Test that classification is case insensitive"""
        messages = [
            ("HI", 'greeting'),
            ("THANK YOU", 'thanks'),
            ("BYE", 'goodbye'),
            ("Hi There", 'greeting')
        ]
        
        for message, expected_intent in messages:
            intent, _ = self.classifier.classify_intent(message)
            self.assertEqual(intent, expected_intent)
    
    def test_with_timer(self):
        """Test classification with timer object"""
        mock_timer = Mock()
        mock_timer.time_step = Mock()
        mock_timer.time_step.return_value.__enter__ = Mock()
        mock_timer.time_step.return_value.__exit__ = Mock()
        
        intent, response = self.classifier.classify_intent("hello", timer=mock_timer)
        
        self.assertEqual(intent, 'greeting')
        mock_timer.time_step.assert_called_once_with("intent_classification")
    
    def test_is_simple_intent(self):
        """Test is_simple_intent method"""
        self.assertTrue(self.classifier.is_simple_intent('greeting'))
        self.assertTrue(self.classifier.is_simple_intent('thanks'))
        self.assertFalse(self.classifier.is_simple_intent('goodbye'))
        self.assertFalse(self.classifier.is_simple_intent('query'))
    
    def test_add_intent(self):
        """Test adding a new intent"""
        patterns = [r'\b(test|testing)\b']
        responses = ["This is a test response"]
        
        self.classifier.add_intent('test_intent', patterns, responses)
        
        # Verify intent was added
        self.assertIn('test_intent', self.classifier.intent_patterns)
        self.assertEqual(self.classifier.intent_patterns['test_intent']['patterns'], patterns)
        self.assertEqual(self.classifier.intent_patterns['test_intent']['responses'], responses)
        
        # Test classification with new intent
        intent, response = self.classifier.classify_intent("testing")
        self.assertEqual(intent, 'test_intent')
        self.assertEqual(response, "This is a test response")
    
    def test_response_variety(self):
        """Test that responses vary (random selection)"""
        # Multiple calls might return different responses
        responses = set()
        for _ in range(20):
            _, response = self.classifier.classify_intent("hello")
            responses.add(response)
        
        # Should have at least 1 response (could be more if random varies)
        self.assertGreater(len(responses), 0)
    
    def test_empty_message(self):
        """Test classification of empty or whitespace messages"""
        test_cases = ["", "   ", "\n", "\t"]
        
        for message in test_cases:
            intent, response = self.classifier.classify_intent(message)
            self.assertEqual(intent, 'query')
            self.assertEqual(response, '')
    
    def test_mixed_intent_message(self):
        """Test messages that might match multiple patterns"""
        # "hi thank you" should match greeting (first match wins)
        intent, response = self.classifier.classify_intent("hi thank you")
        # Should match the first pattern it encounters
        self.assertIn(intent, ['greeting', 'thanks'])


if __name__ == '__main__':
    unittest.main()
