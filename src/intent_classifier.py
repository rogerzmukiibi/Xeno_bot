"""
Intent Classification module for XENO Bot
Handles classification of user intents (greetings, thanks, goodbye, queries)
"""
import re
import random
from typing import Tuple, List


class IntentClassifier:
    """Classifies user intents and provides appropriate responses"""
    
    def __init__(self):
        self.intent_patterns = {
            'greeting': {
                'patterns': [
                    r'\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b',
                    r'^(hi|hello|hey)[\s!.]*$',
                    r'\b(how are you|how do you do)\b'
                ],
                'responses': [
                    "Hello! I'm XENO Assistant. How can I help you with XENO financial services today?",
                    "Hi there! I'm here to assist you with any questions about XENO services. What can I help you with?",
                    "Good day! Welcome to XENO Support. How may I assist you today?"
                ]
            },
            'thanks': {
                'patterns': [
                    r'\b(thank you|thanks|thank u|thx|appreciate|grateful)\b',
                    r'^(thanks|thank you)[\s!.]*$',
                    r'\b(much appreciated|thanks a lot|thank you so much)\b'
                ],
                'responses': [
                    "You're welcome! Is there anything else I can help you with regarding XENO services?",
                    "Happy to help! Feel free to ask if you have any other questions about XENO.",
                    "Glad I could assist you! Let me know if you need help with anything else."
                ]
            },
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|farewell|take care|have a good day)\b',
                    r'^(bye|goodbye)[\s!.]*$',
                    r'\b(talk to you later|see you later|until next time)\b'
                ],
                'responses': [
                    "Goodbye! Thank you for using XENO services. Have a great day!",
                    "Take care! Feel free to return anytime you need help with XENO services.",
                    "Have a wonderful day! Don't hesitate to reach out if you need assistance with XENO."
                ]
            },
            'join': {
                'patterns': [
                    r'\b(join|sign up|register|create account|open account|start investing)\b',
                    r'how do i join',
                    r'how to join'
                ],
                'responses': [
                    "To join XENO, you can download the XENO App from your mobile app store (Android or iOS) and select 'Join'. Alternatively, you can sign up at www.myxeno.com. If you don't have a smartphone, you can dial *165*5*7# on MTN."
                ]
            }
        }
    
    def classify_intent(self, message: str, timer=None) -> Tuple[str, str]:
        """
        Classify the intent of a user message
        
        Args:
            message: User's message
            timer: Optional timer object for tracking
        
        Returns:
            Tuple of (intent_name, response_text)
        """
        if timer:
            with timer.time_step("intent_classification"):
                return self._classify_intent_impl(message)
        else:
            return self._classify_intent_impl(message)
    
    def _classify_intent_impl(self, message: str) -> Tuple[str, str]:
        """Internal implementation of intent classification"""
        message_lower = message.lower().strip()
        
        # 1. Check for specific intents first (Join, Thanks, Goodbye)
        # These are usually standalone or specific enough to override 'query'
        for intent_name in ['join', 'thanks', 'goodbye']:
            if intent_name in self.intent_patterns:
                intent_data = self.intent_patterns[intent_name]
                for pattern in intent_data['patterns']:
                    if re.search(pattern, message_lower, re.IGNORECASE):
                        response = random.choice(intent_data['responses'])
                        return intent_name, response

        # 2. Check for Greeting
        # Special logic: Only classify as 'greeting' if it DOES NOT look like a question.
        if 'greeting' in self.intent_patterns:
            intent_data = self.intent_patterns['greeting']
            for pattern in intent_data['patterns']:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    # Heuristic 1: Check for question indicators
                    query_indicators = [
                        r'\bhow\b', r'\bwhat\b', r'\bwhy\b', r'\bwhen\b', r'\bwhere\b', 
                        r'\bcan\b', r'\bcould\b', r'\bwould\b', r'\bhelp\b', 
                        r'\binvest\b', r'\bwithdraw\b', r'\bdeposit\b', r'\bfees\b', 
                        r'\brates\b', r'\binterest\b', r'\bbalance\b', r'\baccount\b'
                    ]
                    
                    # If any query indicator is present, treat as query
                    for indicator in query_indicators:
                        if re.search(indicator, message_lower, re.IGNORECASE):
                            # Exception: "How are you" is a greeting, not a query
                            if "how are you" in message_lower or "how do you do" in message_lower:
                                continue
                            return 'query', ''

                    # Heuristic 2: Length check (fallback)
                    # If it's still very long (> 10 words) but no keywords found, still treat as query to be safe
                    word_count = len(message_lower.split())
                    if word_count > 10:
                        return 'query', ''
                    
                    response = random.choice(intent_data['responses'])
                    return 'greeting', response
        
        return 'query', ''
    
    def is_simple_intent(self, intent: str) -> bool:
        """
        Check if the intent is a simple one that doesn't require RAG
        
        Args:
            intent: Intent name
        
        Returns:
            True if simple intent, False otherwise
        """
        simple_intents = ['greeting', 'thanks', 'goodbye', 'join']
        return intent in simple_intents
    
    def add_intent(self, intent_name: str, patterns: List[str], responses: List[str]):
        """
        Add a new intent to the classifier
        
        Args:
            intent_name: Name of the intent
            patterns: List of regex patterns to match
            responses: List of possible responses
        """
        self.intent_patterns[intent_name] = {
            'patterns': patterns,
            'responses': responses
        }
