"""
Intent Classification module for XENO Bot
Handles classification of user intents (greetings, thanks, goodbye, queries)
"""

import random
import re
from typing import List, Tuple


class IntentClassifier:
    """Classifies user intents and provides appropriate responses"""

    def __init__(self):
        self.intent_patterns = {
            "greeting": {
                "patterns": [
                    r"\b(hi|hello|hey|good morning|good afternoon|good evening|greetings)\b",
                    r"^(hi|hello|hey)[\s!.]*$",
                    r"\b(how are you|how do you do)\b",
                ],
                "responses": [
                    "Hello! I'm XENO Assistant. How can I help you with XENO financial services today?",
                    "Hi there! I'm here to assist you with any questions about XENO services. What can I help you with?",
                    "Good day! Welcome to XENO Support. How may I assist you today?",
                ],
            },
            "thanks": {
                "patterns": [
                    r"\b(thank you|thanks|thank u|thx|appreciate|grateful)\b",
                    r"^(thanks|thank you)[\s!.]*$",
                    r"\b(much appreciated|thanks a lot|thank you so much)\b",
                ],
                "responses": [
                    "You're welcome! Is there anything else I can help you with regarding XENO services?",
                    "Happy to help! Feel free to ask if you have any other questions about XENO.",
                    "Glad I could assist you! Let me know if you need help with anything else.",
                ],
            },
            "goodbye": {
                "patterns": [
                    r"\b(bye|goodbye|see you|farewell|take care|have a good day)\b",
                    r"^(bye|goodbye)[\s!.]*$",
                    r"\b(talk to you later|see you later|until next time)\b",
                ],
                "responses": [
                    "Goodbye! Thank you for using XENO services. Have a great day!",
                    "Take care! Feel free to return anytime you need help with XENO services.",
                    "Have a wonderful day! Don't hesitate to reach out if you need assistance with XENO.",
                ],
            },
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

        for intent_name, intent_data in self.intent_patterns.items():
            for pattern in intent_data["patterns"]:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    response = random.choice(intent_data["responses"])
                    return intent_name, response

        return "query", ""

    def is_simple_intent(self, intent: str) -> bool:
        """
        Check if the intent is a simple one that doesn't require RAG

        Args:
            intent: Intent name

        Returns:
            True if simple intent, False otherwise
        """
        simple_intents = ["greeting", "thanks"]
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
            "patterns": patterns,
            "responses": responses,
        }
