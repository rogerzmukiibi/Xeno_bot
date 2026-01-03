
import sys
import os
import time
import logging
from langchain_core.messages import HumanMessage

# Add root to path
sys.path.append(os.getcwd())

from src.intent_classifier import IntentClassifier
from src.agent_graph import agent_app

# Configure logging to see our structured logs in console
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_intent_classifier():
    print("\n=== Testing Intent Classifier Logic ===")
    classifier = IntentClassifier()
    
    test_cases = [
        ("Hello", "greeting"),
        ("Hi there", "greeting"),
        ("Hello, how are you?", "greeting"),
        ("Hello, how do I invest?", "query"),
        ("Hi, can you tell me about fees?", "query"),
        ("Good morning, I have a problem with my account", "query"),
        ("What is XENO?", "query"),
        ("Bye", "goodbye")
    ]
    
    for text, expected in test_cases:
        intent, _ = classifier.classify_intent(text)
        result = "✅ PASS" if intent == expected else f"❌ FAIL (Got {intent})"
        print(f"Input: '{text}' -> Expected: {expected} | {result}")

def test_agent_flow():
    print("\n=== Testing Full Agent Flow (with Logging) ===")
    
    # Test a simple query
    question = "What is the minimum investment amount?"
    print(f"Invoking agent with: '{question}'")
    
    inputs = {
        "session_id": "test_session_RAG_1",
        "question": question,
        "messages": [HumanMessage(content=question)],
        "documents": [],
        "context_metadata": [],
        "generation": "",
        "intent": "",
        "steps": [],
        "retry_count": 0
    }
    
    try:
        result = agent_app.invoke(inputs)
        print("\nAgent Execution Result:")
        print(f"Intent: {result['intent']}")
        print(f"Steps Taken: {result['steps']}")
        print(f"Response Length: {len(result['generation'])}")
        print(f"Response Preview: {result['generation'][:100]}...")
    except Exception as e:
        print(f"Agent failed: {e}")

if __name__ == "__main__":
    test_intent_classifier()
    test_agent_flow()
