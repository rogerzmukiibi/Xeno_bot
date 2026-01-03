# Intent Classifier in Agentic RAG Architecture

**Date:** January 3, 2026
**Context:** Overview of how Intent Classification is integrated into the new LangGraph-based Agentic RAG system.

## 1. Architectural Role: "The Traffic Controller"
In our new Agentic approach, the Intent Classifier serves as the **Entry Point** (Node 0) of the decision graph. Its primary role is **Efficiency Routing**:
- **Fast Path:** Instantly handles chit-chat (Greetings, Thanks, Goodbyes) without invoking the LLM or Vector Store.
- **Slow Path:** Identifies complex queries and routes them to the RAG pipeline (Retrieve -> Grade -> Generate).

## 2. Implementation Details

### A. The Core Logic (`src/intent_classifier.py`)
We utilize a **Regex-Based Deterministic Classifier**. This was chosen over an LLM-based classifier for speed and zero cost.

*   **Mechanism:** Iterates through a dictionary of regex patterns.
*   **Intents Supported:**
    *   `greeting`: "hi", "hello", "good morning"
    *   `thanks`: "thank you", "thx"
    *   `goodbye`: "bye", "see you"
    *   `join`: "sign up", "how to join" (Specific fast-path for conversion)
*   **Fallback:** If no pattern matches, it returns `query`.

### B. Integration in the Graph (`src/agent_graph.py`)

The classifier is wrapped in the `classify_input` node:

```python
def classify_input(state: AgentState):
    """
    Node 1: The Gatekeeper
    """
    question = state["question"]
    
    # 1. Run Regex Classification
    intent, response = intent_classifier.classify_intent(question)
    
    # 2. Log the decision (Structured Logging)
    log_event("classify_input", {
        "question": question,
        "intent": intent,
        "is_direct_response": intent != "query"
    })
    
    # 3. Update State
    return {"intent": intent, "generation": response, "steps": ["classify_input"]}
```

### C. The Routing Logic (Conditional Edges)

We use a conditional edge `route_intent` to determine the next step in the graph:

```python
def route_intent(state: AgentState):
    intent = state["intent"]
    
    if intent == "query":
        return "retrieve"  # -> Proceed to Vector Search (RAG)
    else:
        return "end"       # -> Stop and return the pre-canned response
```

## 3. Execution Flow Examples

### Scenario A: User says "Hello"
1.  **Node:** `classify_input` runs.
2.  **Logic:** Regex matches `greeting`.
3.  **State Update:** `intent="greeting"`, `generation="Hello! I'm XENO..."`.
4.  **Router:** Sees `greeting` != `query`.
5.  **Action:** Routes to `END`.
6.  **Result:** Instant response (< 100ms).

### Scenario B: User says "What are the interest rates?"
1.  **Node:** `classify_input` runs.
2.  **Logic:** Regex finds no match.
3.  **State Update:** `intent="query"`, `generation=""`.
4.  **Router:** Sees `intent == "query"`.
5.  **Action:** Routes to `retrieve` node.
6.  **Result:** Agent begins RAG pipeline (Retrieve -> Grade -> Generate).

## 4. Benefits for the Developer
*   **Deterministic:** You know exactly why a greeting triggered. No "LLM probability" guessing.
*   **Low Latency:** The user gets immediate feedback for simple interactions.
*   **Cost Saving:** We don't waste Gemini tokens on "You're welcome".
*   **Extensible:** To add a new fast-path (e.g., "Reset Password"), simply add a regex pattern to `src/intent_classifier.py`. The Agent Graph handles the routing automatically.
