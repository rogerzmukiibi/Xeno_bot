# Human-in-the-Loop (HITL) Learning & Correction Strategies

**Date:** January 3, 2026
**Status:** Deferred to Future Sprint
**Context:** Discussion on enabling the XENO bot to learn from incorrect answers flagged by users or admins.

## Overview
The goal is to move from a **Passive Feedback Loop** (logging errors to a sheet) to an **Active Learning System** where the bot improves based on corrections.

---

## Strategy Options

### Option 1: The "Correction Memory" (Real-time Agentic)
**Mechanism:**
1.  **Trigger:** User provides a correction (e.g., via UI or chat).
2.  **Action:** Agent calls a `save_correction` tool.
3.  **Storage:** Correction is saved to a specialized `Corrections` vector store or JSON file.
4.  **Retrieval:** During future queries, the Agent checks this store *first*. If a correction exists for the topic, it overrides the base knowledge.

*   **Pros:** Instant improvement; "Magic" user experience.
*   **Cons:** High risk of "Data Poisoning" (users teaching the bot wrong things). Requires authentication/admin checks.

### Option 2: The "Reflexion" Pattern (Conversational)
**Mechanism:**
1.  **Trigger:** User says "That's wrong" in chat.
2.  **Action:** Agent enters a "Reflexion" state, apologizes, and asks for the correct info.
3.  **Storage:** Agent summarizes the rule and saves it to a `learned_rules.json` file.
4.  **Usage:** This file is injected into the System Prompt context.

*   **Pros:** Natural conversation flow.
*   **Cons:** Consumes context window tokens; harder to manage at scale.

### Option 3: Automated Pipeline (Enterprise Standard) - **RECOMMENDED**
**Mechanism:**
1.  **Trigger:** User flags an answer (Thumbs Down + Reason).
2.  **Storage:** Flag is logged to a "Review Queue" (e.g., Google Sheet).
3.  **Review:** Admin reviews flags weekly and approves valid corrections.
4.  **Build:** A script automatically updates the `XENO_Uganda_KnowledgeBase.json` and rebuilds the Vector Store.

*   **Pros:** Safe; Human verification ensures accuracy; Compliance friendly.
*   **Cons:** Not real-time (updates happen on the next build cycle).

---

## Implementation Notes for Future Sprint
- **Prerequisite:** Authentication system (to distinguish Admins from Users if using Option 1).
- **Decision:** For XENO (Financial Services), **Option 3** is safest to prevent misinformation. **Option 1** could be enabled *only* for logged-in Admins.
