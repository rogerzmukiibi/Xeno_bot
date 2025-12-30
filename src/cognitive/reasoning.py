"""
Cognitive Reasoning Module for XENO Bot

This module decides:
- Whether an answer should be generated
- What evidence is sufficient
- How to respond when knowledge is missing
"""

from typing import Dict, Any


class CognitiveReasoner:
    """
    Applies reasoning rules on retrieved knowledge
    before allowing response generation.
    """

    def __init__(self):
        pass

    def reason(
        self,
        query: str,
        intent: str,
        retrieval_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply reasoning rules and return a decision object.
        """

        # 1️⃣ Intent-only responses
        if intent in {"greeting", "small_talk"}:
            return self._decision(
                action="respond_directly",
                message="Intent-based response",
                confidence=1.0
            )

        # 2️⃣ No retrieval results
        if retrieval_result["status"] == "empty":
            return self._decision(
                action="ask_for_clarification",
                message=retrieval_result["reason"],
                confidence=0.0
            )

        # 3️⃣ Low confidence safeguard
        if retrieval_result["confidence"] < 0.5:
            return self._decision(
                action="decline_answer",
                message="Insufficient confidence in retrieved knowledge.",
                confidence=retrieval_result["confidence"]
            )

        # 4️⃣ Valid knowledge available
        return self._decision(
            action="generate_answer",
            message="Sufficient knowledge retrieved.",
            confidence=retrieval_result["confidence"],
            context=retrieval_result["formatted_context"],
            source_ids=retrieval_result["source_ids"],
            knowledge_pairs=retrieval_result["knowledge_pairs"]
        )

    @staticmethod
    def _decision(
        action: str,
        message: str,
        confidence: float,
        context: str = "",
        source_ids=None,
        knowledge_pairs=None
    ) -> Dict[str, Any]:
        """
        Standardized reasoning output
        """
        return {
            "action": action,
            "message": message,
            "confidence": confidence,
            "context": context,
            "source_ids": source_ids or [],
            "knowledge_pairs": knowledge_pairs or []
        }
