"""
Cognitive Retrieval Module for XENO Bot

This module is responsible for *deciding and executing*
the correct knowledge retrieval strategy based on intent,
semantic confidence, and system rules.

It does NOT generate answers.
It does NOT call the LLM.
"""

from typing import Dict, Any, List, Optional
from src.vector_store import (
    initialize_vector_store,
    generate_embeddings,
    calculate_similarity,
    process_context
)
from src.config import SIMILARITY_THRESHOLD
from src.neo4j_client import Neo4jClient


class CognitiveRetriever:
    """
    Cognitive Retriever orchestrates controlled access
    to semantic and graph-based knowledge.
    """

    def __init__(self):
        # Initialize vector store components
        self.collection, self.vector_store, self.retriever = initialize_vector_store()

        # Initialize graph client lazily
        self.graph_client = Neo4jClient()

    def retrieve(
        self,
        query: str,
        intent: str,
        timer=None
    ) -> Dict[str, Any]:
        """
        Main retrieval entry point.

        Returns a structured retrieval result
        that downstream reasoning can act upon.
        """

        # 1️⃣ Intent-based short-circuiting
        if intent in {"greeting", "small_talk"}:
            return self._empty_result(
                reason="Intent does not require knowledge retrieval."
            )

        # 2️⃣ Retrieve candidate documents
        results = self.retriever.get_relevant_documents(query)

        if not results:
            return self._empty_result(
                reason="No documents retrieved from vector store."
            )

        # 3️⃣ Generate embeddings & similarity
        query_embedding, doc_embeddings = generate_embeddings(
            query, results, timer=timer
        )

        cosine_scores = calculate_similarity(
            query_embedding, doc_embeddings, timer=timer
        )

        # 4️⃣ Confidence check
        if max(cosine_scores) < SIMILARITY_THRESHOLD:
            return self._empty_result(
                reason="Similarity below confidence threshold."
            )

        # 5️⃣ Process usable context
        formatted_context, source_ids, knowledge_pairs = process_context(
            results, cosine_scores, timer=timer
        )

        # 6️⃣ Optional: fetch structured graph context
        graph_context = self._fetch_graph_context(source_ids)

        return {
            "status": "ok",
            "formatted_context": formatted_context,
            "source_ids": source_ids,
            "knowledge_pairs": knowledge_pairs,
            "graph_context": graph_context,
            "confidence": max(cosine_scores)
        }

    def _fetch_graph_context(self, source_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve structured knowledge from Neo4j
        for reasoning purposes (not generation).
        """
        if not source_ids:
            return []

        query = """
        MATCH (k:Knowledge)
        WHERE k.id IN $ids
        OPTIONAL MATCH (k)-[r]->(n)
        RETURN k.id AS id, k.question AS question, type(r) AS relation, n.name AS related
        """

        try:
            return self.graph_client.run_query(query, {"ids": source_ids})
        except Exception:
            # Graph is optional for retrieval
            return []

    @staticmethod
    def _empty_result(reason: str) -> Dict[str, Any]:
        """
        Standardized empty retrieval response.
        """
        return {
            "status": "empty",
            "reason": reason,
            "formatted_context": "",
            "source_ids": [],
            "knowledge_pairs": [],
            "graph_context": [],
            "confidence": 0.0
        }

    def close(self):
        """Clean shutdown"""
        self.graph_client.close()
