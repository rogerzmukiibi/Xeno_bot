"""
Cognitive Memory Module for XENO Bot
----------------------------------
Responsible for:
- Creating User nodes
- Creating Memory nodes per interaction
- Linking Memory to Knowledge (Question nodes)
- Maintaining session continuity

This module is called AFTER RAG retrieval and BEFORE logging.
"""

from datetime import datetime
from typing import List, Dict
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class CognitiveMemory:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # -----------------------------
    # USER MANAGEMENT
    # -----------------------------
    def upsert_user(self, session_id: str):
        query = """
        MERGE (u:User {session_id: $session_id})
        ON CREATE SET u.created_at = datetime()
        RETURN u
        """
        with self.driver.session() as session:
            session.run(query, session_id=session_id)

    # -----------------------------
    # MEMORY CREATION
    # -----------------------------
    def create_memory(
        self,
        session_id: str,
        user_question: str,
        bot_answer: str,
        source_ids: List[str],
        intent: str,
        confidence: float
    ):
        query = """
        MATCH (u:User {session_id: $session_id})
        CREATE (m:Memory {
            question: $question,
            answer: $answer,
            intent: $intent,
            confidence: $confidence,
            source_ids: $source_ids,
            timestamp: datetime()
        })
        MERGE (u)-[:HAS_MEMORY]->(m)
        RETURN m
        """
        with self.driver.session() as session:
            session.run(
                query,
                session_id=session_id,
                question=user_question,
                answer=bot_answer,
                intent=intent,
                confidence=confidence,
                source_ids=source_ids
            )

    # -----------------------------
    # MEMORY ↔ KNOWLEDGE LINKING
    # -----------------------------
    def link_memory_to_knowledge(self, session_id: str, source_ids: List[str]):
        query = """
        MATCH (u:User {session_id: $session_id})-[:HAS_MEMORY]->(m:Memory)
        WHERE m.timestamp = (
            MATCH (u)-[:HAS_MEMORY]->(mx:Memory)
            RETURN max(mx.timestamp)
        )
        MATCH (q:Question)
        WHERE q.id IN $source_ids
        MERGE (m)-[:REFERENCES]->(q)
        """
        with self.driver.session() as session:
            session.run(query, session_id=session_id, source_ids=source_ids)

    # -----------------------------
    # FULL MEMORY WRITE (PIPELINE)
    # -----------------------------
    def write_memory(
        self,
        session_id: str,
        question: str,
        answer: str,
        source_ids: List[str],
        intent: str,
        confidence: float
    ):
        """One-call cognitive memory write"""
        self.upsert_user(session_id)
        self.create_memory(
            session_id=session_id,
            user_question=question,
            bot_answer=answer,
            source_ids=source_ids,
            intent=intent,
            confidence=confidence
        )
        self.link_memory_to_knowledge(session_id, source_ids)
