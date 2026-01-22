"""
schema.py
-----------
Defines and initializes the Neo4j graph schema for XENO Bot.

This file is responsible for:
- Creating node constraints
- Creating indexes for fast retrieval
- Defining the core graph structure used by RAG + Memory
"""

from src.neo4j_client import Neo4jClient


class XenoGraphSchema:
    def __init__(self):
        self.client = Neo4jClient()

    def create_constraints(self):
        """Create uniqueness constraints"""
        queries = [
            # Knowledge Base
            """
            CREATE CONSTRAINT kb_question_id IF NOT EXISTS
            FOR (q:Question)
            REQUIRE q.id IS UNIQUE
            """,

            # Sections
            """
            CREATE CONSTRAINT section_name IF NOT EXISTS
            FOR (s:Section)
            REQUIRE s.name IS UNIQUE
            """,

            # Tags
            """
            CREATE CONSTRAINT tag_name IF NOT EXISTS
            FOR (t:Tag)
            REQUIRE t.name IS UNIQUE
            """,

            # Users (session-based)
            """
            CREATE CONSTRAINT user_session IF NOT EXISTS
            FOR (u:User)
            REQUIRE u.session_id IS UNIQUE
            """
        ]

        for q in queries:
            self.client.run_query(q)

    def create_indexes(self):
        """Create indexes for faster search"""
        queries = [
            """
            CREATE INDEX question_text IF NOT EXISTS
            FOR (q:Question)
            ON (q.text)
            """,

            """
            CREATE INDEX question_embedding IF NOT EXISTS
            FOR (q:Question)
            ON (q.embedding)
            """,

            """
            CREATE INDEX memory_timestamp IF NOT EXISTS
            FOR (m:Memory)
            ON (m.timestamp)
            """
        ]

        for q in queries:
            self.client.run_query(q)

    def initialize(self):
        """Initialize full schema"""
        print("🔧 Creating Neo4j schema...")
        self.create_constraints()
        self.create_indexes()
        print("✅ Neo4j schema initialized successfully.")

    def close(self):
        self.client.close()


if __name__ == "__main__":
    schema = XenoGraphSchema()
    schema.initialize()
    schema.close()
