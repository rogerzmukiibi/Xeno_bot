"""
Neo4j Client for XENO Cognitive System
Handles connection and basic query execution
"""

from neo4j import GraphDatabase
from typing import Optional, Dict, Any
from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD
)


class Neo4jClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None):
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
