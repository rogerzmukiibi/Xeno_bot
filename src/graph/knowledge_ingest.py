"""
knowledge_ingest.py
-------------------
Cleans and ingests the JSON knowledge base into Neo4j.
Handles ID / id inconsistencies safely.
"""

import json
from pathlib import Path
from typing import Dict, Optional

from src.neo4j_client import Neo4jClient
from src.config import KNOWLEDGE_BASE_PATH


class KnowledgeGraphIngestor:
    def __init__(self):
        self.client = Neo4jClient()

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _extract_id(self, item: Dict) -> Optional[str]:
        """Handle both ID and id fields"""
        return item.get("ID") or item.get("id")

    def load_knowledge_base(self):
        kb_path = Path(KNOWLEDGE_BASE_PATH)
        if not kb_path.exists():
            raise FileNotFoundError(f"Knowledge base not found at {kb_path}")

        with open(kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------

    def clear_existing_knowledge(self):
        print("🧹 Clearing existing knowledge graph...")
        self.client.run_query("""
            MATCH (q:Question)
            DETACH DELETE q
        """)
        print("✅ Existing knowledge removed.")

    # --------------------------------------------------
    # Ingestion
    # --------------------------------------------------

    def ingest_question(self, item: Dict):
        question_id = self._extract_id(item)
        if not question_id:
            raise ValueError("Missing ID / id field")

        query = """
        MERGE (q:Question {id: $id})
        SET
            q.text = $question,
            q.content = $content,
            q.source = $source

        MERGE (s:Section {name: $section})
        MERGE (t:Tag {name: $tag})

        MERGE (s)-[:HAS_QUESTION]->(q)
        MERGE (q)-[:TAGGED_AS]->(t)
        """

        self.client.run_query(
            query,
            {
                "id": question_id,
                "question": item.get("Question", "").strip(),
                "content": item.get("Content", "").strip(),
                "source": item.get("Source", "KnowledgeBase"),
                "section": item.get("Section", "General"),
                "tag": item.get("Tag", "General"),
            },
        )

    def ingest_all(self, wipe_existing: bool = True):
        data = self.load_knowledge_base()

        if wipe_existing:
            self.clear_existing_knowledge()

        success, failed = 0, 0

        print(f"📚 Ingesting {len(data)} knowledge items into Neo4j...")

        for item in data:
            try:
                self.ingest_question(item)
                success += 1
            except Exception as e:
                failed += 1
                print(f"❌ Skipped record: {e} | Raw item: {item}")

        print("--------------------------------------------------")
        print(f"✅ Successfully ingested: {success}")
        print(f"⚠️ Failed / skipped: {failed}")
        print("🧠 Knowledge graph ingestion complete.")

    def close(self):
        self.client.close()


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    ingestor = KnowledgeGraphIngestor()
    ingestor.ingest_all(wipe_existing=True)
    ingestor.close()

