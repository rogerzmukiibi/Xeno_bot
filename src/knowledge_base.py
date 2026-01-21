"""
Knowledge Base module for XENO Bot
Handles loading and preparing knowledge base data
"""
import pandas as pd
from typing import Hashable, List, Dict, Tuple, Any
from src.config import KNOWLEDGE_BASE_PATH


def load_knowledge_base(filepath: str = KNOWLEDGE_BASE_PATH) -> pd.DataFrame:
    """
    Load knowledge base from JSON file
    
    Args:
        filepath: Path to the knowledge base JSON file
    
    Returns:
        DataFrame with knowledge base data
    """
    try:
        df = pd.read_json(filepath)
        df.dropna(subset=['Content'], inplace=True)
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        df = pd.DataFrame()
    return df


def prepare_documents(data: List[Dict[Hashable, Any]]) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Prepare documents for vector store
    
    Args:
        data: List of knowledge base entries
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    documents, metadatas, ids = [], [], []
    
    try:
        for item in data:
            # Create document text with question and answer
            document_text = f"Question: {item['Question']}\nAnswer: {item['Content']}"
            documents.append(document_text)
            
            # Create metadata
            metadata = {
                "question": item["Question"],
                "content": item["Content"],
                "section": item.get("Section", ""),
                "source": item.get("Source", ""),
                "owner": item.get("Owner", ""),
                "tag": item.get("Tag", ""),
                "id": item["ID"]
            }
            metadatas.append(metadata)
            
            # Add ID
            ids.append(item["ID"])
    except KeyError as e:
        print(f"Missing expected key in data item: {e}")

    return documents, metadatas, ids


def get_knowledge_base_data() -> Tuple[List[str], List[Dict], List[str]]:
    """
    Load and prepare knowledge base data
    
    Returns:
        Tuple of (documents, metadatas, ids)
    """
    df = load_knowledge_base()
    data_list = df.to_dict('records')
    return prepare_documents(data_list)
