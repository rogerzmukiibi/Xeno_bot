"""
Vector Store module for XENO Bot
Handles ChromaDB vector store operations
"""
import chromadb
import numpy as np
import torch
from langchain_chroma import Chroma
from sentence_transformers import util
from typing import List, Tuple, Any
from google import genai
from src.config import (
    client,
    COLLECTION_NAME, 
    CHROMA_DB_PATH, 
    RAG_TOP_K, 
    RAG_MAX_RESULTS,
    EMBEDDING_MODEL
)
from src.knowledge_base import get_knowledge_base_data


def initialize_vector_store() -> Tuple[chromadb.Collection, Chroma, Any]:
    """
    Initialize ChromaDB vector store
    
    Returns:
        Tuple of (collection, vector_store, retriever)
    """
    # Get knowledge base data
    documents, metadatas, ids = get_knowledge_base_data()
    
    # Initialize ChromaDB client
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Try to get existing collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            print(f"Loaded existing ChromaDB collection: {COLLECTION_NAME}")
        except:
            # Create new collection if it doesn't exist
            print(f"Creating new ChromaDB collection: {COLLECTION_NAME}")
            collection = client.create_collection(name=COLLECTION_NAME)
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        
        # Create vector store and retriever
        vector_store = Chroma(client=client, collection_name=COLLECTION_NAME)
        retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": RAG_TOP_K}
        )
        
        return collection, vector_store, retriever
        
    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        raise


def generate_embeddings(query: str, documents: List[Any], timer=None) -> Tuple[List[float], List[List[float]]]:
    """
    Generate embeddings for query and documents
    
    Args:
        query: User query
        documents: List of retrieved documents
        timer: Optional timer object for tracking
    
    Returns:
        Tuple of (query_embedding, doc_embeddings)
    """
    if timer:
        with timer.time_step("embedding_generation"):
            return _generate_embeddings_impl(query, documents)
    else:
        return _generate_embeddings_impl(query, documents)


def _generate_embeddings_impl(query: str, documents: List[Any]) -> Tuple[List[float], List[List[float]]]:
    """Internal implementation of embedding generation"""
    # Generate query embedding
    query_result = client.models.embed_content(
        model=EMBEDDING_MODEL, 
        contents=query
    )
    query_embedding = query_result.embeddings[0].values
    
    # Generate document embeddings
    doc_embeddings = []
    for doc in documents:
        doc_result = client.models.embed_content(
            model=EMBEDDING_MODEL, 
            contents=doc.page_content
        )
        doc_embeddings.append(doc_result.embeddings[0].values)
    
    return query_embedding, doc_embeddings


def calculate_similarity(query_embedding: List[float], doc_embeddings: List[List[float]], timer=None) -> List[float]:
    """
    Calculate cosine similarity between query and documents
    
    Args:
        query_embedding: Query embedding vector
        doc_embeddings: List of document embedding vectors
        timer: Optional timer object for tracking
    
    Returns:
        List of cosine similarity scores
    """
    if timer:
        with timer.time_step("similarity_calculation"):
            return _calculate_similarity_impl(query_embedding, doc_embeddings)
    else:
        return _calculate_similarity_impl(query_embedding, doc_embeddings)


def _calculate_similarity_impl(query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
    """Internal implementation of similarity calculation"""
    cosine_scores = util.cos_sim(
        torch.tensor(query_embedding).float(), 
        torch.tensor(doc_embeddings).float()
    )[0].tolist()
    
    return cosine_scores


def process_context(results: List[Any], cosine_scores: List[float], 
                    max_results: int = RAG_MAX_RESULTS, timer=None) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """
    Process retrieved context and format for LLM
    
    Args:
        results: List of retrieved documents
        cosine_scores: List of similarity scores
        max_results: Maximum number of results to include
        timer: Optional timer object for tracking
    
    Returns:
        Tuple of (formatted_context, source_ids, knowledge_pairs)
    """
    if timer:
        with timer.time_step("context_processing"):
            return _process_context_impl(results, cosine_scores, max_results)
    else:
        return _process_context_impl(results, cosine_scores, max_results)


def _process_context_impl(results: List[Any], cosine_scores: List[float], 
                          max_results: int) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Internal implementation of context processing"""
    sorted_indices = np.argsort(cosine_scores)[::-1][:max_results]
    
    formatted_context = ""
    source_ids = []
    knowledge_pairs = []
    
    for i, idx in enumerate(sorted_indices, 1):
        result = results[idx]
        score = cosine_scores[idx]
        
        question = result.metadata.get('question', 'N/A')
        answer = result.metadata.get('content', 'N/A')
        
        formatted_context += f"Knowledge Entry {i}:\n"
        formatted_context += f"Q: {question}\n"
        formatted_context += f"A: {answer}\n"
        formatted_context += "-" * 40 + "\n"
        
        source_ids.append(result.metadata.get('id', 'N/A'))
        knowledge_pairs.append((question, answer))
    
    return formatted_context, source_ids, knowledge_pairs
