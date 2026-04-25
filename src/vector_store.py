"""
Vector Store module for XENO Bot
Handles ChromaDB vector store operations
"""

from typing import Any, List, Tuple, cast

import chromadb
import numpy as np
import torch
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer, util

from src.config import (CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
                        RAG_MAX_RESULTS, RAG_TOP_K)
from src.knowledge_base import get_knowledge_base_data


_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the local embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def encode_documents_for_collection(documents: List[str]) -> List[List[float]]:
    """Encode knowledge-base documents for persistent Chroma storage."""
    if not documents:
        return []

    encoded = get_embedding_model().encode(documents)
    if hasattr(encoded, "ndim") and encoded.ndim == 1:
        return [cast(List[float], encoded.tolist())]
    if hasattr(encoded, "tolist"):
        return cast(List[List[float]], encoded.tolist())
    return cast(List[List[float]], [list(item) for item in encoded])


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
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"embedding_model": EMBEDDING_MODEL},
            )
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=encode_documents_for_collection(documents),
                )

        # Create vector store and retriever
        vector_store = Chroma(client=client, collection_name=COLLECTION_NAME)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": RAG_TOP_K}
        )

        return collection, vector_store, retriever

    except Exception as e:
        print(f"Failed to initialize ChromaDB: {e}")
        raise


def generate_embeddings(
    query: str, documents: List[Any], timer=None
) -> Tuple[List[float], List[List[float]]]:
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


def _generate_embeddings_impl(
    query: str, documents: List[Any]
) -> Tuple[List[float], List[List[float]]]:
    """Internal implementation of embedding generation"""
    model = get_embedding_model()

    query_embedding = model.encode(query)
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    query_embedding = cast(List[float], query_embedding)

    doc_contents = [doc.page_content for doc in documents]
    if not doc_contents:
        return query_embedding, []

    doc_matrix = model.encode(doc_contents)

    # Convert model output to list[list[float]] while handling one/many documents.
    if hasattr(doc_matrix, "ndim") and doc_matrix.ndim == 1:
        doc_embeddings = [doc_matrix.tolist()]
    elif hasattr(doc_matrix, "tolist"):
        doc_embeddings = doc_matrix.tolist()
    else:
        doc_embeddings = [list(embedding) for embedding in doc_matrix]

    if doc_embeddings and isinstance(doc_embeddings[0], float):
        doc_embeddings = [doc_embeddings]

    doc_embeddings = cast(List[List[float]], doc_embeddings)

    return query_embedding, doc_embeddings


def calculate_similarity(
    query_embedding: List[float], doc_embeddings: List[List[float]], timer=None
) -> List[float]:
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


def _calculate_similarity_impl(
    query_embedding: List[float], doc_embeddings: List[List[float]]
) -> List[float]:
    """Internal implementation of similarity calculation"""
    cosine_scores = util.cos_sim(
        torch.tensor(query_embedding).float(), torch.tensor(doc_embeddings).float()
    )[0].tolist()

    return cosine_scores


def process_context(
    results: List[Any],
    cosine_scores: List[float],
    max_results: int = RAG_MAX_RESULTS,
    timer=None,
) -> Tuple[str, List[str], List[Tuple[str, str]]]:
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


def _process_context_impl(
    results: List[Any], cosine_scores: List[float], max_results: int
) -> Tuple[str, List[str], List[Tuple[str, str]]]:
    """Internal implementation of context processing"""
    sorted_indices = np.argsort(cosine_scores)[::-1][:max_results]

    formatted_context = ""
    source_ids = []
    knowledge_pairs = []

    for i, idx in enumerate(sorted_indices, 1):
        result = results[idx]
        cosine_scores[idx]

        question = result.metadata.get("question", "N/A")
        answer = result.metadata.get("content", "N/A")

        formatted_context += f"Knowledge Entry {i}:\n"
        formatted_context += f"Q: {question}\n"
        formatted_context += f"A: {answer}\n"
        formatted_context += "-" * 40 + "\n"

        source_ids.append(result.metadata.get("id", "N/A"))
        knowledge_pairs.append((question, answer))

    return formatted_context, source_ids, knowledge_pairs
