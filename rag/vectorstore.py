"""
rag/vectorstore.py — ChromaDB vector store for medical knowledge.
Persists locally — no server needed, fully free.
"""
import logging
from typing import List, Dict, Any
from typing import Optional
import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_PERSIST_DIR, COLLECTION_NAME, TOP_K_RESULTS
from rag.embedder import embed_texts, embed_query

logger = logging.getLogger(__name__)

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    """
    Embed and store documents in ChromaDB.
    Called once during ingestion (ingest.py).
    """
    collection = get_collection()
    embeddings = embed_texts(texts)
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info(f"Upserted {len(texts)} documents into ChromaDB.")


def similarity_search(
    query: str,
    n_results: int = TOP_K_RESULTS,
    filter_meta: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most relevant chunks for a query.
    Returns list of dicts with 'text', 'metadata', and 'distance'.
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count() or 1),
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_meta:
        kwargs["where"] = filter_meta

    results = collection.query(**kwargs)

    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({"text": doc, "metadata": meta, "distance": dist})
    return docs


def count_documents() -> int:
    return get_collection().count()