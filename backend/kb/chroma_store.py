"""ChromaDB wrapper using local sentence-transformers embeddings."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = os.getenv("CHROMA_PATH", str(Path(__file__).parents[2] / "chroma_db"))
COLLECTION_NAME = "ml_failure_modes"
EMBED_MODEL = "all-MiniLM-L6-v2"


class ChromaStore:
    def __init__(self, path: str = CHROMA_PATH) -> None:
        self._client = chromadb.PersistentClient(path=path)
        self._ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add a list of KB documents.

        Each dict must contain: symptom, diagnosis, fix, code_snippet,
        citation, domain.  The embedding text is symptom + diagnosis.
        """
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"kb_{i:03d}")
            embed_text = f"{doc['symptom']} {doc['diagnosis']}"
            ids.append(doc_id)
            texts.append(embed_text)
            metadatas.append(
                {
                    "symptom": doc["symptom"],
                    "diagnosis": doc["diagnosis"],
                    "fix": doc["fix"],
                    "code_snippet": doc.get("code_snippet", ""),
                    "citation": doc["citation"],
                    "domain": doc["domain"],
                }
            )

        self._collection.add(ids=ids, documents=texts, metadatas=metadatas)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Return top-n matching documents for the query text."""
        results = self._collection.query(
            query_texts=[text],
            n_results=min(n_results, self.count()),
            include=["metadatas", "distances", "documents"],
        )

        docs: list[dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return docs

        for j, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][j]
            docs.append(
                {
                    "doc_id": doc_id,
                    "symptom": meta.get("symptom", ""),
                    "diagnosis": meta.get("diagnosis", ""),
                    "fix": meta.get("fix", ""),
                    "code_snippet": meta.get("code_snippet", ""),
                    "citation": meta.get("citation", ""),
                    "domain": meta.get("domain", ""),
                    "distance": results["distances"][0][j],
                }
            )
        return docs

    def count(self) -> int:
        return self._collection.count()

    def existing_ids(self) -> set[str]:
        """Return the set of all document IDs currently in the collection."""
        result = self._collection.get(include=[])
        return set(result["ids"])

    def reset(self) -> None:
        """Delete and recreate the collection (useful for re-seeding)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )


# Module-level singleton — imported by retriever and health endpoint
_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
