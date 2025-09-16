# rag.py
from __future__ import annotations

from typing import List, Union, Optional
from threading import RLock

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


class RAGMemory:
    """
    In-memory context store:
    - Keeps texts & embeddings in RAM.
    - No persistence, fast for a few thousand items.
    - Thread-safe add/search operations.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_items: Optional[int] = 2000,
        normalize: bool = True,
    ):
        """
        Args:
            model_name: HF sentence-transformers model for embeddings.
            max_items: Keep at most this many items (FIFO evict). None = unbounded.
            normalize: If True, normalize vectors -> cosine becomes dot product.
        """
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.max_items = max_items
        self.normalize = normalize

        self._texts: List[str] = []
        self._embeds: Optional[np.ndarray] = None  # shape: (N, D)
        self._lock = RLock()

    # ---------- Public API ----------

    def add_to_memory(self, texts: Union[str, List[str]]) -> None:
        """Add one or many texts into memory and compute embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return

        vecs = self._encode(texts)

        with self._lock:
            if self._embeds is None:
                self._embeds = vecs
            else:
                self._embeds = np.vstack([self._embeds, vecs])
            self._texts.extend(texts)

            # Optional cap: evict oldest if exceeding max_items
            if self.max_items and len(self._texts) > self.max_items:
                excess = len(self._texts) - self.max_items
                self._texts = self._texts[excess:]
                self._embeds = self._embeds[excess:, :]

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Return top-k texts joined as a context block."""
        hits = self.similarity_search(query, k=k)
        return "\n".join(hits)

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """Return top-k matching texts by cosine similarity."""
        with self._lock:
            if not self._texts or self._embeds is None:
                return []

            qv = self._encode([query])[0]  # (D,)
            M = self._embeds  # (N, D)

            if self.normalize:
                # cosine sim == dot product when both are unit-normalized
                scores = M @ qv  # (N,)
            else:
                num = M @ qv
                denom = (np.linalg.norm(M, axis=1) * (np.linalg.norm(qv) + 1e-9))
                scores = num / (denom + 1e-9)

            top_idx = np.argsort(-scores)[: max(1, k)]
            return [self._texts[i] for i in top_idx.tolist()]

    def clear(self) -> None:
        """Remove all in-memory items."""
        with self._lock:
            self._texts.clear()
            self._embeds = None

    # ---------- Internals ----------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to vectors; optionally L2-normalize."""
        vecs = np.asarray(self.embedder.embed_documents(texts), dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            vecs = vecs / norms
        return vecs
