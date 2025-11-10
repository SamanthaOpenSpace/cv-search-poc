from __future__ import annotations

import os
from typing import Any, Dict, List

import faiss
import numpy as np

from cv_search.config.settings import Settings
from cv_search.db.database import CVDatabase
from cv_search.retrieval.local_embedder import LocalEmbedder


class LocalSemanticRetriever:
    """Vector-based retrieval backed by a local FAISS index."""

    def __init__(self, db: CVDatabase, settings: Settings):
        self.db = db
        self.settings = settings
        self.local_embedder = LocalEmbedder()
        self.vector_db = self._load_index()

    def _load_index(self):
        index_path = str(self.settings.faiss_index_path)
        if not os.path.exists(index_path):
            print(f"Warning: FAISS index '{index_path}' not found.")
            print("Please run the ingestion pipeline to create the index.")
            return None
        try:
            return faiss.read_index(index_path)
        except Exception as exc:  # pragma: no cover - protective logging
            print(f"Error loading local FAISS DB: {exc}")
            return None

    def _build_vs_query(self, seat: Dict[str, Any]) -> str:
        role = seat["role"].replace("_", " ")
        seniority = seat["seniority"]
        domains = ", ".join(seat["domains"]) if seat["domains"] else "any domain"
        musts = ", ".join(seat["must_have"]) if seat["must_have"] else "(no hard musts)"
        nice = ", ".join(seat["nice_to_have"]) if seat["nice_to_have"] else "(optional)"
        return (
            f"{seniority}+ {role} in {domains}. "
            f"Must: {musts}. Nice: {nice}. "
            f"Prefer demonstrated outcomes and similar patterns (latency, availability, LCP, etc.)."
        )

    def search(self, gated_ids: List[str], seat: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        vs_query = self._build_vs_query(seat)
        if self.vector_db is None:
            print("Warning: No FAISS index loaded. Skipping semantic search.")
            return {"query": vs_query, "hits": [], "source": "local_faiss_no_index"}

        try:
            query_vec = self.local_embedder.get_embeddings([vs_query])[0]
            query_array = np.array([query_vec]).astype("float32")
            faiss.normalize_L2(query_array)

            search_k = min(max(top_k * 5, 100), self.vector_db.ntotal)
            distances, indices = self.vector_db.search(query_array, k=search_k)

            faiss_ids_to_query = [int(i) for i in indices[0] if i >= 0]
            if not faiss_ids_to_query:
                return {"query": vs_query, "hits": [], "source": "local_faiss_no_hits"}

            id_map = self.db.get_candidate_ids_from_faiss_ids(faiss_ids_to_query)
            gated_set = set(gated_ids)

            hits: List[Dict[str, Any]] = []
            for pos, faiss_id in enumerate(indices[0]):
                candidate_id = id_map.get(int(faiss_id))
                if not candidate_id or candidate_id not in gated_set:
                    continue
                hits.append(
                    {
                        "rank": len(hits) + 1,
                        "candidate_id": candidate_id,
                        "file_id": f"faiss_index_{int(faiss_id)}",
                        "score": float(distances[0][pos]),
                        "reason": "local_faiss_search",
                    }
                )
                if len(hits) >= top_k:
                    break

            return {"query": vs_query, "hits": hits, "source": "local_faiss_search"}
        except Exception as exc:  # pragma: no cover - protective logging
            print(f"Error during local FAISS search: {exc}")
            return {"query": vs_query, "hits": [], "source": "local_faiss_error"}
