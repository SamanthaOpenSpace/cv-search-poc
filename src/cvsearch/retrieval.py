from __future__ import annotations
import sqlite3
import json
import os
from typing import Any, Dict, List, Tuple, Optional
import faiss
import numpy as np

from src.cvsearch.storage import CVDatabase
from src.cvsearch.local_embedder import LocalEmbedder
from src.cvsearch.settings import Settings
from src.cvsearch.utils import cosine

def _allowed_seniorities(seat_seniority: str) -> Tuple[str, ...]:
    """
    Helper to get all seniorities >= the requested one.
    (Moved from storage.py to be local to its only user)
    """
    ladder = ("junior", "mid", "senior", "staff", "principal")
    if seat_seniority not in ladder:
        return ("senior",)
    idx = ladder.index(seat_seniority)
    return ladder[idx:]

class GatingFilter:
    """
    Handles the initial, strict filtering of candidates based on
    non-negotiable criteria like 'role' and 'seniority'.
    """
    def __init__(self, db: CVDatabase):
        self.db = db

    def filter_candidates(self, seat: Dict[str, Any]) -> Tuple[List[str], str, List[Dict[str, Any]]]:
        """
        Runs the SQL query to get candidates matching the seat.
        (Logic moved from search.py/_gate_candidate_ids)
        """
        role = seat["role"]
        allowed_sens = tuple(_allowed_seniorities(seat["seniority"]))
        sql = f"""
    WITH gated AS (
      SELECT c.candidate_id
      FROM candidate c
      WHERE EXISTS (
        SELECT 1
        FROM candidate_tag t
        WHERE t.candidate_id = c.candidate_id
          AND t.tag_type = 'role'
          AND t.tag_key  = ?
      )
      AND EXISTS (
        SELECT 1
        FROM candidate_tag t
        WHERE t.candidate_id = c.candidate_id
          AND t.tag_type = 'seniority'
          AND t.tag_key  IN ({",".join(["?"]*len(allowed_sens))})
      )
    )
    SELECT candidate_id FROM gated
    """
        params = [role] + list(allowed_sens)
        plan = self.db.explain_query_plan(sql, params)
        rows = self.db.conn.execute(sql, params).fetchall()
        return [r["candidate_id"] for r in rows], sql.strip(), plan

class LexicalRetriever:
    """
    Handles the weighted-set SQL-based ranking for candidates.
    """
    def __init__(self, db: CVDatabase):
        self.db = db

    def search(self,
               gated_ids: List[str],
               seat: Dict[str, Any],
               top_k: int) -> Tuple[List[sqlite3.Row], str, List[Dict[str, Any]]]:
        """
        Runs the weighted-set ranking query.
        (Logic moved from search.py/_rank_weighted_set)
        """
        must_have = seat.get("must_have", [])
        nice_to_have = seat.get("nice_to_have", [])
        domains = seat.get("domains", [])

        idf_must = self.db.compute_idf(must_have, "tech")
        idf_nice = self.db.compute_idf(nice_to_have, "tech")

        ranked_rows, ranking_sql, ranking_plan = self.db.rank_weighted_set(
            gated_ids=gated_ids,
            must_have=must_have,
            nice_to_have=nice_to_have,
            domains=domains,
            idf_must=idf_must,
            idf_nice=idf_nice,
            top_k=top_k,
        )
        return ranked_rows, ranking_sql, ranking_plan

class LocalSemanticRetriever:
    """
    Handles vector-based semantic search using a local FAISS index
    and an ID map stored in SQLite.
    """
    def __init__(self, db: CVDatabase, settings: Settings):
        self.db = db
        self.settings = settings
        self.local_embedder = LocalEmbedder()
        self._load_db()

    def _load_db(self):
        """
        Loads the FAISS index into memory.
        (The document map is now in SQLite, so we don't load it here)
        """
        index_path = str(self.settings.faiss_index_path)

        if not os.path.exists(index_path):
            print(f"Warning: FAISS index '{index_path}' not found.")
            print("Please run the ingestion pipeline to create the index.")
            self.vector_db = None
            return

        try:
            self.vector_db = faiss.read_index(index_path)
            print(f"Loaded FAISS index with {self.vector_db.ntotal} vectors.")

        except Exception as e:
            print(f"Error loading local FAISS DB: {e}")
            self.vector_db = None

    def _build_vs_query(self, seat: Dict[str, Any]) -> str:
        """(Unchanged from old SemanticRetriever)"""
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

    def search(self,
               gated_ids: List[str],
               seat: Dict[str, Any],
               top_k: int) -> Dict[str, Any]:
        """
        Runs the local vector store search.
        """
        vs_query = self._build_vs_query(seat)

        if self.vector_db is None:
            print("Warning: No FAISS index loaded. Skipping semantic search.")
            return {"query": vs_query, "hits": [], "source": "local_faiss_no_index"}

        try:
            query_vec = self.local_embedder.get_embeddings([vs_query])[0]
            query_array = np.array([query_vec]).astype('float32')
            faiss.normalize_L2(query_array)

            search_k = max(top_k * 5, 100)
            if search_k > self.vector_db.ntotal:
                search_k = self.vector_db.ntotal

            distances, indices = self.vector_db.search(query_array, k=search_k)

            faiss_ids_to_query = [int(i) for i in indices[0] if i >= 0]
            if not faiss_ids_to_query:
                return {"query": vs_query, "hits": [], "source": "local_faiss_no_hits"}

            id_map = self.db.get_candidate_ids_from_faiss_ids(faiss_ids_to_query)

            gated_set = set(gated_ids)
            hits: List[Dict[str, Any]] = []

            for i in range(len(indices[0])):
                faiss_id = int(indices[0][i])

                candidate_id = id_map.get(faiss_id)

                if not candidate_id:
                    continue

                if candidate_id in gated_set:
                    hits.append({
                        "rank": len(hits) + 1,
                        "candidate_id": candidate_id,
                        "file_id": f"faiss_index_{faiss_id}",
                        "score": float(distances[0][i]),
                        "reason": "local_faiss_search"
                    })

                if len(hits) >= top_k:
                    break

            return {
                "query": vs_query,
                "hits": hits,
                "source": "local_faiss_search"
            }

        except Exception as e:
            print(f"Error during local FAISS search: {e}")
            return {"query": vs_query, "hits": [], "source": "local_faiss_error"}