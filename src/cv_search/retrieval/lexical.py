from __future__ import annotations

from typing import Any, Dict, List, Tuple

from cv_search.db.database import CVDatabase


class LexicalRetriever:
    """Weighted-set SQL ranking over structured candidate tags."""

    def __init__(self, db: CVDatabase):
        self.db = db

    def search(
        self,
        gated_ids: List[str],
        seat: Dict[str, Any],
        top_k: int,
    ) -> Tuple[List[Any], str, List[Dict[str, Any]]]:
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
