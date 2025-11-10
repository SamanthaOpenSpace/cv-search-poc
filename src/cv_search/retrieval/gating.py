from __future__ import annotations

from typing import Any, Dict, List, Tuple

from cv_search.db.database import CVDatabase


def _allowed_seniorities(seat_seniority: str) -> Tuple[str, ...]:
    ladder = ("junior", "mid", "senior", "staff", "principal")
    if seat_seniority not in ladder:
        return ("senior",)
    idx = ladder.index(seat_seniority)
    return ladder[idx:]


class GatingFilter:
    """Strict candidate filtering based on role and seniority."""

    def __init__(self, db: CVDatabase):
        self.db = db

    def filter_candidates(self, seat: Dict[str, Any]) -> Tuple[List[str], str, List[Dict[str, Any]]]:
        role = seat["role"]
        allowed_seniorities = tuple(_allowed_seniorities(seat["seniority"]))
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
          AND t.tag_key  IN ({",".join(["?"] * len(allowed_seniorities))})
      )
    )
    SELECT candidate_id FROM gated
    """
        params = [role] + list(allowed_seniorities)
        plan = self.db.explain_query_plan(sql, params)
        rows = self.db.conn.execute(sql, params).fetchall()
        return [r["candidate_id"] for r in rows], sql.strip(), plan
