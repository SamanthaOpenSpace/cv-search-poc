from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from cv_search.clients.openai_client import OpenAIClient
from cv_search.config.settings import Settings
from cv_search.db.database import CVDatabase
from cv_search.planner.service import Planner
from cv_search.ranking.hybrid import HybridRanker
from cv_search.retrieval import GatingFilter, LexicalRetriever, LocalSemanticRetriever
from cv_search.search.artifacts import SearchRunArtifactWriter
from cv_search.search.justification import JustificationService


def default_run_dir() -> str:
    return f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"


class SearchProcessor:
    """High-level orchestrator for single-seat and multi-seat searches."""

    def __init__(self, db: CVDatabase, client: OpenAIClient, settings: Settings):
        self.db = db
        self.client = client
        self.settings = settings

        self.gating_filter = GatingFilter(db)
        self.lexical_retriever = LexicalRetriever(db)
        self.semantic_retriever = LocalSemanticRetriever(db, settings)
        self.hybrid_ranker = HybridRanker(db, settings)
        self.planner = Planner()
        self.artifact_writer = SearchRunArtifactWriter()
        self.justification_service = JustificationService(client, settings)

    def _extract_seat(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        seat = criteria["team_size"]["members"][0]
        return {
            "role": seat["role"],
            "seniority": seat["seniority"],
            "domains": seat.get("domains", []),
            "must_have": seat.get("tech_tags", []),
            "nice_to_have": seat.get("nice_to_have", []),
        }

    def _fingerprint(self, seat: Dict[str, Any], mode: str) -> Dict[str, Any]:
        return {
            "role": seat["role"],
            "seniority": seat["seniority"],
            "domains": sorted(seat["domains"]),
            "must_have": sorted(seat["must_have"]),
            "nice_to_have": sorted(seat["nice_to_have"]),
            "mode": mode,
        }

    def _run_single_seat(
        self,
        criteria: Dict[str, Any],
        top_k: int,
        mode_override: str | None,
        vs_topk_override: int | None,
        with_justification: bool,
    ) -> Dict[str, Any]:
        mode = (mode_override or self.settings.search_mode).lower()
        vs_topk = vs_topk_override or self.settings.search_vs_topk

        seat = self._extract_seat(criteria)

        start = time.perf_counter()

        g0 = time.perf_counter()
        gated_ids, gating_sql, gating_plan = self.gating_filter.filter_candidates(seat)
        g1 = time.perf_counter()

        if not gated_ids:
            return {
                "query_fingerprint": self._fingerprint(seat, mode),
                "metrics": {
                    "gate_count": 0,
                    "gate_time_ms": round((g1 - g0) * 1000, 2),
                    "rank_time_ms": 0.0,
                    "total_time_ms": round((g1 - start) * 1000, 2),
                },
                "gating_sql": gating_sql,
                "gating_explain": gating_plan,
                "ranking_sql": None,
                "ranking_explain": [],
                "vs_query": None,
                "vs_results": None,
                "fusion": None,
                "results": [],
                "reason": "strict_gate_empty",
            }

        lex_limit = max(top_k, vs_topk, len(gated_ids))
        r0 = time.perf_counter()
        lex_rows, ranking_sql, ranking_plan = self.lexical_retriever.search(gated_ids, seat, lex_limit)
        r1 = time.perf_counter()
        rank_time_ms = round((r1 - r0) * 1000, 2)

        sem_raw = self.semantic_retriever.search(gated_ids, seat, vs_topk)
        sem_hits = sem_raw.get("hits", [])

        final_results, fusion_dump = self.hybrid_ranker.rank(seat, lex_rows, sem_hits, mode, top_k)

        if with_justification and final_results:
            justifications = self.justification_service.generate(final_results, seat)
            for item in final_results:
                item["llm_justification"] = justifications.get(item["candidate_id"])

        total = time.perf_counter() - start
        return {
            "criteria": criteria,
            "query_fingerprint": self._fingerprint(seat, mode),
            "metrics": {
                "gate_count": len(gated_ids),
                "gate_time_ms": round((g1 - g0) * 1000, 2),
                "rank_time_ms": rank_time_ms,
                "total_time_ms": round(total * 1000, 2),
            },
            "vs_query": sem_raw.get("query"),
            "vs_results": sem_raw,
            "fusion": fusion_dump or None,
            "results": final_results,
            "gating_sql": gating_sql,
            "gating_explain": gating_plan,
            "ranking_sql": ranking_sql,
            "ranking_explain": ranking_plan,
        }

    def search_for_seat(
        self,
        criteria: Dict[str, Any],
        top_k: int = 10,
        run_dir: str | None = None,
        mode_override: str | None = None,
        vs_topk_override: int | None = None,
        with_justification: bool = True,
    ) -> Dict[str, Any]:
        payload = self._run_single_seat(
            criteria=criteria,
            top_k=top_k,
            mode_override=mode_override,
            vs_topk_override=vs_topk_override,
            with_justification=with_justification,
        )
        if run_dir:
            self.artifact_writer.write(run_dir, payload)
        return payload

    def search_for_project(
        self,
        criteria: Any,
        top_k: int = 3,
        run_dir: Optional[str] = None,
        raw_text: Optional[str] = None,
        with_justification: bool = True,
    ) -> Dict[str, Any]:
        crit_with_seats = self.planner.derive_project_seats(criteria, raw_text=raw_text)
        base_dict = self.planner._criteria_dict(crit_with_seats)

        out_dir = run_dir or default_run_dir()
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        seats = base_dict["team_size"]["members"]
        aggregated: List[Dict[str, Any]] = []
        gaps: List[int] = []

        for idx, seat in enumerate(seats, start=1):
            seat_dir = Path(out_dir) / f"seat_{idx:02d}_{seat['role']}"
            single_criteria = self.planner._pack_single_seat_criteria(base_dict, seat)
            payload = self.search_for_seat(
                criteria=single_criteria,
                top_k=top_k,
                run_dir=str(seat_dir),
                mode_override=None,
                vs_topk_override=None,
                with_justification=with_justification,
            )
            aggregated.append(
                {
                    "index": idx,
                    "role": seat["role"],
                    "criteria": single_criteria,
                    "metrics": payload.get("metrics", {}),
                    "results": payload.get("results", []),
                }
            )
            if not payload.get("results"):
                gaps.append(idx)

        return {
            "project_criteria": base_dict,
            "seats": aggregated,
            "gaps": gaps,
            "run_dir": out_dir,
            "note": "Stateless multi-seat search; per-seat artifacts live under run_dir.",
        }
