from __future__ import annotations
import os, json, time, sqlite3
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.cvsearch.api_client import OpenAIClient
from src.cvsearch.settings import Settings
from src.cvsearch.storage import CVDatabase
from src.cvsearch.retrieval import GatingFilter, LexicalRetriever, LocalSemanticRetriever
from src.cvsearch.ranking import HybridRanker
from src.cvsearch.planner import Planner
from src.cvsearch.parser import Criteria

def default_run_dir() -> str:
    """Helper for default run dir stamp (used by main)."""
    return f"runs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

class SearchProcessor:
    """
    Orchestrates the full search pipeline:
    1. Instantiates all necessary services (retrievers, rankers).
    2. Runs single-seat searches (gate -> retrieve -> rank).
    3. Runs multi-seat project searches by calling the Planner and
       looping over single-seat searches.
    """

    def __init__(self, db: CVDatabase, client: OpenAIClient, settings: Settings):
        self.db = db
        self.client = client
        self.settings = settings

        self.gating_filter = GatingFilter(db)
        self.lexical_retriever = LexicalRetriever(db)
        self.semantic_retriever = LocalSemanticRetriever(db, settings)
        self.hybrid_ranker = HybridRanker(db, settings)
        self.planner = Planner()

    def _extract_seat(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts the first member from criteria as the 'seat'."""
        seat = criteria["team_size"]["members"][0]
        return {
            "role": seat["role"],
            "seniority": seat["seniority"],
            "domains": seat.get("domains", []),
            "must_have": seat.get("tech_tags", []),
            "nice_to_have": seat.get("nice_to_have", []),
        }

    def _fingerprint(self, seat: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Creates a simple hashable dict for the query."""
        return {
            "role": seat["role"],
            "seniority": seat["seniority"],
            "domains": sorted(seat["domains"]),
            "must_have": sorted(seat["must_have"]),
            "nice_to_have": sorted(seat["nice_to_have"]),
            "mode": mode,
        }

    def _write_run_artifacts(self, run_dir: str, payload: Dict[str, Any]) -> None:
        """Writes all debug and result files for a run."""
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "gating.sql.txt"), "w", encoding="utf-8") as f:
            f.write((payload.get("gating_sql") or "").strip() + "\n")
        with open(os.path.join(run_dir, "gating.explain.txt"), "w", encoding="utf-8") as f:
            for row in payload.get("gating_explain", []):
                f.write(json.dumps(row) + "\n")
        if payload.get("ranking_sql"):
            with open(os.path.join(run_dir, "ranking.sql.txt"), "w", encoding="utf-8") as f:
                f.write((payload.get("ranking_sql") or "").strip() + "\n")
        if payload.get("ranking_explain"):
            with open(os.path.join(run_dir, "ranking.explain.txt"), "w", encoding="utf-8") as f:
                for row in payload.get("ranking_explain", []):
                    f.write(json.dumps(row) + "\n")
        if payload.get("vs_query"):
            with open(os.path.join(run_dir, "vs.query.txt"), "w", encoding="utf-8") as f:
                f.write((payload.get("vs_query") or "").strip() + "\n")
        if payload.get("vs_results"):
            with open(os.path.join(run_dir, "vs.results.json"), "w", encoding="utf-8") as f:
                json.dump(payload["vs_results"], f, indent=2, ensure_ascii=False)
        if payload.get("fusion"):
            with open(os.path.join(run_dir, "ranking.fusion.json"), "w", encoding="utf-8") as f:
                json.dump(payload["fusion"], f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload["metrics"], f, indent=2)
        with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(payload["results"], f, indent=2)

    def _get_single_justification(self, candidate_id: str, seat_json: str) -> Tuple[str, Dict[str, Any]]:
        """
        Worker function: Fetches context and calls LLM for one candidate.
        Returns a tuple of (candidate_id, justification_dict).
        """

        local_db = None
        context = None
        try:
            local_db = CVDatabase(self.settings)
            context = local_db.get_full_candidate_context(candidate_id)
        finally:
            if local_db:
                local_db.close()

        if not context:
            return candidate_id, {
                "match_summary": "Error: Candidate context not found.",
                "strength_analysis": [],
                "gap_analysis": ["Could not retrieve candidate document from DB."],
                "overall_match_score": 0.0
            }

        cv_context_str = (
            f"SUMMARY:\n{context.get('summary_text', '')}\n\n"
            f"EXPERIENCE:\n{context.get('experience_text', '')}\n\n"
            f"TAGS:\n{context.get('tags_text', '')}"
        )

        justification = self.client.get_candidate_justification(
            seat_details=seat_json,
            cv_context=cv_context_str
        )
        return candidate_id, justification

    def _get_justifications_parallel(self, final_results: List[Dict[str, Any]], seat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Orchestrator: Gets justifications for all top-K candidates in parallel.
        """
        seat_json = json.dumps(seat, indent=2)
        candidate_ids = [r['candidate_id'] for r in final_results]

        justification_map = {}
        if not candidate_ids:
            return justification_map

        with ThreadPoolExecutor(max_workers=len(candidate_ids)) as executor:
            futures = [
                executor.submit(self._get_single_justification, cid, seat_json)
                for cid in candidate_ids
            ]

            for future in futures:
                cid, justification = future.result()
                justification_map[cid] = justification

        return justification_map

    def search_for_seat(self,
                        criteria: Dict[str, Any],
                        top_k: int = 10,
                        run_dir: str | None = None,
                        mode_override: str | None = None,
                        vs_topk_override: int | None = None,
                        with_justification: bool = True
                        ) -> Dict[str, Any]:
        """
        Orchestrates the full pipeline for a SINGLE seat.
        (Logic moved from old search.py:search_seat)
        """
        mode = (mode_override or self.settings.search_mode).lower()
        vs_topk = vs_topk_override or self.settings.search_vs_topk

        t0 = time.perf_counter()
        seat = self._extract_seat(criteria)

        g0 = time.perf_counter()
        gated_ids, gating_sql, gating_plan = self.gating_filter.filter_candidates(seat)
        g1 = time.perf_counter()

        if not gated_ids:
            total = time.perf_counter() - t0
            payload = {
                "query_fingerprint": self._fingerprint(seat, mode),
                "metrics": {
                    "gate_count": 0, "gate_time_ms": round((g1 - g0) * 1000, 2),
                    "rank_time_ms": 0.0, "total_time_ms": round(total * 1000, 2),
                },
                "gating_sql": gating_sql, "gating_explain": gating_plan,
                "ranking_sql": None, "ranking_explain": [],
                "vs_query": None, "vs_results": None, "fusion": None,
                "results": [], "reason": "strict_gate_empty",
            }
            if run_dir: self._write_run_artifacts(run_dir, payload)
            return payload

        lex_limit = max(top_k, vs_topk, len(gated_ids))
        r0 = time.perf_counter()
        lex_rows, ranking_sql, ranking_plan = self.lexical_retriever.search(
            gated_ids, seat, lex_limit
        )
        r1 = time.perf_counter()
        rank_time_ms = round((r1 - r0) * 1000, 2)

        sem_raw = self.semantic_retriever.search(
            gated_ids, seat, vs_topk
        )
        sem_hits = sem_raw.get("hits", [])

        final_results, fusion_dump = self.hybrid_ranker.rank(
            seat, lex_rows, sem_hits, mode, top_k
        )

        if with_justification and final_results:
            justification_map = self._get_justifications_parallel(final_results, seat)
            for result_item in final_results:
                cid = result_item['candidate_id']
                result_item['llm_justification'] = justification_map.get(cid)

        total = time.perf_counter() - t0
        payload = {
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
            "fusion": fusion_dump if fusion_dump else None,
            "results": final_results,
            "gating_sql": gating_sql,
            "gating_explain": gating_plan,
            "ranking_sql": ranking_sql,
            "ranking_explain": ranking_plan,
        }

        if run_dir: self._write_run_artifacts(run_dir, payload)
        return payload

    def search_for_project(self,
                           criteria: Criteria,
                           top_k: int = 3,
                           run_dir: Optional[str] = None,
                           raw_text: Optional[str] = None,
                           with_justification: bool = True
                           ) -> Dict[str, Any]:
        """
        Orchestrates a multi-seat search for a full project.
        (Logic moved from planner.py:search_project)
        """
        crit_with_seats = self.planner.derive_project_seats(criteria, raw_text=raw_text)
        base_dict = self.planner._criteria_dict(crit_with_seats)

        out_dir = run_dir or default_run_dir()
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        seats = base_dict["team_size"]["members"]
        aggregated: List[Dict[str, Any]] = []
        gaps: List[int] = []

        for idx, seat in enumerate(seats, start=1):
            seat_dir = os.path.join(out_dir, f"seat_{idx:02d}_{seat['role']}")
            single_criteria = self.planner._pack_single_seat_criteria(base_dict, seat)

            payload = self.search_for_seat(
                criteria=single_criteria,
                top_k=top_k,
                run_dir=seat_dir,
                mode_override=None,
                vs_topk_override=None,
                with_justification=with_justification
            )

            aggregated.append({
                "index": idx,
                "role": seat["role"],
                "criteria": single_criteria,
                "metrics": payload.get("metrics", {}),
                "results": payload.get("results", []),
            })
            if not payload.get("results"):
                gaps.append(idx)

        return {
            "project_criteria": base_dict,
            "seats": aggregated,
            "gaps": gaps,
            "run_dir": out_dir,
            "note": "Stateless multi-seat search; per-seat artifacts live under run_dir.",
        }