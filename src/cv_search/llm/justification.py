from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from cv_search.clients.openai_client import OpenAIClient
from cv_search.config.settings import Settings
from cv_search.db.database import CVDatabase
from cv_search.llm.logger import set_run_dir as _llm_set_run_dir, reset_run_dir as _llm_reset_run_dir


class JustificationService:
    """Generate LLM justifications for ranked candidates."""

    def __init__(self, client: OpenAIClient, settings: Settings):
        self.client = client
        self.settings = settings

    def _build_cv_context(self, candidate_id: str) -> str | None:
        database = CVDatabase(self.settings)
        try:
            context = database.get_full_candidate_context(candidate_id)
        finally:
            database.close()
        if not context:
            return None
        summary = context.get("summary_text", "")
        experience = context.get("experience_text", "")
        tags = context.get("tags_text", "")
        return f"SUMMARY:\n{summary}\n\nEXPERIENCE:\n{experience}\n\nTAGS:\n{tags}"

    def _justify_candidate(self, candidate_id: str, seat_json: str, run_dir: str | None) -> Dict[str, object]:
        token = _llm_set_run_dir(run_dir) if run_dir else None
        try:
            cv_context = self._build_cv_context(candidate_id)
            if cv_context is None:
                return {
                    "match_summary": "Error: Candidate context not found.",
                    "strength_analysis": [],
                    "gap_analysis": ["Could not retrieve candidate document from DB."],
                    "overall_match_score": 0.0,
                }
            return self.client.get_candidate_justification(seat_json, cv_context)
        finally:
            if token is not None:
                _llm_reset_run_dir(token)

    def generate(self, candidates: List[Dict[str, object]], seat: Dict[str, object], run_dir: str | None = None) -> Dict[str, Dict[str, object]]:
        seat_json = json.dumps(seat, indent=2)
        candidate_ids = [item["candidate_id"] for item in candidates]
        if not candidate_ids:
            return {}
        results: Dict[str, Dict[str, object]] = {}
        with ThreadPoolExecutor(max_workers=len(candidate_ids)) as pool:
            futures = {pool.submit(self._justify_candidate, cid, seat_json, run_dir): cid for cid in candidate_ids}
            for future, cid in futures.items():
                results[cid] = future.result()
        return results
