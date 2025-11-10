from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class SearchRunArtifactWriter:
    """Persist search diagnostics and results for later inspection."""

    def write(self, run_dir: str | Path, payload: Dict[str, Any]) -> None:
        target = Path(run_dir)
        target.mkdir(parents=True, exist_ok=True)

        def _write_text(name: str, content: str) -> None:
            (target / name).write_text(content, encoding="utf-8")

        def _write_json(name: str, content: Any) -> None:
            (target / name).write_text(
                json.dumps(content, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        gating_sql = (payload.get("gating_sql") or "").strip()
        _write_text("gating.sql.txt", f"{gating_sql}\n")

        gating_plan = payload.get("gating_explain", [])
        if gating_plan:
            _write_json("gating.explain.json", gating_plan)

        ranking_sql = payload.get("ranking_sql")
        if ranking_sql:
            _write_text("ranking.sql.txt", f"{ranking_sql.strip()}\n")

        ranking_plan = payload.get("ranking_explain", [])
        if ranking_plan:
            _write_json("ranking.explain.json", ranking_plan)

        vs_query = payload.get("vs_query")
        if vs_query:
            _write_text("vs.query.txt", vs_query.strip() + "\n")

        vs_results = payload.get("vs_results")
        if vs_results:
            _write_json("vs.results.json", vs_results)

        fusion = payload.get("fusion")
        if fusion:
            _write_json("ranking.fusion.json", fusion)

        _write_json("metrics.json", payload.get("metrics", {}))
        _write_json("results.json", payload.get("results", []))
