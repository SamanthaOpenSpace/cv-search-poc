from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Any

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_mock_cvs(data_dir: Path) -> List[Dict[str, Any]]:
    """Load mock CVs from the provided data directory."""
    return _load_json(data_dir / "mock_cvs.json")