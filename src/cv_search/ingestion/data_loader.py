from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_mock_cvs(test_data_dir: Path) -> List[Dict[str, Any]]:
    """Load mock CVs from the test data directory (data/test/mock_cvs.json)."""
    return _load_json(test_data_dir / "mock_cvs.json")
