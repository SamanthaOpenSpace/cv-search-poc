from __future__ import annotations

import re
from typing import Dict, List, Iterable, Set

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

# REMOVED: build_inverse_index(lex: Dict[str, List[str]])
# This function is incompatible with the new List[str] lexicon format.

# REMOVED: extract_by_lexicon(text: str, inv_index: Dict[str, str])
# This function depended on build_inverse_index.

PROJECT_TYPES = ["greenfield", "modernization", "migration", "support"]

def extract_project_type(text: str) -> str | None:
    t = _norm(text)
    for p in PROJECT_TYPES:
        if p in t:
            return p
    return None

TEAM_SIZE_RE = re.compile(
    r"(?:team(?:\s*size)?\s*(?:of)?\s*)?(\d{1,3})\s*[-–—]\s*(\d{1,3})|team\s*(?:size)?\s*(?:of\s*)?(\d{1,3})",
    re.I,
)

def extract_team_size(text: str) -> str | None:
    m = TEAM_SIZE_RE.search(text)
    if not m:
        return None
    if m.group(1) and m.group(2):
        return f"{m.group(1)}-{m.group(2)}"
    return m.group(3)

# Heuristic role hints when not explicit
def infer_roles(text: str, existing: Iterable[str]) -> List[str]:
    t = _norm(text)
    out = list(existing)
    if not any("data analyst" in r.lower() for r in out):
        if any(w in t for w in ["analytics", "data analytics", "kpi", "bi", "dashboard"]):
            out.append("Data Analyst")
    if not any("ui developer" in r.lower() for r in out):
        if any(w in t for w in ["ui", "frontend", "user interface", "react", "typescript"]):
            out.append("UI Developer")
    return out

# REMOVED: expand_terms(canon_list: Iterable[str], lex: Dict[str, List[str]])
# This function is incompatible with the new List[str] lexicon format.