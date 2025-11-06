from __future__ import annotations

import re
from typing import Dict, List, Iterable, Set

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def build_inverse_index(lex: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Build map from any synonym -> canonical label.
    """
    inv: Dict[str, str] = {}
    for canon, syns in lex.items():
        inv[_norm(canon)] = canon
        for s in syns:
            inv[_norm(s)] = canon
    return inv

def extract_by_lexicon(text: str, inv_index: Dict[str, str]) -> List[str]:
    """
    Greedy substring match using the inverse index (normalized).
    Returns unique canonicals in insertion order.
    """
    t = _norm(text)
    seen: Set[str] = set()
    out: List[str] = []
    # Match longer phrases first
    keys = sorted(inv_index.keys(), key=len, reverse=True)
    for k in keys:
        if k and k in t:
            canon = inv_index[k]
            if canon not in seen:
                seen.add(canon)
                out.append(canon)
    return out

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

def expand_terms(canon_list: Iterable[str], lex: Dict[str, List[str]]) -> List[str]:
    """
    Expand canonical labels to a list including their synonyms (for FTS queries).
    """
    out: List[str] = []
    for c in canon_list:
        out.append(c)
        out.extend(lex.get(c, []))
    # de-dup while preserving order
    seen = set()
    keep = []
    for x in out:
        if x.lower() not in seen:
            seen.add(x.lower())
            keep.append(x)
    return keep