from __future__ import annotations
import os, math
from typing import List
# --- embed_texts function REMOVED ---

def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b): return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)