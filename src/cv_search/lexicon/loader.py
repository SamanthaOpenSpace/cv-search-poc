from __future__ import annotations
import json
from pathlib import Path
from typing import List

# Removed PKG_DIR, REPO_ROOT, DEFAULT_LEXICON_DIR, os.getenv

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_role_lexicon(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical role keys."""
    return _load_json(lexicon_dir / "role_lexicon.json")


def load_expertise_lexicon(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical expertise keys."""
    data = _load_json(lexicon_dir / "expertise_lexicon.json")
    if not isinstance(data, list):
        raise ValueError("expertise_lexicon.json must be a list")

    normalized: List[str] = []
    seen = set()
    for item in data:
        text = str(item).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)

    return normalized

def load_tech_synonyms(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical tech keys."""
    return _load_json(lexicon_dir / "tech_synonyms.json")

def load_domain_lexicon(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical domain keys."""
    return _load_json(lexicon_dir / "domain_lexicon.json")