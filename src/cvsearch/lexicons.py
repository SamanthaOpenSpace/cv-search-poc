from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, List

# Removed PKG_DIR, REPO_ROOT, DEFAULT_LEXICON_DIR, os.getenv

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f: return json.load(f)

def load_role_lexicon(lexicon_dir: Path) -> Dict[str, List[str]]:
    return _load_json(lexicon_dir / "role_lexicon.json")

def load_tech_synonyms(lexicon_dir: Path) -> Dict[str, List[str]]:
    return _load_json(lexicon_dir / "tech_synonyms.json")

def load_domain_lexicon(lexicon_dir: Path) -> Dict[str, List[str]]:
    return _load_json(lexicon_dir / "domain_lexicon.json")