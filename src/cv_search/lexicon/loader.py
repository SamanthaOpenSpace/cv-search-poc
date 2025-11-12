from __future__ import annotations
import json
from pathlib import Path
from typing import List, TypedDict

# Removed PKG_DIR, REPO_ROOT, DEFAULT_LEXICON_DIR, os.getenv

def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class RoleExpertiseEntry(TypedDict):
    roles: List[str]
    expertises: List[str]

def load_role_lexicon(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical role keys."""
    return _load_json(lexicon_dir / "role_lexicon.json")


def load_expertise_lexicon(lexicon_dir: Path) -> List[RoleExpertiseEntry]:
    """Loads the role-to-expertise rollups."""
    data = _load_json(lexicon_dir / "role_expertise_lexicon.json")
    if not isinstance(data, list):
        raise ValueError("role_expertise_lexicon.json must be a list of mappings")
    normalized: List[RoleExpertiseEntry] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each expertise entry must be an object")
        roles = entry.get("roles", [])
        expertises = entry.get("expertises", [])
        if not isinstance(roles, list) or not isinstance(expertises, list):
            raise ValueError("roles and expertises must be lists")
        normalized.append(
            RoleExpertiseEntry(
                roles=[str(role).strip().lower() for role in roles if str(role).strip()],
                expertises=[str(exp).strip().lower() for exp in expertises if str(exp).strip()],
            )
        )
    return normalized

def load_tech_synonyms(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical tech keys."""
    return _load_json(lexicon_dir / "tech_synonyms.json")

def load_domain_lexicon(lexicon_dir: Path) -> List[str]:
    """Loads the flat list of canonical domain keys."""
    return _load_json(lexicon_dir / "domain_lexicon.json")