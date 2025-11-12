from __future__ import annotations

from typing import Dict, List

from cv_search.clients.openai_client import OpenAIClient
from cv_search.config.settings import Settings
from cv_search.lexicon.loader import (
    RoleExpertiseEntry,
    load_domain_lexicon,
    load_expertise_lexicon,
    load_role_lexicon,
    load_tech_synonyms,
)
from cv_search.planner.service import Planner


def load_stateless_services() -> Dict[str, object]:
    settings = Settings()
    client = OpenAIClient(settings)
    planner = Planner()

    lexicon_dir = settings.lexicon_dir
    role_lex: List[str] = load_role_lexicon(lexicon_dir)
    tech_lex: List[str] = load_tech_synonyms(lexicon_dir)
    domain_lex: List[str] = load_domain_lexicon(lexicon_dir)
    expertise_lex: List[RoleExpertiseEntry] = load_expertise_lexicon(lexicon_dir)

    return {
        "settings": settings,
        "client": client,
        "planner": planner,
        "role_lex": role_lex,
        "tech_lex": tech_lex,
        "domain_lex": domain_lex,
        "expertise_lex": expertise_lex,
    }
