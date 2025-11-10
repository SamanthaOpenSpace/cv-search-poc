from __future__ import annotations

from typing import List

from cv_search.clients.openai_client import OpenAIClient
from cv_search.config.settings import Settings
from cv_search.core.criteria import Criteria, TeamMember, TeamSize


def _build_team_size(payload: dict) -> TeamSize | None:
    raw_members: List[dict] = payload.get("members") or []
    members = [
        TeamMember(
            role=m["role"],
            seniority=m.get("seniority"),
            domains=m.get("domains", []) or [],
            tech_tags=m.get("tech_tags", []) or [],
            nice_to_have=m.get("nice_to_have", []) or [],
            rationale=m.get("rationale"),
        )
        for m in raw_members
    ]
    if not members and payload.get("total") is None:
        return None
    return TeamSize(total=payload.get("total"), members=members)


def parse_request(text: str, model: str, settings: Settings, client: OpenAIClient) -> Criteria:
    """Extract canonical search criteria using the LLM client."""

    data = client.get_structured_criteria(text, model=model, settings=settings)
    team_payload = data.get("team_size") or {}
    team_size = _build_team_size(team_payload)

    return Criteria(
        domain=data.get("domain", []) or [],
        tech_stack=data.get("tech_stack", []) or [],
        expert_roles=data.get("expert_roles", []) or [],
        project_type=data.get("project_type"),
        team_size=team_size,
    )
