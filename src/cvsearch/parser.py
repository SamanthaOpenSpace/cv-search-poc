from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Iterable
from enum import Enum
from pathlib import Path

from src.cvsearch.api_client import OpenAIClient
from src.cvsearch.settings import Settings
class SeniorityEnum(str, Enum):
    junior = "junior"
    middle = "middle"
    senior = "senior"
    lead = "lead"
    manager = "manager"

@dataclass
class TeamMember:
    role: str
    seniority: Optional[SeniorityEnum] = None
    domains: List[str] = field(default_factory=list)
    tech_tags: List[str] = field(default_factory=list)
    nice_to_have: List[str] = field(default_factory=list)
    rationale: Optional[str] = None

@dataclass
class TeamSize:
    total: Optional[int] = None
    members: List[TeamMember] = field(default_factory=list)

@dataclass
class Criteria:
    domain: List[str]
    tech_stack: List[str]
    expert_roles: List[str]
    project_type: Optional[str] = None
    team_size: Optional[TeamSize] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

def parse_request(text: str, model: str, settings: Settings, client: OpenAIClient) -> Criteria:
    """
    LLM-first extractor (strict). Returns canonical labels directly from the schema.
    This function now delegates the API call to the client.
    """
    data = client.get_structured_criteria(
        text,
        model=model,
        settings=settings
    )
    ts = data.get("team_size") or {}
    members = [
        TeamMember(
            role=m["role"],
            seniority=m.get("seniority"),
            domains=m.get("domains", []) or [],
            tech_tags=m.get("tech_tags", []) or [],
            nice_to_have=m.get("nice_to_have", []) or [],
            rationale=m.get("rationale"),
        )
        for m in (ts.get("members") or [])
    ]
    team = TeamSize(total=ts.get("total"), members=members) if (members or ts.get("total") is not None) else None
    return Criteria(
        domain=data.get("domain", []) or [],
        tech_stack=data.get("tech_stack", []) or [],
        expert_roles=data.get("expert_roles", []) or [],
        project_type=data.get("project_type"),
        team_size=team,
    )