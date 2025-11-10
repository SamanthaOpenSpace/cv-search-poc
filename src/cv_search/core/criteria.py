from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional


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
