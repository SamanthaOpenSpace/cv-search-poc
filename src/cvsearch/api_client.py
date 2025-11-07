from __future__ import annotations

import os
import json
import tempfile
import sqlite3
import time
import re
import math
from typing import Dict, Any, Optional, List, Tuple, Iterable, Type
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, Field, model_validator
from openai import OpenAI

from src.cvsearch.settings import Settings
from src.cvsearch.lexicons import load_role_lexicon, load_tech_synonyms, load_domain_lexicon
from src.cvsearch.normalize import build_inverse_index, extract_by_lexicon
from src.cvsearch.storage import CVDatabase
from src.cvsearch.justification import CandidateJustification
# --- DELETED IMPORT ---
# from src.cvsearch.schemas.cv import ExperienceItemStrict, CandidateCVStrict

class SeniorityEnum(str, Enum):
    junior = "junior"
    middle = "middle"
    senior = "senior"
    lead = "lead"
    manager = "manager"
_PROJECT_TYPES = ("greenfield", "modernization", "migration", "support")
_SENIORITY = tuple(s.value for s in SeniorityEnum)
def _enum_from_keys(name: str, keys: Iterable[str]) -> Enum:
    uniq = {k: k for k in sorted(set(keys))}
    return Enum(name, uniq)
def _dedupe_enum_list(seq: List[Enum]) -> List[Enum]:
    seen: set[str] = set()
    out: List[Enum] = []
    for x in seq:
        val = x.value if isinstance(x, Enum) else str(x)
        if val not in seen:
            seen.add(val)
            out.append(x)
    return out


class OpenAIClient:
    """
    Centralized client for all OpenAI API interactions, including
    chat/parsing, embeddings, and vector store management.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.openai_api_key_str
        )
        self._strict_schema_cache: Dict[str, Any] = {}
        self._cv_schema_cache: Dict[str, Any] = {}

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Batch-embed texts using the model specified in settings.
        (This is now only used by the old, soon-to-be-deleted code)
        """
        if not texts:
            return []
        res = self.client.embeddings.create(
            model=self.settings.openai_embed_model,
            input=texts,
            encoding_format="float"
        )
        return [d.embedding for d in res.data]

    def _build_strict_schema(self, lexicon_dir: Path) -> Type[BaseModel]:
        """
        Create strict Pydantic models with closed enums built from lexicons.
        Caches the schema class for efficiency.
        """
        cache_key = str(lexicon_dir)
        if cache_key in self._strict_schema_cache:
            return self._strict_schema_cache[cache_key]

        role_lex = load_role_lexicon(lexicon_dir)
        tech_syn = load_tech_synonyms(lexicon_dir)
        domain_lex = load_domain_lexicon(lexicon_dir)

        RoleEnumDyn = _enum_from_keys("RoleEnumDyn", role_lex.keys())
        TechEnumDyn = _enum_from_keys("TechEnumDyn", tech_syn.keys())
        DomainEnumDyn = _enum_from_keys("DomainEnumDyn", domain_lex.keys())
        ProjectTypeEnumDyn = _enum_from_keys("ProjectTypeEnumDyn", _PROJECT_TYPES)

        _FE_BLACKLIST: set[str] = {
            "kubernetes", "docker", "postgresql", "sql_server", "mysql",
            "redis", "kafka", "rabbitmq", "aws", "azure", "gcp"
        }

        class TeamMemberStrict(BaseModel):
            role: RoleEnumDyn = Field(..., description="Canonical role key from role_lexicon.json")
            seniority: Optional[SeniorityEnum] = Field(default=None)
            domains: List[DomainEnumDyn] = Field(default_factory=list)
            tech_tags: List[TechEnumDyn] = Field(default_factory=list)
            nice_to_have: List[TechEnumDyn] = Field(default_factory=list)
            rationale: Optional[str] = Field(default=None)

            @model_validator(mode="after")
            def _dedupe_and_partition(self) -> "TeamMemberStrict":
                self.domains = _dedupe_enum_list(list(self.domains))
                self.tech_tags = _dedupe_enum_list(list(self.tech_tags))
                self.nice_to_have = _dedupe_enum_list(list(self.nice_to_have))
                must_vals = {t.value for t in self.tech_tags}
                self.nice_to_have = [t for t in self.nice_to_have if t.value not in must_vals]
                if self.role.value == "frontend_engineer":
                    self.tech_tags = [t for t in self.tech_tags if t.value not in _FE_BLACKLIST]
                    self.nice_to_have = [t for t in self.nice_to_have if t.value not in _FE_BLACKLIST]
                return self

        class TeamSizeStrict(BaseModel):
            total: Optional[int] = Field(default=None, ge=0)
            members: List[TeamMemberStrict] = Field(default_factory=list)

            @model_validator(mode="after")
            def ensure_total(self) -> "TeamSizeStrict":
                if self.total is None:
                    self.total = len(self.members)
                return self

        class CriteriaStrict(BaseModel):
            domain: List[DomainEnumDyn] = Field(default_factory=list)
            tech_stack: List[TechEnumDyn] = Field(default_factory=list)
            expert_roles: List[RoleEnumDyn] = Field(default_factory=list)
            project_type: Optional[ProjectTypeEnumDyn] = Field(default=None)
            team_size: TeamSizeStrict

            @model_validator(mode="after")
            def dedupe_all(self) -> "CriteriaStrict":
                self.domain = _dedupe_enum_list(list(self.domain))
                self.tech_stack = _dedupe_enum_list(list(self.tech_stack))
                self.expert_roles = _dedupe_enum_list(list(self.expert_roles))
                return self

            @model_validator(mode="after")
            def ensure_roles_cover_members(self) -> "CriteriaStrict":
                existing_vals = [r.value for r in self.expert_roles]
                for m in self.team_size.members:
                    rv = m.role.value
                    if rv not in existing_vals:
                        self.expert_roles.append(RoleEnumDyn(rv))
                        existing_vals.append(rv)
                self.expert_roles = _dedupe_enum_list(list(self.expert_roles))
                return self

        self._strict_schema_cache[cache_key] = (CriteriaStrict, role_lex, tech_syn, domain_lex)
        return self._strict_schema_cache[cache_key]

    def _preselect_candidates(self, text: str, lex: Dict[str, List[str]], max_aliases_per_key: int = 6) -> Dict[str, List[str]]:
        inv = build_inverse_index(lex)
        matched = extract_by_lexicon(text, inv)
        out: Dict[str, List[str]] = {}
        for canon in matched:
            aliases = [canon] + (lex.get(canon, [])[:max_aliases_per_key])
            out[canon] = aliases
        return out

    def _build_system_prompt(self, text: str,
                             role_lex: Dict[str, List[str]],
                             tech_syn: Dict[str, List[str]],
                             domain_lex: Dict[str, List[str]]) -> str:

        cand_roles = self._preselect_candidates(text, role_lex)
        cand_techs = self._preselect_candidates(text, tech_syn)
        cand_domains = self._preselect_candidates(text, domain_lex)

        def block(name: str, cand: Dict[str, List[str]], full_keys: Iterable[str]) -> str:
            if cand:
                lines = [f"- {canon}: {', '.join(aliases)}" for canon, aliases in cand.items()]
                return f"{name} (canonical → aliases):\n" + "\n".join(lines)
            sample = ", ".join(list(sorted(set(full_keys)))[:40])
            return f"{name} (canonical keys): {sample}"

        instr = [
            "You normalize a free-text client brief into a strict JSON object.",
            "Rules:",
            "- Only output values that are in the allowed enums (the schema enforces this).",
            "- Map any mention or synonym to the proper canonical key before output.",
            "- For team_size: materialize one member per requested headcount; set total = len(members).",
            "- expert_roles must include every role listed in team_size.members (no omissions).",
            f"- project_type is one of: {', '.join(_PROJECT_TYPES)}.",
            f"- seniority is one of: {', '.join(_SENIORITY)}.",
            "- For each team member, set tech_tags (primary) and nice_to_have (optional/possible/nice to have) .",
            "- Partition the global tech list across roles using typical responsibility boundaries.",
            "- For example Do NOT assign infra/DB/cloud tech (e.g., kubernetes, docker, postgresql, kafka, cloud providers) to frontend_engineer unless the brief explicitly demands it.",
            "- Always set member.domains; inherit from top-level if not seat-specific.",
            "- Include a one-line rationale explaining why allocated this member.",
            "",
            block("ROLES",   cand_roles,   role_lex.keys()),
            "",
            block("TECH",    cand_techs,   tech_syn.keys()),
            "",
            block("DOMAINS", cand_domains, domain_lex.keys()),
        ]
        return "\n".join(instr)

    def get_structured_criteria(self, text: str, model: str, settings: Settings) -> Dict[str, Any]:
        """
        Single call that returns canonicalized, schema-valid data.
        (Moved from parser.py)
        """
        CriteriaStrict, role_lex, tech_syn, domain_lex = self._build_strict_schema(settings.lexicon_dir)
        sys_prompt = self._build_system_prompt(text, role_lex, tech_syn, domain_lex)

        resp = self.client.responses.parse(
            model=model,
            instructions=sys_prompt,
            input=text,
            text_format=CriteriaStrict,
            temperature=0,
            max_output_tokens=600,
        )
        parsed: BaseModel = resp.output_parsed  # type: ignore
        return parsed.model_dump(mode="json")

    # --- START REVISED CV PARSING METHODS ---

    def _build_cv_schema(self, lexicon_dir: Path) -> Tuple[Type[BaseModel], Dict, Dict, Dict]:
        """
        Create strict Pydantic models for CV parsing, using lexicons
        for normalization. Caches for efficiency.
        """
        cache_key = str(lexicon_dir)
        if cache_key in self._cv_schema_cache:
            return self._cv_schema_cache[cache_key]

        # 1. Load lexicons
        role_lex = load_role_lexicon(lexicon_dir)
        tech_syn = load_tech_synonyms(lexicon_dir)
        domain_lex = load_domain_lexicon(lexicon_dir)

        # 2. Create dynamic Enums
        RoleEnumDyn = _enum_from_keys("RoleEnumDyn", role_lex.keys())
        TechEnumDyn = _enum_from_keys("TechEnumDyn", tech_syn.keys())
        DomainEnumDyn = _enum_from_keys("DomainEnumDyn", domain_lex.keys())

        # 3. Define schemas *inside* the function, using the dynamic Enums
        #    This mirrors the robust pattern in _build_strict_schema

        class ExperienceItemStrict(BaseModel):
            """
            Pydantic model for a single experience entry in a CV,
            enforcing normalized tags.
            """
            title: str = Field(..., description="The job title, e.g., 'Software Engineer'.")
            company: str = Field(..., description="The company name or project description.")
            domain_tags: List[DomainEnumDyn] = Field(
                default_factory=list,
                description="A list of canonical domain tags, e.g., 'fintech'."
            )
            tech_tags: List[TechEnumDyn] = Field(
                default_factory=list,
                description="A list of canonical tech tags used in this role."
            )
            from_date: Optional[str] = Field(
                default=None,
                alias="from",
                description="Start date (YYYY-MM), if available."
            )
            to_date: Optional[str] = Field(
                default=None,
                alias="to",
                description="End date (YYYY-MM or 'Present'), if available."
            )
            highlights: List[str] = Field(
                default_factory=list,
                description="A list of bullet points for responsibilities or achievements."
            )

            class Config:
                populate_by_name = True # Allows using 'from' and 'to' as field names

        class CandidateCVStrict(BaseModel):
            """
            Pydantic model for a full candidate CV, enforcing normalized
            tags from lexicons.
            """
            name: str = Field(..., description="Candidate's full name.")
            location: Optional[str] = Field(
                default=None,
                description="Candidate's location, if available."
            )
            seniority: Optional[SeniorityEnum] = Field(
                default=None,
                description="Inferred seniority level."
            )
            role_tags: List[RoleEnumDyn] = Field(
                default_factory=list,
                description="A list of canonical role tags, e.g., 'backend_engineer'."
            )
            summary: str = Field(
                ...,
                description="The 1-3 sentence summary from the top of the CV."
            )
            experience: List[ExperienceItemStrict] = Field(
                default_factory=list,
                description="A list of the candidate's work experiences or projects."
            )
            tech_tags: List[TechEnumDyn] = Field(
                default_factory=list,
                description="A top-level list of all canonical tech tags."
            )
            unmapped_tags: Optional[str] = Field(
                default=None,
                description="A comma-separated list of technologies found in the CV that were not in the official tech enum. This is for admin review."
            )

        # 4. Cache and return
        self._cv_schema_cache[cache_key] = (CandidateCVStrict, role_lex, tech_syn, domain_lex)
        return self._cv_schema_cache[cache_key]

    def _build_cv_system_prompt(self,
                                text: str,
                                role_lex: Dict[str, List[str]],
                                tech_syn: Dict[str, List[str]],
                                domain_lex: Dict[str, List[str]]) -> str:
        """
        Builds the system prompt for CV parsing, including lexicon hints.
        """
        cand_roles = self._preselect_candidates(text, role_lex)
        cand_techs = self._preselect_candidates(text, tech_syn, max_aliases_per_key=3) # Limit noise
        cand_domains = self._preselect_candidates(text, domain_lex)

        # Reuse the 'block' helper from _build_system_prompt
        def block(name: str, cand: Dict[str, List[str]], full_keys: Iterable[str]) -> str:
            if cand:
                lines = [f"- {canon}: {', '.join(aliases)}" for canon, aliases in cand.items()]
                return f"{name} (canonical → aliases):\n" + "\n".join(lines)
            sample = ", ".join(list(sorted(set(full_keys)))[:40])
            return f"{name} (canonical keys): {sample}"

        instr = [
            "You are an expert HR and technical recruiting analyst. Your task is to parse the raw text from a CV and convert it into a structured JSON object, adhering strictly to the provided Pydantic schema.",
            "Rules:",
            "- Only output values that are in the allowed enums (the schema enforces this).",
            "- Map any mention or synonym (e.g., 'K8S', 'Kubernetes (AKS)') to the proper canonical key (e.g., 'kubernetes').",
            "- Infer 'seniority' and 'role_tags' from the candidate's title and summary.",
            "- Aggregate all technologies from 'Qualifications', 'Skills', and 'Tools' sections into the single top-level 'tech_tags' list.",
            "- Treat each 'Project' or 'Work Experience' entry as one item in the 'experience' array.",
            "- **IMPORTANT**: For each 'experience' item, if a company name is not obvious, use the 'Project Description' text as the 'company' field.",
            "- Map 'Responsibilities' to the 'highlights' array.",
            "- Map project-specific technologies to the 'experience.tech_tags' list.",
            "- Infer 'experience.domain_tags' from the project context (e.g., 'digital banking' -> 'fintech').",
            "- If you find technologies not in the 'TECH' hints, add them to the 'unmapped_tags' field as a comma-separated string.",
            "",
            block("ROLES",   cand_roles,   role_lex.keys()),
            "",
            block("TECH",    cand_techs,   tech_syn.keys()),
            "",
            block("DOMAINS", cand_domains, domain_lex.keys()),
        ]
        return "\n".join(instr)

    def get_structured_cv(self, text: str, model: str, settings: Settings) -> Dict[str, Any]:
        """
        Single call that parses raw CV text and returns normalized,
        schema-valid data.
        """
        # 1. Get the strict schema and lexicons
        CandidateCVStrict, role_lex, tech_syn, domain_lex = self._build_cv_schema(settings.lexicon_dir)

        # 2. Build the targeted system prompt
        sys_prompt = self._build_cv_system_prompt(text, role_lex, tech_syn, domain_lex)

        # 3. Call the LLM
        resp = self.client.responses.parse(
            model=model,
            instructions=sys_prompt,
            input=text,
            text_format=CandidateCVStrict,
            temperature=0,
            max_output_tokens=2048, # CVs can be long
        )
        parsed: BaseModel = resp.output_parsed  # type: ignore

        # 4. Return as a dict
        return parsed.model_dump(mode="json", by_alias=True) # Use by_alias=True for from/to

    # --- END REVISED CV PARSING METHODS ---

    def _build_justification_prompt(self, seat_details: str, cv_context: str) -> Tuple[str, str]:
        """Builds the system and user prompts for justification."""

        system_prompt = (
            "You are an expert technical recruiter and talent analyst. "
            "Your job is to write a concise, evidence-based justification for why a "
            "candidate is a good or bad match for a specific role, based on their CV. "
            "You must be critical and balanced, citing specific strengths and weaknesses. "
            "You must provide an 'overall_match_score' from 0.0 (no match) to 1.0 (perfect match) "
            "based *only* on the provided context."
        )

        user_prompt = f"""
Here are the job requirements (the 'seat'):
---
{seat_details}
---

Here is the candidate's CV data:
---
{cv_context}
---

Analyze the candidate's CV against the job requirements and provide your justification.
Base your analysis *only* on the provided texts.
"""
        return system_prompt, user_prompt

    def get_candidate_justification(self, seat_details: str, cv_context: str) -> Dict[str, Any]:
        """
        Calls the LLM to get a structured justification for a single candidate.
        """
        system_prompt, user_prompt = self._build_justification_prompt(seat_details, cv_context)

        try:
            resp = self.client.responses.parse(
                model=self.settings.openai_model,
                instructions=system_prompt,
                input=user_prompt,
                text_format=CandidateJustification,
                temperature=0,
            )
            parsed: CandidateJustification = resp.output_parsed  # type: ignore
            return parsed.model_dump(mode="json")
        except Exception as e:
            return {
                "match_summary": "Error generating justification.",
                "strength_analysis": [],
                "gap_analysis": [f"Error: {str(e)}"],
                "overall_match_score": 0.0
            }