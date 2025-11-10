# src/cvsearch/planner.py
# Stateless business logic for deriving project/presale teams.
# No orchestration, no DB, no API calls.

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

from cv_search.core.criteria import Criteria, SeniorityEnum, TeamMember, TeamSize

class Planner:
    """
    Contains stateless business logic for deriving team compositions
    from a user request.
    """

    def __init__(self):
        # --- (Internal constant helpers) ---
        self._AI_TEXT_TOKENS = {
            "openai", "chatgpt", "gpt", "llm", "ai chatbot", "ai assistant", "assistant"
        }
        self._AI_TECH_TOKENS = {
            "machine_learning", "deep_learning", "pytorch", "tensorflow",
            "huggingface", "vertex_ai", "azure_ml"
        }
        self._MOBILE_TECH_TOKENS = {
            "flutter", "react_native", "android", "ios", "kotlin", "swift", "xamarin", "dart"
        }
        self._WEB_TECH_TOKENS = {
            "react", "angular", "vue", "svelte", "nextjs", "nuxt", "sveltekit"
        }
        self._BACKEND_TECH_HINTS = {
            "dotnet", "nodejs", "java", "python", "go", "rust", "php", "ruby", "scala"
        }
        self._BACKEND_NICE_DEFAULTS = ["rest", "openapi", "oauth2", "postgresql"]

    # ---------- private helpers (signal detection) ----------

    def _text_has_any(self, text: Optional[str], needles: set[str]) -> bool:
        if not text:
            return False
        t = text.lower()
        return any(n in t for n in needles)

    def _has_any(self, tokens: List[str], needles: set[str]) -> bool:
        s = set((tokens or []))
        return any(x in s for x in needles)

    def _first_present(self, tokens: List[str], ordered: List[str]) -> Optional[str]:
        st = set(tokens or [])
        for k in ordered:
            if k in st:
                return k
        return None


    # ---------- public: presale team derivation ----------

    def derive_presale_team(self, crit: Criteria, raw_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a budget-agnostic presale team strictly from the brief's tech/features.
        """
        techs = crit.tech_stack or []
        ai = self._has_any(techs, self._AI_TECH_TOKENS) or self._text_has_any(raw_text, self._AI_TEXT_TOKENS)
        mobile = self._has_any(techs, self._MOBILE_TECH_TOKENS) or self._text_has_any(raw_text, {"ios", "android", "mobile"})

        minimum = [
            {"role": "Business Analyst", "purpose": "Uncover needs, success metrics, scope."},
            {"role": "Technical Architect", "purpose": "MVP skeleton: auth, data, integrations."},
            {"role": "Account Manager", "purpose": "Expectations, comms, next steps."},
        ]
        extended = []
        if ai:
            extended.append({"role": "AI Specialist", "purpose": "Model choice, prompting, evals, cost guardrails."})
        if mobile:
            extended.append({"role": "Mobile Development Expert", "purpose": "Flutter/React Native tradeoffs, app release."})

        return {
            "minimum_roles": minimum,
            "extended_roles": extended,
            "triggers": {"ai": ai, "mobile": mobile},
            "notes": "Composed purely from tech/features; no budget heuristics.",
        }


    # ---------- public: project seats derivation ----------

    def derive_project_seats(self, crit: Criteria, raw_text: Optional[str] = None) -> Criteria:
        """
        Deterministically build seat definitions (if not already provided) from normalized tech/features.
        Always returns a Criteria that has team_size.members >= 1.
        """
        if crit.team_size and (crit.team_size.members or 0):
            return crit

        techs = crit.tech_stack or []
        domains = crit.domain or []
        roles: List[TeamMember] = []

        ai = self._has_any(techs, self._AI_TECH_TOKENS) or self._text_has_any(raw_text, self._AI_TEXT_TOKENS)
        mobile = self._has_any(techs, self._MOBILE_TECH_TOKENS) or self._text_has_any(raw_text, {"ios", "android", "mobile"})
        web = self._has_any(techs, self._WEB_TECH_TOKENS) or self._text_has_any(raw_text, {"web", "react", "frontend", "ui"})
        needs_backend = (
                self._has_any(techs, self._BACKEND_TECH_HINTS)
                or self._text_has_any(raw_text, {"donate", "payment", "stripe", "paypal", "consequence", "partner", "auth", "api", "backend"})
        )

        if mobile:
            must = self._first_present(techs, ["flutter", "react_native", "android", "ios"]) or "flutter"
            roles.append(
                TeamMember(
                    role="mobile_engineer", seniority=SeniorityEnum.senior, domains=list(domains),
                    tech_tags=[must], nice_to_have=[],
                    rationale=f"Mobile app presence implied; must-have {must}."
                )
            )
        if web:
            fe_must = self._first_present(techs, ["react", "nextjs", "vue", "angular", "svelte"]) or "react"
            roles.append(
                TeamMember(
                    role="frontend_engineer", seniority=SeniorityEnum.senior, domains=list(domains),
                    tech_tags=[fe_must if fe_must != "nextjs" else "react"],
                    nice_to_have=(["nextjs"] if fe_must == "react" or fe_must == "nextjs" else []),
                    rationale=f"Web client implied; must-have {fe_must}."
                )
            )
        if needs_backend:
            musts = [t for t in techs if t in self._BACKEND_TECH_HINTS]
            roles.append(
                TeamMember(
                    role="backend_engineer", seniority=SeniorityEnum.senior, domains=list(domains),
                    tech_tags=musts, nice_to_have=list(self._BACKEND_NICE_DEFAULTS),
                    rationale="APIs/auth/payments implied by flows; backend required."
                )
            )
        if ai:
            roles.append(
                TeamMember(
                    role="ml_engineer", seniority=SeniorityEnum.senior, domains=list(domains),
                    tech_tags=[], nice_to_have=["python", "mlops", "pytorch", "tensorflow"],
                    rationale="AI/assistant/chatbot mentioned; add ML engineering for integration & evals."
                )
            )
        if not roles:
            fallback_role = "frontend_engineer" if self._has_any(techs, self._WEB_TECH_TOKENS) else "backend_engineer"
            roles = [
                TeamMember(
                    role=fallback_role, seniority=SeniorityEnum.senior, domains=list(domains),
                    tech_tags=[self._first_present(techs, list(self._WEB_TECH_TOKENS if fallback_role == "frontend_engineer" else self._BACKEND_TECH_HINTS)) or ""],
                    nice_to_have=[], rationale="Fallback seat due to sparse brief."
                )
            ]

        team = TeamSize(total=len(roles), members=roles)
        expert_roles = list({m.role for m in roles})

        return Criteria(
            domain=crit.domain or [],
            tech_stack=crit.tech_stack or [],
            expert_roles=expert_roles,
            project_type=crit.project_type or "greenfield",
            team_size=team,
        )

    # ---------- Helpers for project search (used by SearchProcessor) ----------

    def _criteria_dict(self, obj: Any) -> Dict[str, Any]:
        """Converts Criteria/dataclass/dict to a plain dict."""
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, Criteria):
            return json.loads(obj.to_json())
        return json.loads(json.dumps(obj))

    def _pack_single_seat_criteria(self, base_criteria: Dict[str, Any], seat: Dict[str, Any]) -> Dict[str, Any]:
        """Creates a new Criteria dict for a single seat."""
        expert = list(dict.fromkeys((base_criteria.get("expert_roles") or []) + [seat["role"]]))
        return {
            "domain": base_criteria.get("domain", []),
            "tech_stack": base_criteria.get("tech_stack", []),
            "expert_roles": expert,
            "project_type": base_criteria.get("project_type", "greenfield"),
            "team_size": {"total": 1, "members": [seat]},
        }