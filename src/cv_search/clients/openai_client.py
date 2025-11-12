from __future__ import annotations

import json
from typing import Any, Dict, List, Type

from openai import AzureOpenAI, OpenAI

from cv_search.config.settings import Settings
from cv_search.lexicon.loader import (
    load_domain_lexicon,
    load_expertise_lexicon,
    load_role_lexicon,
    load_tech_synonyms,
)
from cv_search.llm.schemas import CandidateJustification
from cv_search.llm.logger import log_chat

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field


class LLMCriteria(BaseModel):
    domain: List[str]
    tech_stack: List[str]
    expert_roles: List[str]
    project_type: str | None = None
    team_size: Dict[str, Any] | None = None


class LLMCV(BaseModel):
    name: str | None = None
    location: str | None = None
    seniority: str
    role_tags: List[str]
    expertise_tags: List[str] = Field(default_factory=list)
    summary: str | None = None
    experience: List[Dict[str, Any]]
    tech_tags: List[str]
    unmapped_tags: str | None = None
    source_folder_role_hint: str | None = None


class OpenAIClient:
    """Adapter around the OpenAI / Azure OpenAI SDKs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.use_azure_openai:
            if not all([settings.azure_endpoint, settings.azure_api_version, settings.openai_api_key_str]):
                raise ValueError("Azure settings (endpoint, version, key) are not fully configured.")
            self.client = AzureOpenAI(
                api_key=settings.openai_api_key_str,
                api_version=settings.azure_api_version,
                azure_endpoint=settings.azure_endpoint,
            )
        else:
            if not settings.openai_api_key_str:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=settings.openai_api_key_str)

    def _schema_json(self, model_cls: Type[BaseModel]) -> str:
        if hasattr(model_cls, "schema_json"):
            return model_cls.schema_json(indent=2)  # pydantic v1
        if hasattr(model_cls, "model_json_schema"):
            return json.dumps(model_cls.model_json_schema(), indent=2)  # pydantic v2
        return "{}"

    def _get_structured_response(
            self,
            prompt: str,
            system_prompt: str,
            model: str,
            pydantic_model: Type[BaseModel],
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    f"{system_prompt}\n\n"
                    "You must respond with JSON matching the following Pydantic schema:\n"
                    f"{self._schema_json(pydantic_model)}"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        usage_obj = getattr(response, "usage", None)
        usage = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else None
        provider = "azure_openai" if self.settings.use_azure_openai else "openai"
        log_chat(
            messages=messages,
            model=model,
            response_content=content,
            provider=provider,
            usage=usage,
            meta={"pydantic_model": getattr(pydantic_model, "__name__", "Unknown")},
        )
        return json.loads(content)

    def get_structured_criteria(self, text: str, model: str, settings: Settings) -> Dict[str, Any]:
        role_lex_list = load_role_lexicon(settings.lexicon_dir)
        tech_lex_list = load_tech_synonyms(settings.lexicon_dir)
        domain_lex_list = load_domain_lexicon(settings.lexicon_dir)

        system_prompt = f"""
        You are a TA (Talent Acquisition) expert. Your task is to parse a user's free-text project brief
        into a structured JSON object.

        Available Role Lexicon: {json.dumps(role_lex_list, indent=2)}
        Available Tech Lexicon: {json.dumps(tech_lex_list, indent=2)}
        Available Domain Lexicon: {json.dumps(domain_lex_list, indent=2)}

        Strictly follow these rules:
        1.  Map all found skills, roles, and domains to their *canonical* keys from the lexicons provided.
        2.  If a `team_size` or specific roles (e.g., "2 .net devs") are mentioned, populate the `team_size.members` list.
        3.  `tech_stack` should be a rollup of *all* technologies mentioned in the brief.
        """
        return self._get_structured_response(
            prompt=text,
            system_prompt=system_prompt,
            model=model,
            pydantic_model=LLMCriteria,
        )

    def get_candidate_justification(self, seat_details: str, cv_context: str) -> Dict[str, Any]:
        system_prompt = """
        You are an expert Talent Acquisition manager. Your task is to evaluate a candidate's CV
        against a specific role's requirements.

        You will be given:
        1.  [ROLE]: A JSON object describing the role's requirements (role, seniority, must-have tech, nice-to-have tech, domains).
        2.  [CV]: The candidate's CV context (summary, experience, and tags).

        Your task is to provide a structured JSON justification.
        - `match_summary`: A 1-2 sentence executive summary.
        - `strength_analysis`: Bullet points of specific strengths, citing CV evidence.
        - `gap_analysis`: Bullet points of missing skills or gaps.
        - `overall_match_score`: A float from 0.0 to 1.0.
        """

        prompt = f"[ROLE]\n{seat_details}\n\n[CV]\n{cv_context}"

        return self._get_structured_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model=self.settings.openai_model,
            pydantic_model=CandidateJustification,
        )

    def get_structured_cv(
            self,
            raw_text: str,
            role_folder_hint: str,
            model: str,
            settings: Settings,
    ) -> Dict[str, Any]:
        role_lex_list = load_role_lexicon(settings.lexicon_dir)
        tech_lex_list = load_tech_synonyms(settings.lexicon_dir)
        domain_lex_list = load_domain_lexicon(settings.lexicon_dir)
        expertise_lex_list = load_expertise_lexicon(settings.lexicon_dir)

        system_prompt = f"""
        You are an expert CV parser. Your task is to parse raw text from a CV slide deck
        into a structured JSON object.

        You are given a HINT: the normalized parent folder for this CV is '{role_folder_hint}'.

        Available Role Lexicon (Canonical Keys): {json.dumps(role_lex_list, indent=2)}
        Available Tech Lexicon (Canonical Keys): {json.dumps(tech_lex_list, indent=2)}
        Available Domain Lexicon (Canonical Keys): {json.dumps(domain_lex_list, indent=2)}
        Available Role-to-Expertise Lexicon: {json.dumps(expertise_lex_list, indent=2)}

        Strictly follow these rules:
        1.  `source_folder_role_hint`:
            * Analyze the HINT: `{role_folder_hint}`.
            * Determine if this hint represents a valid professional role.
            * If it **is** a valid role, find the **best matching canonical key** from the Role Lexicon.
            * If the HINT is **not** a valid role, you **MUST** set this field to `null`.

        2.  `role_tags`: Extract all relevant roles from the CV *text itself*, mapping them to the Role Lexicon keys.
        3.  `expertise_tags`: Infer direction/expertise areas for the candidate. Use the Role-to-Expertise lexicon to
            choose canonical expertise keys aligned with the detected roles whenever possible.
        4.  `tech_tags`: Extract all technologies from the CV *text*, mapping them to the Tech Lexicon keys.
        5.  `unmapped_tags`: List any tech/tools found but *not* in the lexicons as a comma-separated string.
        6.  `experience.domain_tags` / `experience.tech_tags`: Map these to the lexicons as well.
        """

        return self._get_structured_response(
            prompt=raw_text,
            system_prompt=system_prompt,
            model=model,
            pydantic_model=LLMCV,
        )
