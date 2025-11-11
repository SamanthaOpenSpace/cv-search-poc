from __future__ import annotations

from typing import List

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:
    ConfigDict = None


class CandidateJustification(BaseModel):
    """Structured justification output returned by the LLM."""

    match_summary: str = Field(description="One to two sentence executive summary of the candidate fit.")
    strength_analysis: List[str] = Field(default_factory=list, description="Bulleted list of candidate strengths.")
    gap_analysis: List[str] = Field(default_factory=list, description="Bulleted list of gaps or missing skills.")
    overall_match_score: float = Field(description="Overall match score from 0.0 to 1.0.")

    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:
        class Config:
            extra = "allow"
