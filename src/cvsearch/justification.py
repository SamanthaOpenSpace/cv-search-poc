# src/cvsearch/justification.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List

class CandidateJustification(BaseModel):
    """
    An LLM-generated justification for a candidate's match to a role,
    including a measurable score.
    """

    match_summary: str = Field(
        ...,
        description="A 1-2 sentence executive summary of the candidate's fit for the role."
    )

    strength_analysis: List[str] = Field(
        ...,
        description="A list of 2-4 bullet points detailing specific strengths. Must cite evidence from the CV (e.g., 'Strong .NET experience at Company X')."
    )

    gap_analysis: List[str] = Field(
        ...,
        description="A list of 1-3 bullet points detailing missing skills or gaps (e.g., 'No direct experience with Kafka, which was a must-have')."
    )

    overall_match_score: float = Field(
        ...,
        description="A float score from 0.0 (No Match) to 1.0 (Perfect Match) representing the candidate's overall fit for the role.",
        ge=0.0,
        le=1.0
    )