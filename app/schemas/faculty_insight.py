from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class FacultyInsightBase(BaseModel):
    category: str = Field(..., description="Insight category: PACING, VARIETY, INTERACTION, ATTENTION_PATTERNS, COVERAGE")
    start_time: Optional[float] = Field(None, description="Start timestamp of observation window in seconds")
    end_time: Optional[float] = Field(None, description="End timestamp of observation window in seconds")
    observation: str = Field(..., description="Objective observable class-level behavior pattern")
    pedagogical_context: str = Field(..., description="Learning science context grounding the observation")
    suggested_action: str = Field(..., description="Constructive, actionable suggestion for FUTURE classes")
    coverage_context: Optional[str] = Field(None, description="Technical coverage and tracking sample context")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Observational confidence score")


class FacultyInsightCreate(FacultyInsightBase):
    job_id: UUID


class FacultyInsightResponse(FacultyInsightBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: UUID
    created_at: datetime
