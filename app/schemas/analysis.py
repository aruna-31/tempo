from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObservableBehaviourType(str, Enum):
    Looking_Toward_Instruction = "Looking_Toward_Instruction"
    Reading = "Reading"
    Writing = "Writing"
    Peer_Interaction = "Peer_Interaction"
    Looking_Away = "Looking_Away"


# Mapping dictionary for indexing
BEHAVIOUR_INDEX_MAP = {
    0: ObservableBehaviourType.Looking_Toward_Instruction,
    1: ObservableBehaviourType.Reading,
    2: ObservableBehaviourType.Writing,
    3: ObservableBehaviourType.Peer_Interaction,
    4: ObservableBehaviourType.Looking_Away,
}

BEHAVIOUR_NAME_MAP = {
    ObservableBehaviourType.Looking_Toward_Instruction.value: 0,
    ObservableBehaviourType.Reading.value: 1,
    ObservableBehaviourType.Writing.value: 2,
    ObservableBehaviourType.Peer_Interaction.value: 3,
    ObservableBehaviourType.Looking_Away.value: 4,
}


# --- Analysis Job Schemas ---
class AnalysisJobCreate(BaseModel):
    video_id: UUID
    config_json: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AnalysisJobStatusUpdate(BaseModel):
    status: str  # PENDING, QUEUED, PROCESSING, COMPLETED, FAILED
    progress_pct: Optional[int] = None
    error_message: Optional[str] = None


class AnalysisJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    video_id: UUID
    status: str
    current_stage: str = "QUEUED"
    progress_pct: int = 0
    retry_count: int = 0
    max_retries: int = 3
    source_type: str = "MANUAL_UPLOAD"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    processing_logs: List[Dict[str, Any]] = []
    config_json: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime


# --- Student Tracking Schemas (Temporary Per-Session Track ID) ---
class StudentTrackResultCreate(BaseModel):
    job_id: UUID
    track_id: int = Field(..., description="Temporary per-session numeric Track ID")
    student_id: Optional[UUID] = Field(None, description="Optional link if manually associated")
    bounding_box_history: List[Dict[str, Any]]
    start_frame: int
    end_frame: int


class StudentTrackResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: UUID
    track_id: int
    student_id: Optional[UUID] = None
    bounding_box_history: List[Dict[str, Any]]
    start_frame: int
    end_frame: int
    created_at: datetime


# --- Behaviour Result Schemas (Strictly 5 Observable Classes) ---
class BehaviourResultCreate(BaseModel):
    job_id: UUID
    track_id: int = Field(..., description="Temporary per-session student Track ID")
    frame_number: int
    timestamp_seconds: float
    behaviour_type: ObservableBehaviourType
    confidence: float = Field(..., ge=0.0, le=1.0)
    student_id: Optional[UUID] = None
    metadata_json: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("behaviour_type", mode="before")
    @classmethod
    def validate_behaviour_type(cls, v: Any) -> ObservableBehaviourType:
        if isinstance(v, str):
            # Normalize casing / underscores if needed
            v_clean = v.strip()
            # Match case-insensitively
            for enum_val in ObservableBehaviourType:
                if v_clean.lower() == enum_val.value.lower():
                    return enum_val
            allowed = [e.value for e in ObservableBehaviourType]
            raise ValueError(
                f"Invalid behaviour type '{v}'. Allowed observable behaviours: {', '.join(allowed)}"
            )
        return v


class BehaviourResultBulkCreate(BaseModel):
    job_id: UUID
    results: List[BehaviourResultCreate]


class BehaviourResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: UUID
    track_id: int
    frame_number: int
    timestamp_seconds: float
    behaviour_type: ObservableBehaviourType
    confidence: float
    student_id: Optional[UUID] = None
    metadata_json: Dict[str, Any]
    created_at: datetime


# --- Aggregated Analytics Summary Schemas ---
class TemporalBehaviourBucket(BaseModel):
    timestamp_start: float
    timestamp_end: float
    looking_toward_instruction_count: int = 0
    reading_count: int = 0
    writing_count: int = 0
    peer_interaction_count: int = 0
    looking_away_count: int = 0
    total_detections: int = 0
    engagement_index: float = 0.0  # 0.0 to 100.0% (Instruction + Reading + Writing)


class StudentTrackMetrics(BaseModel):
    track_id: int
    student_id: Optional[UUID] = None
    display_label: str = "Student Track"
    looking_toward_instruction_pct: float
    reading_pct: float
    writing_pct: float
    peer_interaction_pct: float
    looking_away_pct: float
    dominant_behaviour: str


class SessionAnalyticsSummary(BaseModel):
    job_id: UUID
    video_id: UUID
    session_id: UUID
    status: str
    total_frames_analyzed: int
    total_tracks_identified: int
    overall_engagement_score: float  # e.g., 82.5%
    timeline_buckets: List[TemporalBehaviourBucket] = []
    track_metrics: List[StudentTrackMetrics] = []
