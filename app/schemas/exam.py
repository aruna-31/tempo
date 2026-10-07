from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, ConfigDict


class CameraSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_name: str
    device_code: str
    source_type: str = "MP4_DEMO"
    location_in_room: str = "FRONT"
    status: str = "ACTIVE"
    is_healthy: bool = True
    demo_video_path: Optional[str] = None


class RoomCameraItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    room_number: str
    building: str
    floor: int
    capacity: int
    cameras: List[CameraSummary] = []


class FloorRoomHierarchyItem(BaseModel):
    floor: int
    building: str
    rooms: List[RoomCameraItem] = []


class ExamSessionCreate(BaseModel):
    title: str = Field(..., max_length=150)
    session_code: Optional[str] = Field(None, max_length=50)
    room_id: uuid.UUID
    camera_id: Optional[uuid.UUID] = None
    expected_students: int = Field(70, ge=1, le=500)
    invigilator_name: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None


class ExamReviewEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    exam_session_id: uuid.UUID
    anonymous_student_id: str
    track_id: Optional[int] = None
    event_type: str
    event_description: Optional[str] = None
    confidence: float
    start_time_offset: float
    end_time_offset: float
    duration_seconds: float
    observation_coverage: float
    observation_uncertainty: float
    spatial_zone: str = "DESK_AREA"
    bounding_box: Optional[List[float]] = None
    evidence_window_start: Optional[float] = None
    evidence_window_end: Optional[float] = None
    evidence_clip_path: Optional[str] = None
    supporting_observations: Optional[Dict[str, Any]] = None
    review_status: str = "PENDING_REVIEW"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    created_at: datetime


class ExamSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_code: str
    title: str
    room_id: uuid.UUID
    room_number: Optional[str] = None
    floor: Optional[int] = None
    building: Optional[str] = None
    camera_id: Optional[uuid.UUID] = None
    camera_device_code: Optional[str] = None
    camera_source_type: Optional[str] = None
    status: str
    room_monitoring_status: str
    expected_students: int
    detected_students_count: int
    coverage_ratio: float
    uncertainty_score: float
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    invigilator_name: Optional[str] = None
    notes: Optional[str] = None
    total_events: int = 0
    pending_events: int = 0
    created_at: datetime


class ExamEventReviewRequest(BaseModel):
    review_status: str = Field(..., description="CONFIRMED_OBSERVATION, DISMISSED, ACTION_TAKEN")
    reviewed_by: str = Field(..., max_length=100)
    review_notes: Optional[str] = None


class InvigilatorDashboardData(BaseModel):
    session: ExamSessionResponse
    camera_health: Dict[str, Any]
    monitoring_metrics: Dict[str, Any]
    review_events: List[ExamReviewEventResponse]
    summary_counts: Dict[str, int]
