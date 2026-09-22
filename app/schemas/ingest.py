import uuid
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NVRWebhookPayload(BaseModel):
    event_type: str = Field(default="RECORDING_COMPLETED")
    device_code: Optional[str] = None  # e.g. CAM-301-FRONT
    room_number: Optional[str] = None  # e.g. 301 or Room 301
    recording_date: Optional[date] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    file_path: Optional[str] = None  # Local path or mounted NFS path
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = 0
    duration_seconds: Optional[float] = None
    signature: Optional[str] = None
    metadata: Dict[str, Any] = {}


class LMSDropPayload(BaseModel):
    course_code: Optional[str] = None
    section_name: Optional[str] = None
    recording_url: Optional[str] = None
    file_path: Optional[str] = None
    faculty_email: Optional[str] = None
    session_date: Optional[date] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    metadata: Dict[str, Any] = {}


class IngestJobResponse(BaseModel):
    success: bool
    message: str
    video_id: Optional[uuid.UUID] = None
    session_id: Optional[uuid.UUID] = None
    job_id: Optional[uuid.UUID] = None
    match_status: str  # EXACT_MATCH, APPROXIMATE_MATCH, UNASSIGNED, AUTO_DISPATCHED
    details: Dict[str, Any] = {}


class DropzoneScanResult(BaseModel):
    files_scanned: int
    matched_and_ingested: int
    unmatched_files: List[str]
    errors: List[str]
