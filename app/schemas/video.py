from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict
from app.schemas.analysis import AnalysisJobResponse


class VideoBase(BaseModel):
    file_path: str
    original_filename: str
    file_size_bytes: int
    duration_seconds: Optional[float] = None
    content_type: str = "video/mp4"
    output_video_path: Optional[str] = None


class VideoCreate(BaseModel):
    session_id: UUID
    original_filename: str
    file_size_bytes: int
    file_path: str
    content_type: str = "video/mp4"
    duration_seconds: Optional[float] = None
    output_video_path: Optional[str] = None


class VideoStatusUpdate(BaseModel):
    status: str
    duration_seconds: Optional[float] = None
    output_video_path: Optional[str] = None


class VideoResponse(VideoBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    session_id: UUID
    faculty_id: UUID
    status: str
    created_at: datetime
    updated_at: datetime


class VideoUploadResponse(BaseModel):
    video: VideoResponse
    analysis_job: AnalysisJobResponse
    message: str = "Video uploaded and analysis job initiated successfully"


class VideoStatusDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    video_id: UUID
    session_id: UUID
    original_filename: str
    file_size_bytes: int
    duration_seconds: Optional[float] = None
    video_status: str
    content_type: str
    latest_analysis_job_id: Optional[UUID] = None
    analysis_status: Optional[str] = None
    analysis_progress_pct: int = 0
    error_message: Optional[str] = None
    stream_url: str
    output_stream_url: Optional[str] = None
    is_output_ready: bool = False
