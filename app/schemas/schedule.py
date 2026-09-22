import uuid
from datetime import date, datetime, time
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class FacultyScheduleCreate(BaseModel):
    subject_id: uuid.UUID
    section_id: uuid.UUID
    room_id: Optional[uuid.UUID] = None
    day_of_week: str = Field(..., description="MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY")
    start_time: time
    end_time: time
    academic_year: str = Field(default="2025-2026", max_length=20)
    semester: str = Field(default="ODD", max_length=20)


class FacultyScheduleResponse(BaseModel):
    id: uuid.UUID
    faculty_id: uuid.UUID
    subject_id: uuid.UUID
    section_id: uuid.UUID
    room_id: Optional[uuid.UUID] = None
    day_of_week: str
    start_time: time
    end_time: time
    academic_year: str
    semester: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionGenerateRequest(BaseModel):
    start_date: date
    end_date: date
    faculty_id: Optional[uuid.UUID] = None


class TodayClassResponse(BaseModel):
    session_id: uuid.UUID
    schedule_id: Optional[uuid.UUID] = None
    title: str
    subject_code: str
    subject_name: str
    section_name: str
    room_number: Optional[str] = None
    start_time: str
    end_time: str
    day_of_week: str
    session_date: date
    status: str
    videos_count: int = 0
    latest_video_id: Optional[uuid.UUID] = None
    latest_job_id: Optional[uuid.UUID] = None
    latest_job_status: Optional[str] = None
    has_analysis: bool = False

    model_config = ConfigDict(from_attributes=True)


class TimetableExtractedSlot(BaseModel):
    id: Optional[str] = None
    day_of_week: str = ""  # empty when the day could not be read; faculty completes it
    period_name: Optional[str] = None
    start_time: str = ""  # "09:00" - empty when not read from the document
    end_time: str = ""    # "10:00" - empty when not read from the document
    subject_code: str = ""
    subject_name: str = ""
    section_name: str = ""
    room_number: Optional[str] = ""
    confidence: float = 0.0


class TimetableExtractResponse(BaseModel):
    filename: str
    total_slots_extracted: int
    slots: List[TimetableExtractedSlot]
    message: str


class TimetableConfirmSlot(BaseModel):
    day_of_week: str
    start_time: str  # "09:00" or "09:00:00"
    end_time: str    # "10:00" or "10:00:00"
    subject_code: str
    subject_name: str = ""
    section_name: str = ""
    room_number: Optional[str] = None
    academic_year: Optional[str] = "2025-2026"
    semester: Optional[str] = "EVEN"


class TimetableConfirmRequest(BaseModel):
    slots: List[TimetableConfirmSlot]
    clear_existing: bool = True


class TimetableConfirmResponse(BaseModel):
    saved_slots_count: int
    created_subjects_count: int
    created_sections_count: int
    message: str

