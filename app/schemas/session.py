from datetime import date, time, datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, field_validator


class ClassSessionBase(BaseModel):
    title: str
    session_date: date
    start_time: time
    end_time: time
    room_number: Optional[str] = None
    room_id: Optional[UUID] = None
    day_of_week: Optional[str] = None
    status: str = "SCHEDULED"

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Title cannot be empty")
        return v


class ClassSessionCreate(ClassSessionBase):
    section_id: UUID


class ClassSessionUpdate(BaseModel):
    title: Optional[str] = None
    session_date: Optional[date] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    room_number: Optional[str] = None
    room_id: Optional[UUID] = None
    day_of_week: Optional[str] = None
    status: Optional[str] = None


class ClassSessionResponse(ClassSessionBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    faculty_id: UUID
    section_id: UUID
    created_at: datetime
    updated_at: datetime


class ClassSessionDetailResponse(ClassSessionResponse):
    section_name: Optional[str] = None
    subject_name: Optional[str] = None
    subject_code: Optional[str] = None
    videos_count: int = 0
