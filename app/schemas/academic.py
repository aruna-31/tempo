from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, EmailStr, field_validator


# --- Subject Schemas ---
class SubjectBase(BaseModel):
    code: str
    name: str
    description: Optional[str] = None

    @field_validator("code", "name")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


class SubjectCreate(SubjectBase):
    pass


class SubjectUpdate(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class SubjectResponse(SubjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    faculty_id: UUID
    created_at: datetime
    updated_at: datetime


# --- Section Schemas ---
class SectionBase(BaseModel):
    name: str
    academic_year: str
    semester: str

    @field_validator("name", "academic_year", "semester")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


class SectionCreate(SectionBase):
    subject_id: UUID


class SectionUpdate(BaseModel):
    name: Optional[str] = None
    academic_year: Optional[str] = None
    semester: Optional[str] = None


class SectionResponse(SectionBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    subject_id: UUID
    created_at: datetime
    updated_at: datetime


# --- Student Schemas ---
class StudentBase(BaseModel):
    roll_number: str
    name: str
    email: Optional[EmailStr] = None

    @field_validator("roll_number", "name")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty")
        return v


class StudentCreate(StudentBase):
    section_id: UUID


class StudentBulkItem(BaseModel):
    roll_number: str
    name: str
    email: Optional[EmailStr] = None


class StudentBulkCreate(BaseModel):
    section_id: UUID
    students: List[StudentBulkItem]


class StudentUpdate(BaseModel):
    roll_number: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class StudentResponse(StudentBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    section_id: UUID
    created_at: datetime
    updated_at: datetime


# --- Detailed Nested Schemas ---
class SectionDetailResponse(SectionResponse):
    students_count: int = 0
    students: List[StudentResponse] = []


class SubjectDetailResponse(SubjectResponse):
    sections: List[SectionResponse] = []
