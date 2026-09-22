from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, EmailStr


class FacultyBase(BaseModel):
    email: EmailStr
    full_name: str
    department: str
    designation: Optional[str] = None


class FacultyDetail(FacultyBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
