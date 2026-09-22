from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, EmailStr, field_validator
from app.core.config import settings


class FacultyRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    department: str
    designation: Optional[str] = None

    @field_validator("email")
    @classmethod
    def validate_klu_email(cls, v: str) -> str:
        v = str(v).strip().lower()
        required_domain = settings.ALLOWED_EMAIL_DOMAIN.lower()
        if not v.endswith(f"@{required_domain}") and not v.endswith(f".{required_domain}"):
            raise ValueError(
                f"Registration is restricted to official university emails ending in '@{required_domain}'"
            )
        return v

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v

    @field_validator("full_name", "department")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Field cannot be empty or blank")
        return v


class FacultyLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def validate_klu_email(cls, v: str) -> str:
        v = str(v).strip().lower()
        required_domain = settings.ALLOWED_EMAIL_DOMAIN.lower()
        if not v.endswith(f"@{required_domain}") and not v.endswith(f".{required_domain}"):
            raise ValueError(
                f"Login is restricted to official university emails ending in '@{required_domain}'"
            )
        return v


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    type: Optional[str] = None
    exp: Optional[int] = None


class FacultyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    full_name: str
    department: str
    designation: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class FacultyUpdate(BaseModel):
    full_name: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None
