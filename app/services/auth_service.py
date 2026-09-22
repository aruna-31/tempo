import uuid
from typing import Tuple
from fastapi import HTTPException, status
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import (
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)
from app.models.faculty import Faculty
from app.schemas.auth import (
    FacultyLogin,
    FacultyRegister,
    FacultyUpdate,
    Token,
)


class AuthService:
    @staticmethod
    def register_faculty(db: Session, data: FacultyRegister) -> Faculty:
        """Register a new faculty member with @klu.ac.in email validation."""
        email = data.email.strip().lower()
        
        # Check if email already exists
        existing = db.query(Faculty).filter(Faculty.email == email).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A faculty account with this @klu.ac.in email already exists"
            )

        new_faculty = Faculty(
            email=email,
            hashed_password=get_password_hash(data.password),
            full_name=data.full_name.strip(),
            department=data.department.strip(),
            designation=data.designation.strip() if data.designation else None,
            is_active=True
        )
        db.add(new_faculty)
        db.commit()
        db.refresh(new_faculty)
        return new_faculty

    @staticmethod
    def authenticate_faculty(db: Session, data: FacultyLogin) -> Tuple[Faculty, Token]:
        """Authenticate faculty and return access + refresh tokens."""
        email = data.email.strip().lower()
        faculty = db.query(Faculty).filter(Faculty.email == email).first()

        if not faculty or not verify_password(data.password, faculty.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect @klu.ac.in email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not faculty.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Faculty account is inactive. Please contact your administrator."
            )

        access_token = create_access_token(
            subject=str(faculty.id),
            extra_claims={"email": faculty.email, "name": faculty.full_name}
        )
        refresh_token = create_refresh_token(
            subject=str(faculty.id)
        )

        tokens = Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        return faculty, tokens

    @staticmethod
    def refresh_token(db: Session, refresh_token_str: str) -> Token:
        """Issue new access token from valid refresh token."""
        try:
            payload = jwt.decode(
                refresh_token_str,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            faculty_id_str = payload.get("sub")
            token_type = payload.get("type")

            if not faculty_id_str or token_type != TOKEN_TYPE_REFRESH:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token format or type"
                )

            faculty_id = uuid.UUID(faculty_id_str)
        except (JWTError, ValueError):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Expired or invalid refresh token"
            )

        faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
        if not faculty or not faculty.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Faculty account not found or inactive"
            )

        new_access_token = create_access_token(
            subject=str(faculty.id),
            extra_claims={"email": faculty.email, "name": faculty.full_name}
        )
        # Optional: rotate refresh token
        new_refresh_token = create_refresh_token(subject=str(faculty.id))

        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    @staticmethod
    def update_profile(db: Session, faculty: Faculty, data: FacultyUpdate) -> Faculty:
        """Update faculty profile information and/or password."""
        if data.full_name is not None:
            faculty.full_name = data.full_name.strip()
        if data.department is not None:
            faculty.department = data.department.strip()
        if data.designation is not None:
            faculty.designation = data.designation.strip()

        if data.new_password:
            if not data.current_password:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password is required to set a new password"
                )
            if not verify_password(data.current_password, faculty.hashed_password):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current password does not match"
                )
            if len(data.new_password) < 8:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="New password must be at least 8 characters long"
                )
            faculty.hashed_password = get_password_hash(data.new_password)

        db.commit()
        db.refresh(faculty)
        return faculty
