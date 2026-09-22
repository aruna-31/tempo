import uuid
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.security import TOKEN_TYPE_ACCESS
from app.db.session import get_db
from app.models.faculty import Faculty

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False
)


def get_current_faculty(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Faculty:
    """
    FastAPI dependency that decodes JWT access token and fetches authenticated faculty.
    Strictly verifies token validity, token type, and faculty active status.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing. Please log in with your @klu.ac.in credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        faculty_id_str: Optional[str] = payload.get("sub")
        token_type: Optional[str] = payload.get("type")

        if not faculty_id_str or token_type != TOKEN_TYPE_ACCESS:
            raise credentials_exception

        faculty_id = uuid.UUID(faculty_id_str)
    except (JWTError, ValueError):
        raise credentials_exception

    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Faculty account not found"
        )

    if not faculty.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Faculty account is deactivated. Contact university administrator."
        )

    return faculty


def get_current_active_faculty(
    current_faculty: Faculty = Depends(get_current_faculty),
) -> Faculty:
    """Alias for active faculty verification."""
    return current_faculty


def require_admin(
    current_faculty: Faculty = Depends(get_current_active_faculty),
) -> Faculty:
    """Requires ADMIN, DEAN, or superuser role."""
    if current_faculty.role not in ("ADMIN", "DEAN") and not current_faculty.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required for this action."
        )
    return current_faculty


def verify_session_ownership(
    session_id: uuid.UUID,
    faculty: Faculty,
    db: Session
):
    """Verifies that the faculty owns the session or is an admin."""
    from app.models.session import ClassSession
    session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="ClassSession not found")
    if session.faculty_id != faculty.id and faculty.role not in ("ADMIN", "DEAN") and not faculty.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You are not authorized to view or modify this session."
        )
    return session


def verify_video_ownership(
    video_id: uuid.UUID,
    faculty: Faculty,
    db: Session
):
    """Verifies that the faculty owns the video or is an admin."""
    from app.models.video import Video
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    if video.faculty_id != faculty.id and faculty.role not in ("ADMIN", "DEAN") and not faculty.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. You are not authorized to access this video recording."
        )
    return video
