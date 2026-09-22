from datetime import date
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.session import (
    ClassSessionCreate,
    ClassSessionResponse,
    ClassSessionUpdate,
)
from app.services.session_service import SessionService

router = APIRouter(prefix="/sessions", tags=["Class Sessions"])


@router.post(
    "",
    response_model=ClassSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule a class session"
)
def create_session(
    data: ClassSessionCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.create_session(
        db=db, faculty_id=current_faculty.id, data=data
    )


@router.post(
    "/{session_id}/videos",
    status_code=status.HTTP_201_CREATED,
    summary="Upload classroom video for a session"
)
def upload_session_video(
    session_id: UUID,
    file: UploadFile = File(...),
    duration_seconds: Optional[float] = Form(None),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    from app.schemas.video import VideoResponse, VideoUploadResponse
    video, job = SessionService.upload_and_create_video(
        db=db,
        faculty_id=current_faculty.id,
        session_id=session_id,
        file=file,
        duration_seconds=duration_seconds
    )
    return VideoUploadResponse(
        video=VideoResponse.model_validate(video),
        analysis_job=job,
        message="Video uploaded successfully and analysis job queued"
    )


@router.get(
    "",
    response_model=List[ClassSessionResponse],
    status_code=status.HTTP_200_OK,
    summary="List faculty's class sessions"
)
def list_sessions(
    section_id: Optional[UUID] = Query(None, description="Filter by section ID"),
    session_date: Optional[date] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.get_faculty_sessions(
        db=db,
        faculty_id=current_faculty.id,
        section_id=section_id,
        session_date=session_date,
        skip=skip,
        limit=limit
    )


@router.get(
    "/{session_id}",
    response_model=ClassSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get class session by ID"
)
def get_session(
    session_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.get_session_by_id(
        db=db, faculty_id=current_faculty.id, session_id=session_id
    )


@router.put(
    "/{session_id}",
    response_model=ClassSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update class session"
)
def update_session(
    session_id: UUID,
    data: ClassSessionUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.update_session(
        db=db, faculty_id=current_faculty.id, session_id=session_id, data=data
    )


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete class session"
)
def delete_session(
    session_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    SessionService.delete_session(
        db=db, faculty_id=current_faculty.id, session_id=session_id
    )
