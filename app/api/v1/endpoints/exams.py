from typing import Any, Dict, List, Optional
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.exam import (
    ExamEventReviewRequest,
    ExamReviewEventResponse,
    ExamSessionCreate,
    ExamSessionResponse,
    FloorRoomHierarchyItem,
    InvigilatorDashboardData,
)
from app.services.exam_service import exam_service

router = APIRouter(prefix="/exams", tags=["Exam Mode"])


@router.get("/floors", response_model=List[FloorRoomHierarchyItem])
def get_floors_and_rooms(
    db: Session = Depends(get_db),
) -> List[FloorRoomHierarchyItem]:
    """
    Returns the hierarchy: Floor → Room → Registered Camera.
    Seeds default demo rooms if empty.
    """
    return exam_service.get_floor_room_hierarchy(db)


@router.post("/sessions", response_model=ExamSessionResponse, status_code=status.HTTP_201_CREATED)
def create_exam_session(
    data: ExamSessionCreate,
    db: Session = Depends(get_db),
) -> ExamSessionResponse:
    """
    Creates an Exam Session bound to a specific Room and Registered Camera.
    Independent of faculty timetable data.
    """
    try:
        session = exam_service.create_exam_session(db, data)
        return exam_service._format_session_response(session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/sessions", response_model=List[ExamSessionResponse])
def list_exam_sessions(
    room_id: Optional[uuid.UUID] = Query(None, description="Filter by room ID"),
    db: Session = Depends(get_db),
) -> List[ExamSessionResponse]:
    """
    Lists all scheduled, active, and completed exam sessions.
    """
    return exam_service.list_exam_sessions(db, room_id=room_id)


@router.get("/sessions/{session_id}", response_model=ExamSessionResponse)
def get_exam_session(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> ExamSessionResponse:
    """
    Retrieves metadata for a specific exam session.
    """
    try:
        session = exam_service.get_session(db, session_id)
        return exam_service._format_session_response(session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/sessions/{session_id}/monitor", response_model=ExamSessionResponse)
def start_exam_monitoring(
    session_id: uuid.UUID,
    max_duration_seconds: Optional[float] = Query(
        None, description="Optional cap on video ingestion seconds for quick testing/demo"
    ),
    db: Session = Depends(get_db),
) -> ExamSessionResponse:
    """
    Runs the Edge Processing + Fog Room Processing pipeline on the room's registered camera.
    Generates strictly observable temporal review events for human invigilator evaluation.
    """
    try:
        session = exam_service.start_exam_monitoring(
            db=db,
            session_id=session_id,
            max_duration_seconds=max_duration_seconds,
        )
        return exam_service._format_session_response(session)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/sessions/{session_id}/dashboard", response_model=InvigilatorDashboardData)
def get_invigilator_dashboard(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> InvigilatorDashboardData:
    """
    Returns full payload for the Invigilator Dashboard:
    Room status, camera health, coverage/uncertainty metrics, and observable review events.
    """
    try:
        return exam_service.get_invigilator_dashboard_data(db, session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/sessions/{session_id}/events", response_model=List[ExamReviewEventResponse])
def get_session_review_events(
    session_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> List[ExamReviewEventResponse]:
    """
    Lists all observable review events for an exam session.
    """
    try:
        dashboard = exam_service.get_invigilator_dashboard_data(db, session_id)
        return dashboard.review_events
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.patch("/events/{event_id}/review", response_model=ExamReviewEventResponse)
def review_exam_event(
    event_id: uuid.UUID,
    req: ExamEventReviewRequest,
    db: Session = Depends(get_db),
) -> ExamReviewEventResponse:
    """
    Human-in-the-loop review action:
    Invigilator marks event as CONFIRMED_OBSERVATION, DISMISSED, or ACTION_TAKEN.
    """
    try:
        event = exam_service.review_event(db, event_id, req)
        return ExamReviewEventResponse.model_validate(event)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
