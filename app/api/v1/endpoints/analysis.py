import json
import os
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.analysis import (
    AnalysisJobCreate,
    AnalysisJobResponse,
    AnalysisJobStatusUpdate,
    BehaviourResultBulkCreate,
    BehaviourResultCreate,
    BehaviourResultResponse,
    SessionAnalyticsSummary,
    StudentTrackResultCreate,
    StudentTrackResultResponse,
)
from app.schemas.faculty_insight import FacultyInsightResponse
from app.services.analysis_service import AnalysisService
from app.services.faculty_insight_service import FacultyInsightService
from app.services.ml_connector import get_ml_connector
from app.services.temporal_analytics_service import TemporalAnalyticsService

router = APIRouter(prefix="/analysis", tags=["Temporal Behaviour Analysis"])


@router.get(
    "/model-info",
    summary="Get the active temporal model metadata"
)
def get_model_info(
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """Return the metadata used by the active production checkpoint."""
    metadata_path = os.path.join(
        os.path.dirname(os.path.abspath(settings.MODEL_WEIGHTS_PATH)),
        "model_metadata.json"
    )
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    return {
        "model_version": metadata["model_version"],
        "spatial_backbone": metadata["spatial_backbone"],
        "temporal_model_type": metadata["temporal_model_type"],
        "hidden_dim": metadata["hidden_dim"],
        "num_layers": metadata["num_layers"],
        "sequence_length": metadata["sequence_length"],
        "sampling_fps": metadata["sampling_fps"],
    }


# ------------------ JOBS ------------------
@router.post(
    "/jobs",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new analysis job for a video and start background inference",
    description="Provisions an AnalysisJob in PostgreSQL and dispatches the video to the ML background worker."
)
def create_analysis_job(
    data: AnalysisJobCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    job = AnalysisService.create_job(db=db, faculty_id=current_faculty.id, data=data)
    # Dispatch background ML inference
    ml = get_ml_connector()
    ml.submit_job(
        job_id=job.id,
        video_id=job.video_id,
        raw_video_path="",
        config=job.config_json
    )
    return job


@router.get(
    "/jobs/{job_id}",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Get analysis job status and metadata"
)
def get_analysis_job(
    job_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.get_job_by_id(db=db, faculty_id=current_faculty.id, job_id=job_id)


@router.post(
    "/jobs/{job_id}/retry",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Retry a failed analysis job"
)
def retry_analysis_job(
    job_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    from app.services.job_manager import JobManagerService
    from app.services.audit_service import AuditService
    from app.models.analysis import AnalysisJob

    job = AnalysisService.get_job_by_id(db=db, faculty_id=current_faculty.id, job_id=job_id)
    job.status = "QUEUED"
    job.current_stage = "QUEUED"
    job.progress_pct = 0
    job.retry_count = 0
    job.error_message = None
    db.commit()
    db.refresh(job)

    if settings.AUTO_DISPATCH_ML_JOB:
        JobManagerService.dispatch_background_job(job.id)
    AuditService.log_event(
        db=db,
        action="RETRY_JOB",
        resource_type="JOB",
        resource_id=str(job.id),
        user=current_faculty,
        details={"job_id": str(job.id)}
    )
    return job


@router.get(
    "/jobs/{job_id}/logs",
    summary="Get real-time execution step logs for analysis job"
)
def get_job_execution_logs(
    job_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    job = AnalysisService.get_job_by_id(db=db, faculty_id=current_faculty.id, job_id=job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "current_stage": job.current_stage,
        "retry_count": job.retry_count,
        "execution_time_seconds": job.execution_time_seconds,
        "logs": job.processing_logs or []
    }


@router.get(
    "/videos/{video_id}/jobs",
    response_model=List[AnalysisJobResponse],
    status_code=status.HTTP_200_OK,
    summary="List all analysis jobs for a video"
)
def list_video_jobs(
    video_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.get_jobs_by_video(
        db=db, faculty_id=current_faculty.id, video_id=video_id
    )


@router.patch(
    "/jobs/{job_id}/status",
    response_model=AnalysisJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Update analysis job status & progress"
)
def update_analysis_job_status(
    job_id: UUID,
    data: AnalysisJobStatusUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.update_job_status(
        db=db, faculty_id=current_faculty.id, job_id=job_id, data=data
    )


# ------------------ TRACKING RESULTS ------------------
@router.post(
    "/jobs/{job_id}/tracks",
    response_model=StudentTrackResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save student track result for a job"
)
def save_track_result(
    job_id: UUID,
    data: StudentTrackResultCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    data.job_id = job_id
    return AnalysisService.save_track_result(
        db=db, faculty_id=current_faculty.id, data=data
    )


@router.get(
    "/jobs/{job_id}/tracks",
    response_model=List[StudentTrackResultResponse],
    status_code=status.HTTP_200_OK,
    summary="List student tracking trajectories for a job"
)
def get_job_tracks(
    job_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.get_job_tracks(
        db=db, faculty_id=current_faculty.id, job_id=job_id
    )


# ------------------ BEHAVIOUR RESULTS ------------------
@router.post(
    "/jobs/{job_id}/behaviours",
    response_model=List[BehaviourResultResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Ingest batch temporal behaviour results"
)
def ingest_behaviour_results(
    job_id: UUID,
    data: BehaviourResultBulkCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.save_behaviour_results_bulk(
        db=db, faculty_id=current_faculty.id, job_id=job_id, results=data.results
    )


@router.get(
    "/jobs/{job_id}/behaviours",
    response_model=List[BehaviourResultResponse],
    status_code=status.HTTP_200_OK,
    summary="Query behaviour results for an analysis job"
)
@router.get(
    "/jobs/{job_id}/results",
    response_model=List[BehaviourResultResponse],
    status_code=status.HTTP_200_OK,
    summary="Query behaviour results for an analysis job (alias)"
)
def query_behaviour_results(
    job_id: UUID,
    student_id: Optional[UUID] = Query(None, description="Filter by assigned student ID"),
    track_id: Optional[int] = Query(None, description="Filter by track ID"),
    behaviour_type: Optional[str] = Query(None, description="Filter by type (Looking_Toward_Instruction, Reading, Writing, Peer_Interaction, Looking_Away)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.get_job_behaviours(
        db=db,
        faculty_id=current_faculty.id,
        job_id=job_id,
        student_id=student_id,
        track_id=track_id,
        behaviour_type=behaviour_type,
        skip=skip,
        limit=limit
    )


# ------------------ TEMPORAL ANALYTICS & SUMMARY ------------------
@router.get(
    "/jobs/{job_id}/summary",
    response_model=SessionAnalyticsSummary,
    status_code=status.HTTP_200_OK,
    summary="Get comprehensive temporal classroom analytics summary"
)
def get_job_temporal_summary(
    job_id: UUID,
    bucket_seconds: float = Query(60.0, ge=5.0, le=3600.0, description="Timeline bucket window size in seconds"),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AnalysisService.get_session_analytics_summary(
        db=db, faculty_id=current_faculty.id, job_id=job_id, bucket_seconds=bucket_seconds
    )


def _temporal_data(job_id: UUID, current_faculty: Faculty, db: Session):
    AnalysisService.get_job_by_id(db=db, faculty_id=current_faculty.id, job_id=job_id)
    data = TemporalAnalyticsService.read_all(db, job_id)
    if not data["coverage"]:
        data = TemporalAnalyticsService.build_and_persist(db, job_id)
    return data


@router.get("/jobs/{job_id}/temporal/profile")
def get_temporal_profile(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)


@router.get("/jobs/{job_id}/temporal/students")
def get_temporal_students(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)["students"]


@router.get("/jobs/{job_id}/temporal/students/{track_id}")
def get_temporal_student(job_id: UUID, track_id: int, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    data = _temporal_data(job_id, current_faculty, db)
    return next((item for item in data["students"] if item["track_id"] == track_id), {"track_id": track_id, "timeline": [], "transition_matrix": []})


@router.get("/jobs/{job_id}/temporal/transitions")
def get_temporal_transitions(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)["transitions"]


@router.get("/jobs/{job_id}/temporal/classroom-states")
def get_classroom_states(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)["classroom_states"]


@router.get("/jobs/{job_id}/temporal/entropy")
def get_classroom_entropy(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)["entropy"]


@router.get("/jobs/{job_id}/temporal/change-points")
def get_change_points(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)["change_points"]


@router.get("/jobs/{job_id}/temporal/coverage")
def get_temporal_coverage(job_id: UUID, current_faculty: Faculty = Depends(get_current_active_faculty), db: Session = Depends(get_db)):
    return _temporal_data(job_id, current_faculty, db)["coverage"]


@router.get("/jobs/{job_id}/faculty-insights", response_model=List[FacultyInsightResponse])
def get_faculty_insights(
    job_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    AnalysisService.get_job_by_id(db=db, faculty_id=current_faculty.id, job_id=job_id)
    return FacultyInsightService.get_insights_for_job(db=db, job_id=job_id)

