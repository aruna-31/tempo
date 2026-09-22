from typing import List, Optional
from uuid import UUID
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_active_faculty
from app.core.storage import StorageService
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.video import (
    VideoCreate,
    VideoResponse,
    VideoStatusDetailResponse,
    VideoStatusUpdate,
    VideoUploadResponse,
)
from app.services.session_service import SessionService

router = APIRouter(prefix="/videos", tags=["Class Videos & Streaming"])


@router.post(
    "/upload",
    response_model=VideoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload classroom video file and initiate analysis job",
    description=(
        "Uploads a classroom video file (.mp4, .mov, .webm) via multipart/form-data. "
        "Saves the video to storage, creates a Video record in PostgreSQL, and "
        "automatically creates an AnalysisJob with 'PENDING' status."
    )
)
def upload_video_file(
    session_id: UUID = Form(..., description="Target ClassSession ID"),
    duration_seconds: Optional[float] = Form(None, description="Optional video length in seconds"),
    file: UploadFile = File(..., description="Video file (MP4, MOV, WebM)"),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
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


@router.post(
    "",
    response_model=VideoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register pre-existing video metadata"
)
def register_video_metadata(
    data: VideoCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.create_video(db=db, faculty_id=current_faculty.id, data=data)


@router.get(
    "",
    response_model=List[VideoResponse],
    status_code=status.HTTP_200_OK,
    summary="List faculty's uploaded videos"
)
def list_videos(
    session_id: Optional[UUID] = Query(None, description="Filter by ClassSession ID"),
    status_filter: Optional[str] = Query(None, description="Filter by status (UPLOADED, PROCESSING, ANALYZED, FAILED)"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.get_faculty_videos(
        db=db,
        faculty_id=current_faculty.id,
        session_id=session_id,
        status_filter=status_filter,
        skip=skip,
        limit=limit
    )


@router.get(
    "/{video_id}",
    response_model=VideoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get video metadata by ID"
)
def get_video(
    video_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.get_video_by_id(
        db=db, faculty_id=current_faculty.id, video_id=video_id
    )


@router.get(
    "/{video_id}/status",
    response_model=VideoStatusDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Check video processing & analysis status"
)
def get_video_status(
    video_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.get_video_status_detail(
        db=db, faculty_id=current_faculty.id, video_id=video_id
    )


@router.get(
    "/{video_id}/stream",
    summary="Stream raw uploaded video (HTTP 206 Partial Content)",
    description="Streams the raw uploaded video file with byte-range support for browser playback and seeking."
)
def stream_video(
    video_id: UUID,
    range: Optional[str] = Header(None, alias="Range"),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    video = SessionService.get_video_by_id(db=db, faculty_id=current_faculty.id, video_id=video_id)
    return StorageService.create_range_response(
        file_path=video.file_path,
        range_header=range
    )


@router.get(
    "/{video_id}/output-stream",
    summary="Stream completed ML output / annotated video (HTTP 206 Partial Content)",
    description="Streams the rendered/annotated output video with bounding box overlays and behaviour labels."
)
def stream_output_video(
    video_id: UUID,
    range: Optional[str] = Header(None, alias="Range"),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    video = SessionService.get_video_by_id(db=db, faculty_id=current_faculty.id, video_id=video_id)
    if not video.output_video_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed output video is not yet available for this recording"
        )
    return StorageService.create_range_response(
        file_path=video.output_video_path,
        range_header=range
    )


@router.patch(
    "/{video_id}/status",
    response_model=VideoResponse,
    status_code=status.HTTP_200_OK,
    summary="Update video status & output path"
)
def update_video_status(
    video_id: UUID,
    data: VideoStatusUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return SessionService.update_video_status(
        db=db, faculty_id=current_faculty.id, video_id=video_id, data=data
    )


@router.delete(
    "/{video_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete video record and stored video files"
)
def delete_video(
    video_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    SessionService.delete_video(
        db=db, faculty_id=current_faculty.id, video_id=video_id
    )
