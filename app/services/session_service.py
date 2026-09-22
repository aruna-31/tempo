from datetime import date
from typing import List, Optional, Tuple
from uuid import UUID
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from app.core.storage import StorageService
from app.models.academic import Section, Subject
from app.models.analysis import AnalysisJob
from app.models.session import ClassSession
from app.models.video import Video
from app.schemas.session import ClassSessionCreate, ClassSessionUpdate
from app.schemas.video import VideoCreate, VideoStatusDetailResponse, VideoStatusUpdate
from app.services.academic_service import AcademicService
from app.services.ml_connector import get_ml_connector


class SessionService:
    # ------------------ CLASS SESSIONS ------------------
    @staticmethod
    def create_session(
        db: Session, faculty_id: UUID, data: ClassSessionCreate
    ) -> ClassSession:
        # Verify section ownership
        AcademicService.get_section_by_id(db, faculty_id, data.section_id)

        session = ClassSession(
            faculty_id=faculty_id,
            section_id=data.section_id,
            title=data.title.strip(),
            session_date=data.session_date,
            start_time=data.start_time,
            end_time=data.end_time,
            room_number=data.room_number.strip() if data.room_number else None
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def get_faculty_sessions(
        db: Session,
        faculty_id: UUID,
        section_id: Optional[UUID] = None,
        session_date: Optional[date] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ClassSession]:
        query = db.query(ClassSession).filter(ClassSession.faculty_id == faculty_id)
        if section_id:
            query = query.filter(ClassSession.section_id == section_id)
        if session_date:
            query = query.filter(ClassSession.session_date == session_date)
        return query.order_by(ClassSession.session_date.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_session_by_id(
        db: Session, faculty_id: UUID, session_id: UUID
    ) -> ClassSession:
        session = (
            db.query(ClassSession)
            .filter(
                ClassSession.id == session_id,
                ClassSession.faculty_id == faculty_id
            )
            .first()
        )
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Class session not found or does not belong to your account"
            )
        return session

    @staticmethod
    def update_session(
        db: Session, faculty_id: UUID, session_id: UUID, data: ClassSessionUpdate
    ) -> ClassSession:
        session = SessionService.get_session_by_id(db, faculty_id, session_id)
        if data.title is not None:
            session.title = data.title.strip()
        if data.session_date is not None:
            session.session_date = data.session_date
        if data.start_time is not None:
            session.start_time = data.start_time
        if data.end_time is not None:
            session.end_time = data.end_time
        if data.room_number is not None:
            session.room_number = data.room_number.strip()

        db.commit()
        db.refresh(session)
        return session

    @staticmethod
    def delete_session(db: Session, faculty_id: UUID, session_id: UUID) -> None:
        session = SessionService.get_session_by_id(db, faculty_id, session_id)
        db.delete(session)
        db.commit()

    # ------------------ VIDEOS & UPLOADS ------------------
    @staticmethod
    def upload_and_create_video(
        db: Session,
        faculty_id: UUID,
        session_id: UUID,
        file: UploadFile,
        duration_seconds: Optional[float] = None
    ) -> Tuple[Video, AnalysisJob]:
        """
        Validates file, writes to object/file storage, creates Video record,
        automatically provisions a PENDING AnalysisJob, and registers with ML connector.
        """
        # 1. Verify session belongs to faculty
        SessionService.get_session_by_id(db, faculty_id, session_id)

        # 2. Validate video format and MIME type
        clean_name, file_ext = StorageService.validate_video_file(file)

        # 3. Stream write file into storage
        saved_path, file_size = StorageService.save_upload_file_sync(file, clean_name, file_ext)

        content_type = StorageService.get_mime_type(saved_path)

        # 4. Save Video record
        video = Video(
            session_id=session_id,
            faculty_id=faculty_id,
            file_path=saved_path,
            original_filename=clean_name,
            file_size_bytes=file_size,
            duration_seconds=duration_seconds,
            content_type=content_type,
            status="UPLOADED"
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        # 5. Automatically create initial AnalysisJob. The config reflects the real pipeline
        # (YOLO + ByteTrack + ResNet-18 + temporal model) with parameters sourced from
        # models/model_metadata.json - the single source of truth. No fabricated values.
        from app.services.job_manager import job_ml_sequence_length, job_ml_temporal_type
        default_ml_config = {
            "pipeline": "yolo_bytetrack_resnet18_temporal",
            "detector": "yolov8n",
            "tracker": "bytetrack",
            "spatial_backbone": "resnet18",
            "temporal_model_type": job_ml_temporal_type(),
            "sampling_fps": 2.0,
            "temporal_sequence_length": job_ml_sequence_length(),
            "target_behaviours": [
                "Looking_Toward_Instruction",
                "Reading",
                "Writing",
                "Peer_Interaction",
                "Looking_Away"
            ]
        }
        job = AnalysisJob(
            video_id=video.id,
            status="PENDING",
            progress_pct=0,
            config_json=default_ml_config
        )
        db.add(job)
        db.commit()
        db.refresh(job)

        # 6. Dispatch to plug-and-play ML connector stub
        ml = get_ml_connector()
        ml.submit_job(
            job_id=job.id,
            video_id=video.id,
            raw_video_path=saved_path,
            config=default_ml_config
        )

        return video, job

    @staticmethod
    def create_video(db: Session, faculty_id: UUID, data: VideoCreate) -> Video:
        SessionService.get_session_by_id(db, faculty_id, data.session_id)

        video = Video(
            session_id=data.session_id,
            faculty_id=faculty_id,
            file_path=data.file_path.strip(),
            original_filename=data.original_filename.strip(),
            file_size_bytes=data.file_size_bytes,
            duration_seconds=data.duration_seconds,
            content_type=data.content_type,
            output_video_path=data.output_video_path,
            status="UPLOADED"
        )
        db.add(video)
        db.commit()
        db.refresh(video)
        return video

    @staticmethod
    def get_faculty_videos(
        db: Session,
        faculty_id: UUID,
        session_id: Optional[UUID] = None,
        status_filter: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Video]:
        query = db.query(Video).filter(Video.faculty_id == faculty_id)
        if session_id:
            query = query.filter(Video.session_id == session_id)
        if status_filter:
            query = query.filter(Video.status == status_filter.strip().upper())
        return query.order_by(Video.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_video_by_id(db: Session, faculty_id: UUID, video_id: UUID) -> Video:
        video = (
            db.query(Video)
            .filter(Video.id == video_id, Video.faculty_id == faculty_id)
            .first()
        )
        if not video:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Video not found or does not belong to your account"
            )
        return video

    @staticmethod
    def get_video_status_detail(
        db: Session, faculty_id: UUID, video_id: UUID
    ) -> VideoStatusDetailResponse:
        video = SessionService.get_video_by_id(db, faculty_id, video_id)
        latest_job = (
            db.query(AnalysisJob)
            .filter(AnalysisJob.video_id == video_id)
            .order_by(AnalysisJob.created_at.desc())
            .first()
        )

        stream_url = f"/api/v1/videos/{video.id}/stream"
        output_stream_url = (
            f"/api/v1/videos/{video.id}/output-stream" if video.output_video_path else None
        )
        is_ready = bool(
            video.output_video_path or (latest_job and latest_job.status == "COMPLETED")
        )

        return VideoStatusDetailResponse(
            video_id=video.id,
            session_id=video.session_id,
            original_filename=video.original_filename,
            file_size_bytes=video.file_size_bytes,
            duration_seconds=video.duration_seconds,
            video_status=video.status,
            content_type=video.content_type,
            latest_analysis_job_id=latest_job.id if latest_job else None,
            analysis_status=latest_job.status if latest_job else None,
            analysis_progress_pct=latest_job.progress_pct if latest_job else 0,
            error_message=latest_job.error_message if latest_job else None,
            stream_url=stream_url,
            output_stream_url=output_stream_url,
            is_output_ready=is_ready
        )

    @staticmethod
    def update_video_status(
        db: Session, faculty_id: UUID, video_id: UUID, data: VideoStatusUpdate
    ) -> Video:
        video = SessionService.get_video_by_id(db, faculty_id, video_id)
        video.status = data.status.strip()
        if data.duration_seconds is not None:
            video.duration_seconds = data.duration_seconds
        if data.output_video_path is not None:
            video.output_video_path = data.output_video_path
        db.commit()
        db.refresh(video)
        return video

    @staticmethod
    def delete_video(db: Session, faculty_id: UUID, video_id: UUID) -> None:
        video = SessionService.get_video_by_id(db, faculty_id, video_id)
        # Remove files from storage
        StorageService.delete_file(video.file_path)
        if video.output_video_path:
            StorageService.delete_file(video.output_video_path)

        db.delete(video)
        db.commit()
