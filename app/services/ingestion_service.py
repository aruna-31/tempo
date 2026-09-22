import logging
import os
import re
import shutil
import uuid
from datetime import date, datetime, time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.analysis import AnalysisJob
from app.models.faculty import Faculty
from app.models.room import Room
from app.models.session import ClassSession
from app.models.video import Video
from app.schemas.ingest import DropzoneScanResult, IngestJobResponse, NVRWebhookPayload
from app.services.audit_service import AuditService
from app.services.job_manager import JobManagerService
from app.services.session_matcher import SessionMatcherService

logger = logging.getLogger("tempo.services.ingestion")


class IngestionService:
    @classmethod
    def ingest_recording_file(
        cls,
        db: Session,
        file_path: str,
        source_type: str = "NVR_WEBHOOK",
        recording_date: Optional[date] = None,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
        room_number: Optional[str] = None,
        device_code: Optional[str] = None,
        faculty_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None,
        user: Optional[Faculty] = None
    ) -> IngestJobResponse:
        """
        Core ingestion pipeline: matches session, creates Video record, creates AnalysisJob, and dispatches background analysis.
        """
        if not os.path.exists(file_path):
            return IngestJobResponse(
                success=False,
                message=f"Recording file not found on disk at path '{file_path}'",
                match_status="ERROR"
            )

        file_size = os.path.getsize(file_path)
        filename = os.path.basename(file_path)

        # 1. Resolve Target ClassSession
        matched_session: Optional[ClassSession] = None
        match_type = "MANUAL"
        explanation = "Explicit session specified"

        if session_id:
            matched_session = db.query(ClassSession).filter(ClassSession.id == session_id).first()
        elif not matched_session:
            matched_session, match_type, conf, explanation = SessionMatcherService.match_recording_to_session(
                db=db,
                recording_date=recording_date,
                recording_start=start_time,
                recording_end=end_time,
                room_number=room_number,
                device_code=device_code
            )

        if not matched_session:
            return IngestJobResponse(
                success=False,
                message=f"Could not automatically match recording to any active ClassSession: {explanation}",
                match_status="UNMATCHED",
                details={"file_path": file_path, "filename": filename, "explanation": explanation}
            )

        # Target faculty
        target_faculty_id = faculty_id or matched_session.faculty_id

        # 2. Move file to permanent storage if in dropzone or temp
        os.makedirs(settings.UPLOAD_RAW_DIR, exist_ok=True)
        dest_filename = f"{uuid.uuid4()}_{filename}"
        dest_path = os.path.join(settings.UPLOAD_RAW_DIR, dest_filename)
        
        # If source is outside upload dir, copy/move
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            shutil.copy2(file_path, dest_path)
        else:
            dest_path = file_path

        # 3. Create Video Record
        video_rec = Video(
            session_id=matched_session.id,
            faculty_id=target_faculty_id,
            file_path=dest_path,
            original_filename=filename,
            file_size_bytes=file_size,
            content_type="video/mp4",
            status="PROCESSING"
        )
        db.add(video_rec)
        db.commit()
        db.refresh(video_rec)

        # 4. Create AnalysisJob
        job_rec = AnalysisJob(
            video_id=video_rec.id,
            status="QUEUED",
            current_stage="QUEUED",
            progress_pct=0,
            retry_count=0,
            max_retries=settings.MAX_JOB_RETRIES,
            source_type=source_type,
            config_json={
                "match_type": match_type,
                "match_explanation": explanation,
                "room_number": room_number or matched_session.room_number,
                "ingested_at": datetime.now().isoformat()
            }
        )
        db.add(job_rec)
        db.commit()
        db.refresh(job_rec)

        # 5. Dispatch ML Job in Background
        if settings.AUTO_DISPATCH_ML_JOB:
            JobManagerService.dispatch_background_job(job_rec.id)

        # 6. Audit Log
        AuditService.log_event(
            db=db,
            action="NVR_INGESTION" if "NVR" in source_type else "LMS_INGESTION",
            resource_type="VIDEO",
            resource_id=str(video_rec.id),
            user=user,
            details={
                "session_id": str(matched_session.id),
                "job_id": str(job_rec.id),
                "match_type": match_type,
                "filename": filename
            }
        )

        return IngestJobResponse(
            success=True,
            message=f"Recording successfully ingested and matched to '{matched_session.title}'. Background analysis dispatched.",
            video_id=video_rec.id,
            session_id=matched_session.id,
            job_id=job_rec.id,
            match_status=match_type,
            details={"explanation": explanation, "status": "QUEUED"}
        )

    @classmethod
    def handle_nvr_webhook(
        cls,
        db: Session,
        payload: NVRWebhookPayload,
        client_ip: Optional[str] = None
    ) -> IngestJobResponse:
        """
        Receives and processes incoming NVR/Camera recording webhook.
        """
        logger.info(f"Received NVR webhook: device={payload.device_code}, room={payload.room_number}, file={payload.file_path}")

        file_path = payload.file_path
        if not file_path and payload.download_url:
            # If download URL is provided, could download to temp path
            file_path = payload.download_url

        if not file_path:
            return IngestJobResponse(
                success=False,
                message="Webhook payload missing required 'file_path' or 'download_url'",
                match_status="INVALID_PAYLOAD"
            )

        return cls.ingest_recording_file(
            db=db,
            file_path=file_path,
            source_type="NVR_WEBHOOK",
            recording_date=payload.recording_date or date.today(),
            start_time=payload.start_time,
            end_time=payload.end_time,
            room_number=payload.room_number,
            device_code=payload.device_code
        )

    @classmethod
    def scan_dropzone(cls, db: Session) -> DropzoneScanResult:
        """
        Scans the authorized storage dropzone folder for deposited classroom recordings.
        Parses filenames like 'ROOM-301_2026-09-16_10-00-00.mp4' or 'CAM-FRONT-301_20260916_100000.mp4'.
        """
        dropzone_dir = settings.AUTO_INGEST_DROPZONE_DIR
        os.makedirs(dropzone_dir, exist_ok=True)

        scanned = 0
        matched = 0
        unmatched = []
        errors = []

        valid_exts = tuple(settings.ALLOWED_VIDEO_EXTENSIONS)
        for fname in os.listdir(dropzone_dir):
            if not fname.lower().endswith(valid_exts):
                continue

            scanned += 1
            full_path = os.path.join(dropzone_dir, fname)
            
            # Simple metadata parser from filename
            room_num = None
            dev_code = None
            rec_date = date.today()
            rec_time = None

            # Pattern: ROOM{number} or {CODE}
            room_match = re.search(r"(?:ROOM|RM)[_-]?([A-Za-z0-9]+)", fname, re.IGNORECASE)
            if room_match:
                room_num = room_match.group(1)

            cam_match = re.search(r"(CAM[_-][A-Za-z0-9_-]+)", fname, re.IGNORECASE)
            if cam_match:
                dev_code = cam_match.group(1)

            # Try parsing date/time if present: YYYYMMDD or YYYY-MM-DD
            dt_match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", fname)
            if dt_match:
                try:
                    rec_date = date(int(dt_match.group(1)), int(dt_match.group(2)), int(dt_match.group(3)))
                except Exception:
                    pass

            time_match = re.search(r"(\d{2})[-_:](\d{2})[-_:]?(\d{2})?", fname)
            if time_match:
                try:
                    rec_time = time(int(time_match.group(1)), int(time_match.group(2)), int(time_match.group(3) or 0))
                except Exception:
                    pass

            res = cls.ingest_recording_file(
                db=db,
                file_path=full_path,
                source_type="STORAGE_DROPZONE",
                recording_date=rec_date,
                start_time=rec_time,
                room_number=room_num,
                device_code=dev_code
            )

            if res.success:
                matched += 1
                # Remove original from dropzone after moving
                if os.path.exists(full_path) and os.path.abspath(full_path) != os.path.abspath(settings.UPLOAD_RAW_DIR):
                    try:
                        os.remove(full_path)
                    except Exception:
                        pass
            else:
                unmatched.append(f"{fname}: {res.message}")

        return DropzoneScanResult(
            files_scanned=scanned,
            matched_and_ingested=matched,
            unmatched_files=unmatched,
            errors=errors
        )
