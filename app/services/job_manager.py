import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.models.analysis import AnalysisJob, BehaviourResult, StudentTrackResult
from app.models.video import Video
from app.services.audit_service import AuditService

logger = logging.getLogger("tempo.services.job_manager")

_session_factory = None


def set_session_factory(factory):
    global _session_factory
    _session_factory = factory


def get_db_session() -> Session:
    if _session_factory is not None:
        return _session_factory()
    return SessionLocal()


class JobManagerService:
    @staticmethod
    def append_job_log(
        db: Session,
        job: AnalysisJob,
        stage: str,
        message: str,
        level: str = "INFO"
    ) -> None:
        """
        Appends a structured, timestamped log entry to the job's processing_logs JSON array.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "level": level,
            "message": message
        }
        logs = list(job.processing_logs or [])
        logs.append(log_entry)
        job.processing_logs = logs
        job.current_stage = stage
        db.add(job)
        db.commit()
        logger.info(f"[Job {job.id}] [{stage}] {message}")

    @classmethod
    def run_analysis_job_sync(cls, job_id: uuid.UUID) -> bool:
        """
        Synchronously executes an analysis job with stage updates and step logging.

        Pipeline errors (invalid weights, missing model files, missing YOLO weights,
        undecodable video, YOLO inference failures) are NOT retried - they are real
        failures and must mark the job FAILED with the actual error message.
        Never reports COMPLETED when processing actually failed.
        """
        db = get_db_session()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if not job:
                logger.error(f"Job {job_id} not found")
                return False

            video = db.query(Video).filter(Video.id == job.video_id).first()
            if not video:
                job.status = "FAILED"
                job.error_message = "Associated video file not found in database."
                db.commit()
                return False

            start_perf_time = time.time()
            job.started_at = datetime.now(timezone.utc)
            job.status = "PROCESSING"
            job.progress_pct = 5
            video.status = "PROCESSING"
            db.commit()
            cls.append_job_log(
                db, job, "QUEUED",
                f"Job dequeued. Source: {job.source_type}."
            )

            success = False
            last_error: Optional[Exception] = None

            try:
                # Stage 1: Video validation
                cls.append_job_log(db, job, "INITIALIZING", f"Validating input video: {video.file_path}")
                if not os.path.exists(video.file_path):
                    raise FileNotFoundError(f"Video file path '{video.file_path}' does not exist on storage.")

                job.progress_pct = 15
                db.commit()

                # Stage 2: ML Pipeline Execution (YOLO detection + ByteTrack + ResNet-18 + temporal model)
                cls.append_job_log(
                    db, job, "DETECTION",
                    "Executing YOLO person detection with spatial context margins across sampled frames..."
                )
                job.progress_pct = 30
                db.commit()

                cls.append_job_log(
                    db, job, "TRACKING",
                    "Executing ByteTrack multi-object association to construct continuous student trajectories..."
                )
                job.progress_pct = 50
                db.commit()

                cls.append_job_log(
                    db, job, "BEHAVIOUR_ANALYSIS",
                    (
                        f"Extracting temporal sequences (seq_len={job_ml_sequence_length()}) and predicting "
                        f"5 observable behaviours with ResNet-18 + {job_ml_temporal_type()}..."
                    )
                )
                job.progress_pct = 70
                db.commit()

                # Call ML Service (raises on invalid weights / missing models / undecodable video)
                from app.ml.service import get_ml_service
                ml_svc = get_ml_service()
                ml_results = ml_svc.analyze_video_full(
                    video_path=video.file_path,
                    render_output_video=True
                )

                predictions = ml_results.get("predictions", [])
                tracks_data = ml_results.get("tracks", [])
                out_vid = ml_results.get("output_video_path")
                if out_vid:
                    video.output_video_path = out_vid

                cls.append_job_log(
                    db, job, "AGGREGATING",
                    f"ML execution succeeded. Detected {len(tracks_data)} student tracks and {len(predictions)} behaviour windows. Persisting to database..."
                )
                job.progress_pct = 85
                db.commit()

                # Clear existing results for this job if re-running
                db.query(BehaviourResult).filter(BehaviourResult.job_id == job.id).delete()
                db.query(StudentTrackResult).filter(StudentTrackResult.job_id == job.id).delete()

                # Persist StudentTrackResults (session-scoped temporary Track IDs only)
                track_records = []
                for t in tracks_data:
                    track_rec = StudentTrackResult(
                        job_id=job.id,
                        track_id=int(t["track_id"]),
                        bounding_box_history=t.get("bounding_box_history", []),
                        start_frame=int(t.get("start_frame", 0)),
                        end_frame=int(t.get("end_frame", 0))
                    )
                    track_records.append(track_rec)
                db.add_all(track_records)

                # Persist BehaviourResults using the exact fields produced by the ML pipeline
                beh_records = []
                for p in predictions:
                    beh_rec = BehaviourResult(
                        job_id=job.id,
                        track_id=int(p["track_id"]),
                        frame_number=int(p["frame_number"]),
                        timestamp_seconds=float(p["timestamp_seconds"]),
                        behaviour_type=str(p["behaviour_type"]),
                        confidence=float(p["confidence"]),
                        metadata_json=dict(p.get("metadata_json", {}))
                    )
                    beh_records.append(beh_rec)
                db.add_all(beh_records)

                success = True

            except Exception as e:
                last_error = e
                logger.exception(f"Job {job.id} encountered error during execution: {str(e)}")
                stage = job.current_stage or "PROCESSING"
                try:
                    cls.append_job_log(
                        db, job, stage,
                        f"Error during execution: {str(e)}",
                        level="ERROR"
                    )
                    job.retry_count = (job.retry_count or 0) + 1
                    db.commit()
                except Exception:
                    db.rollback()

            end_perf_time = time.time()
            job.execution_time_seconds = round(end_perf_time - start_perf_time, 2)
            job.completed_at = datetime.now(timezone.utc)

            if success:
                job.status = "COMPLETED"
                job.current_stage = "COMPLETED"
                job.progress_pct = 100
                job.error_message = None
                video.status = "ANALYZED"
                db.commit()
                cls.append_job_log(
                    db, job, "COMPLETED",
                    f"Job completed successfully in {job.execution_time_seconds}s. Analytics and per-track reports generated."
                )
            else:
                job.status = "FAILED"
                job.current_stage = "FAILED"
                job.error_message = str(last_error) if last_error else "Unknown processing failure."
                video.status = "FAILED"
                db.commit()
                cls.append_job_log(
                    db, job, "FAILED",
                    f"Fatal failure: {job.error_message}",
                    level="FATAL"
                )

            return success

        finally:
            db.close()

    @classmethod
    def dispatch_background_job(cls, job_id: uuid.UUID) -> threading.Thread:
        """
        Dispatches analysis job to a background daemon thread.
        """
        thread = threading.Thread(
            target=cls.run_analysis_job_sync,
            args=(job_id,),
            daemon=True,
            name=f"MLWorker-Job-{job_id}"
        )
        thread.start()
        return thread


def job_ml_sequence_length() -> int:
    """Sequence length from the single source of truth (model_metadata.json)."""
    try:
        import json
        weights_path = os.path.abspath(settings.MODEL_WEIGHTS_PATH)
        meta_path = os.path.join(os.path.dirname(weights_path), "model_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["sequence_length"])
    except Exception:
        return int(settings.TEMPORAL_SEQUENCE_LENGTH)


def job_ml_temporal_type() -> str:
    """Temporal model type from the single source of truth (model_metadata.json)."""
    try:
        import json
        weights_path = os.path.abspath(settings.MODEL_WEIGHTS_PATH)
        meta_path = os.path.join(os.path.dirname(weights_path), "model_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            return str(json.load(f)["temporal_model_type"]).upper()
    except Exception:
        return str(settings.TEMPORAL_MODEL_TYPE).upper()
