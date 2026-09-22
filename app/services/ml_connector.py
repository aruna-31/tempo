import abc
import concurrent.futures
from datetime import datetime, timezone
import logging
import traceback
from typing import Any, Callable, Dict, Optional
from uuid import UUID
from sqlalchemy.orm import Session, sessionmaker
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.analysis import AnalysisJob, BehaviourResult, StudentTrackResult
from app.models.video import Video
from app.ml.service import get_ml_service

logger = logging.getLogger("tempo.ml.connector")

# Background worker thread pool for video ML inference
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ml_worker_")

# Session factory getter that can be customized for testing
_session_factory: Callable[[], Session] = SessionLocal


def set_session_factory(factory: Callable[[], Session]):
    global _session_factory
    _session_factory = factory


def get_worker_session() -> Session:
    return _session_factory()


def run_video_analysis_worker(job_id: UUID) -> None:
    """
    Background worker task executing video temporal behaviour analysis.
    Uses its own isolated database session and handles failure gracefully.
    """
    db: Session = get_worker_session()
    try:
        job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if not job:
            logger.error(f"[ML Worker] Job {job_id} not found in database.")
            return

        video = db.query(Video).filter(Video.id == job.video_id).first()
        if not video:
            job.status = "FAILED"
            job.error_message = f"Associated video {job.video_id} not found."
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # 1. Update job to PROCESSING
        logger.info(f"[ML Worker] Starting processing for Job ID: {job.id}, Video: {video.file_path}")
        job.status = "PROCESSING"
        job.started_at = datetime.now(timezone.utc)
        job.progress_pct = 5
        video.status = "PROCESSING"
        db.commit()

        def progress_callback(pct: int):
            try:
                job.progress_pct = pct
                db.commit()
            except Exception as pe:
                logger.warning(f"[ML Worker] Error updating progress: {pe}")

        # 2. Run ML Pipeline Inference (Detection + ByteTrack Tracking + ResNet-18 + Temporal Recurrence)
        ml_service = get_ml_service()
        analysis_output = ml_service.analyze_video_full(
            video_path=video.file_path,
            progress_callback=progress_callback,
            config=job.config_json,
            render_output_video=True
        )
        predictions = analysis_output["predictions"]
        track_results = analysis_output["tracks"]
        output_video_path = analysis_output.get("output_video_path")
        if output_video_path:
            video.output_video_path = output_video_path

        # 3. Store Student Track Trajectories in PostgreSQL
        logger.info(f"[ML Worker] Saving {len(track_results)} student track results to PostgreSQL...")
        for tr in track_results:
            track_entry = StudentTrackResult(
                job_id=job.id,
                track_id=tr["track_id"],
                start_frame=tr["start_frame"],
                end_frame=tr["end_frame"],
                bounding_box_history=tr["bounding_box_history"]
            )
            db.add(track_entry)

        # 4. Store prediction results in PostgreSQL
        logger.info(f"[ML Worker] Saving {len(predictions)} behaviour results to PostgreSQL...")
        for pred in predictions:
            result = BehaviourResult(
                job_id=job.id,
                track_id=pred["track_id"],
                frame_number=pred["frame_number"],
                timestamp_seconds=pred["timestamp_seconds"],
                behaviour_type=pred["behaviour_type"],
                confidence=pred["confidence"],
                metadata_json=pred.get("metadata_json", {})
            )
            db.add(result)

        # 5. Mark Job as COMPLETED
        job.status = "COMPLETED"
        job.current_stage = "COMPLETED"
        job.progress_pct = 100
        job.execution_time_seconds = round((datetime.now(timezone.utc) - job.started_at).total_seconds(), 2) if job.started_at else None
        job.completed_at = datetime.now(timezone.utc)
        video.status = "ANALYZED"
        db.commit()
        logger.info(f"[ML Worker] Job {job.id} successfully COMPLETED.")

    except Exception as e:
        logger.error(f"[ML Worker] Error processing Job {job_id}: {e}\n{traceback.format_exc()}")
        try:
            db.rollback()
            failed_job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if failed_job:
                failed_job.status = "FAILED"
                failed_job.current_stage = "FAILED"
                failed_job.error_message = str(e)
                failed_job.completed_at = datetime.now(timezone.utc)
                failed_video = db.query(Video).filter(Video.id == failed_job.video_id).first()
                if failed_video:
                    failed_video.status = "FAILED"
                db.commit()
        except Exception as inner_e:
            logger.critical(f"[ML Worker] Failed to update error state for Job {job_id}: {inner_e}")
    finally:
        db.close()


class BaseMLPipeline(abc.ABC):
    @abc.abstractmethod
    def submit_job(
        self,
        job_id: UUID,
        video_id: UUID,
        raw_video_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def get_job_status(self, job_id: UUID) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def cancel_job(self, job_id: UUID) -> bool:
        pass


class BackgroundMLPipelineConnector(BaseMLPipeline):
    """
    Production-ready asynchronous ML connector.
    Dispatches analysis jobs to background thread pool worker.
    """

    def submit_job(
        self,
        job_id: UUID,
        video_id: UUID,
        raw_video_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        if settings.AUTO_DISPATCH_ML_JOB:
            logger.info(f"Submitting Job {job_id} to asynchronous background worker pool.")
            executor.submit(run_video_analysis_worker, job_id)

        return {
            "job_id": str(job_id),
            "video_id": str(video_id),
            "raw_video_path": raw_video_path,
            "status": "QUEUED",
            "message": "Job successfully queued for temporal behaviour ML pipeline",
            "queued_at": datetime.now(timezone.utc).isoformat()
        }

    def get_job_status(self, job_id: UUID) -> Dict[str, Any]:
        db = get_worker_session()
        try:
            job = db.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
            if job:
                return {
                    "job_id": str(job.id),
                    "status": job.status,
                    "progress_pct": job.progress_pct,
                    "error_message": job.error_message
                }
            return {"job_id": str(job_id), "status": "UNKNOWN", "progress_pct": 0}
        finally:
            db.close()

    def cancel_job(self, job_id: UUID) -> bool:
        return True


# Pluggable global instance
ml_connector: BaseMLPipeline = BackgroundMLPipelineConnector()


def get_ml_connector() -> BaseMLPipeline:
    return ml_connector
