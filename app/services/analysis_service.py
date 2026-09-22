from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from app.models.analysis import AnalysisJob, BehaviourResult, StudentTrackResult
from app.models.academic import Student
from app.models.video import Video
from app.schemas.analysis import (
    AnalysisJobCreate,
    AnalysisJobStatusUpdate,
    BehaviourResultCreate,
    ObservableBehaviourType,
    SessionAnalyticsSummary,
    StudentTrackMetrics,
    StudentTrackResultCreate,
    TemporalBehaviourBucket,
)
from app.services.session_service import SessionService


class AnalysisService:
    # ------------------ ANALYSIS JOBS ------------------
    @staticmethod
    def create_job(db: Session, faculty_id: UUID, data: AnalysisJobCreate) -> AnalysisJob:
        video = SessionService.get_video_by_id(db, faculty_id, data.video_id)

        job = AnalysisJob(
            video_id=video.id,
            status="PENDING",
            progress_pct=0,
            config_json=data.config_json or {}
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def get_job_by_id(db: Session, faculty_id: UUID, job_id: UUID) -> AnalysisJob:
        job = (
            db.query(AnalysisJob)
            .join(Video, AnalysisJob.video_id == Video.id)
            .filter(AnalysisJob.id == job_id, Video.faculty_id == faculty_id)
            .first()
        )
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis job not found or does not belong to your account"
            )
        return job

    @staticmethod
    def get_jobs_by_video(
        db: Session, faculty_id: UUID, video_id: UUID
    ) -> List[AnalysisJob]:
        SessionService.get_video_by_id(db, faculty_id, video_id)
        return (
            db.query(AnalysisJob)
            .filter(AnalysisJob.video_id == video_id)
            .order_by(AnalysisJob.created_at.desc())
            .all()
        )

    @staticmethod
    def update_job_status(
        db: Session, faculty_id: UUID, job_id: UUID, data: AnalysisJobStatusUpdate
    ) -> AnalysisJob:
        job = AnalysisService.get_job_by_id(db, faculty_id, job_id)
        job.status = data.status.strip().upper()

        if data.progress_pct is not None:
            job.progress_pct = max(0, min(100, data.progress_pct))

        if job.status == "PROCESSING" and not job.started_at:
            job.started_at = datetime.now(timezone.utc)
        elif job.status in ("COMPLETED", "FAILED"):
            job.completed_at = datetime.now(timezone.utc)
            if job.status == "COMPLETED":
                job.progress_pct = 100

        if data.error_message is not None:
            job.error_message = data.error_message

        db.commit()
        db.refresh(job)
        return job

    # ------------------ TRACKING & BEHAVIOUR RESULTS ------------------
    @staticmethod
    def save_track_result(
        db: Session, faculty_id: UUID, data: StudentTrackResultCreate
    ) -> StudentTrackResult:
        AnalysisService.get_job_by_id(db, faculty_id, data.job_id)

        track = StudentTrackResult(
            job_id=data.job_id,
            student_id=data.student_id,
            track_id=data.track_id,
            bounding_box_history=data.bounding_box_history,
            start_frame=data.start_frame,
            end_frame=data.end_frame
        )
        db.add(track)
        db.commit()
        db.refresh(track)
        return track

    @staticmethod
    def save_behaviour_results_bulk(
        db: Session, faculty_id: UUID, job_id: UUID, results: List[BehaviourResultCreate]
    ) -> List[BehaviourResult]:
        AnalysisService.get_job_by_id(db, faculty_id, job_id)

        created_records: List[BehaviourResult] = []
        for r in results:
            record = BehaviourResult(
                job_id=job_id,
                student_id=r.student_id,
                track_id=r.track_id,
                frame_number=r.frame_number,
                timestamp_seconds=r.timestamp_seconds,
                behaviour_type=r.behaviour_type.value,
                confidence=r.confidence,
                metadata_json=r.metadata_json or {}
            )
            db.add(record)
            created_records.append(record)

        db.commit()
        return created_records

    @staticmethod
    def get_job_tracks(
        db: Session, faculty_id: UUID, job_id: UUID
    ) -> List[StudentTrackResult]:
        AnalysisService.get_job_by_id(db, faculty_id, job_id)
        return db.query(StudentTrackResult).filter(StudentTrackResult.job_id == job_id).all()

    @staticmethod
    def get_job_behaviours(
        db: Session,
        faculty_id: UUID,
        job_id: UUID,
        student_id: Optional[UUID] = None,
        track_id: Optional[int] = None,
        behaviour_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 500
    ) -> List[BehaviourResult]:
        AnalysisService.get_job_by_id(db, faculty_id, job_id)

        query = db.query(BehaviourResult).filter(BehaviourResult.job_id == job_id)
        if student_id:
            query = query.filter(BehaviourResult.student_id == student_id)
        if track_id is not None:
            query = query.filter(BehaviourResult.track_id == track_id)
        if behaviour_type:
            query = query.filter(BehaviourResult.behaviour_type == behaviour_type.strip())

        return query.order_by(BehaviourResult.timestamp_seconds.asc()).offset(skip).limit(limit).all()

    # ------------------ TEMPORAL ANALYTICS & AGGREGATIONS ------------------
    @staticmethod
    def get_session_analytics_summary(
        db: Session, faculty_id: UUID, job_id: UUID, bucket_seconds: float = 60.0
    ) -> SessionAnalyticsSummary:
        job = AnalysisService.get_job_by_id(db, faculty_id, job_id)
        video = db.query(Video).filter(Video.id == job.video_id).first()
        session_id = video.session_id if video else job.video_id

        behaviours = (
            db.query(BehaviourResult)
            .filter(BehaviourResult.job_id == job_id)
            .order_by(BehaviourResult.timestamp_seconds.asc())
            .all()
        )

        tracks = db.query(StudentTrackResult).filter(StudentTrackResult.job_id == job_id).all()

        if not behaviours:
            return SessionAnalyticsSummary(
                job_id=job.id,
                video_id=job.video_id,
                session_id=session_id,
                status=job.status,
                total_frames_analyzed=0,
                total_tracks_identified=len(tracks),
                overall_engagement_score=0.0,
                timeline_buckets=[],
                track_metrics=[]
            )

        max_time = max(b.timestamp_seconds for b in behaviours)
        buckets_count = int(max_time // bucket_seconds) + 1

        buckets: List[TemporalBehaviourBucket] = []
        for i in range(buckets_count):
            t_start = i * bucket_seconds
            t_end = (i + 1) * bucket_seconds
            bucket = TemporalBehaviourBucket(timestamp_start=t_start, timestamp_end=t_end)
            buckets.append(bucket)

        track_map: Dict[int, Dict[str, Any]] = {}
        total_engaged = 0
        total_records = len(behaviours)

        for b in behaviours:
            b_idx = min(int(b.timestamp_seconds // bucket_seconds), buckets_count - 1)
            cur_bucket = buckets[b_idx]
            cur_bucket.total_detections += 1

            b_type = b.behaviour_type
            # 5 classes
            if b_type == "Looking_Toward_Instruction":
                cur_bucket.looking_toward_instruction_count += 1
                total_engaged += 1
            elif b_type == "Reading":
                cur_bucket.reading_count += 1
                total_engaged += 1
            elif b_type == "Writing":
                cur_bucket.writing_count += 1
                total_engaged += 1
            elif b_type == "Peer_Interaction":
                cur_bucket.peer_interaction_count += 1
            elif b_type == "Looking_Away":
                cur_bucket.looking_away_count += 1

            # Track mapping
            if b.track_id not in track_map:
                track_map[b.track_id] = {
                    "student_id": b.student_id,
                    "track_id": b.track_id,
                    "counts": {
                        "Looking_Toward_Instruction": 0,
                        "Reading": 0,
                        "Writing": 0,
                        "Peer_Interaction": 0,
                        "Looking_Away": 0
                    },
                    "total": 0
                }
            if b_type in track_map[b.track_id]["counts"]:
                track_map[b.track_id]["counts"][b_type] += 1
            track_map[b.track_id]["total"] += 1

        # Calculate engagement index for buckets (Instruction + Reading + Writing)
        for bk in buckets:
            if bk.total_detections > 0:
                engaged_in_bucket = (
                    bk.looking_toward_instruction_count + bk.reading_count + bk.writing_count
                )
                bk.engagement_index = round((engaged_in_bucket / bk.total_detections) * 100.0, 2)

        # Build Per-Track Metrics
        track_metrics: List[StudentTrackMetrics] = []
        for track_id, info in track_map.items():
            tot = info["total"]
            cnts = info["counts"]
            display_label = f"Student {track_id:02d}"
            if info["student_id"]:
                st = db.query(Student).filter(Student.id == info["student_id"]).first()
                if st:
                    display_label = f"{st.name} ({st.roll_number})"

            dominant = max(cnts.items(), key=lambda x: x[1])[0] if tot > 0 else "Looking_Toward_Instruction"
            metrics = StudentTrackMetrics(
                track_id=track_id,
                student_id=info["student_id"],
                display_label=display_label,
                looking_toward_instruction_pct=round((cnts["Looking_Toward_Instruction"] / tot) * 100.0, 2) if tot > 0 else 0.0,
                reading_pct=round((cnts["Reading"] / tot) * 100.0, 2) if tot > 0 else 0.0,
                writing_pct=round((cnts["Writing"] / tot) * 100.0, 2) if tot > 0 else 0.0,
                peer_interaction_pct=round((cnts["Peer_Interaction"] / tot) * 100.0, 2) if tot > 0 else 0.0,
                looking_away_pct=round((cnts["Looking_Away"] / tot) * 100.0, 2) if tot > 0 else 0.0,
                dominant_behaviour=dominant
            )
            track_metrics.append(metrics)

        overall_engagement = round((total_engaged / total_records) * 100.0, 2) if total_records > 0 else 0.0

        return SessionAnalyticsSummary(
            job_id=job.id,
            video_id=job.video_id,
            session_id=session_id,
            status=job.status,
            total_frames_analyzed=total_records,
            total_tracks_identified=len(track_map),
            overall_engagement_score=overall_engagement,
            timeline_buckets=buckets,
            track_metrics=track_metrics
        )
