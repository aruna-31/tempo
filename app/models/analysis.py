import uuid
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.db.session import Base

# Dialect-agnostic JSON type that compiles to native PostgreSQL JSONB on Postgres
JsonType = JSON().with_variant(JSONB, "postgresql")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    video_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    status = Column(String(50), default="PENDING", nullable=False)  # PENDING, QUEUED, PROCESSING, COMPLETED, FAILED, RETRYING
    current_stage = Column(String(50), default="QUEUED", nullable=False)  # QUEUED, DETECTION, TRACKING, BEHAVIOUR_ANALYSIS, AGGREGATING, COMPLETED, FAILED
    progress_pct = Column(Integer, default=0, nullable=False)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)
    source_type = Column(String(50), default="MANUAL_UPLOAD", nullable=False)  # MANUAL_UPLOAD, NVR_WEBHOOK, LMS_INGESTION, STORAGE_DROPZONE
    
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Detailed step execution history: [{"timestamp": "...", "stage": "...", "level": "INFO", "message": "..."}]
    processing_logs = Column(JsonType, default=list, nullable=False)
    config_json = Column(JsonType, default=dict, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    video = relationship("Video", back_populates="analysis_jobs")
    track_results = relationship("StudentTrackResult", back_populates="job", cascade="all, delete-orphan")
    behaviour_results = relationship("BehaviourResult", back_populates="job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<AnalysisJob(id={self.id}, video_id={self.video_id}, status='{self.status}', retries={self.retry_count})>"


class StudentTrackResult(Base):
    __tablename__ = "student_track_results"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    job_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    student_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    track_id = Column(Integer, nullable=False, index=True)
    # List of bounding box data points over time: [{"frame": 1, "timestamp": 0.033, "bbox": [x1, y1, x2, y2]}]
    bounding_box_history = Column(JsonType, default=list, nullable=False)
    start_frame = Column(Integer, nullable=False)
    end_frame = Column(Integer, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    job = relationship("AnalysisJob", back_populates="track_results")
    student = relationship("Student", back_populates="track_results")

    def __repr__(self) -> str:
        return f"<StudentTrackResult(id={self.id}, job_id={self.job_id}, track_id={self.track_id})>"


class BehaviourResult(Base):
    __tablename__ = "behaviour_results"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    job_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    student_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("students.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    track_id = Column(Integer, nullable=False, index=True)
    frame_number = Column(Integer, nullable=False, index=True)
    timestamp_seconds = Column(Float, nullable=False, index=True)
    # Behaviour types: ATTENTIVE, SLEEPING, USING_PHONE, TALKING, DISTRACTED, NOTE_TAKING, HEAD_DOWN, YAWNING, LOOKING_AWAY
    behaviour_type = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    metadata_json = Column(JsonType, default=dict, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    job = relationship("AnalysisJob", back_populates="behaviour_results")
    student = relationship("Student", back_populates="behaviour_results")

    def __repr__(self) -> str:
        return (
            f"<BehaviourResult(id={self.id}, job_id={self.job_id}, "
            f"type='{self.behaviour_type}', t={self.timestamp_seconds}s)>"
        )
