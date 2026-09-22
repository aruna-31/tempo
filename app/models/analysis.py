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
    faculty_insights = relationship("FacultyInsight", back_populates="job", cascade="all, delete-orphan")

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


class StudentTemporalProfile(Base):
    __tablename__ = "student_temporal_profiles"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(Integer, nullable=False, index=True)
    timeline_json = Column(JsonType, default=list, nullable=False)
    behaviour_distribution_json = Column(JsonType, default=dict, nullable=False)
    transition_matrix_json = Column(JsonType, default=list, nullable=False)
    behaviour_duration_json = Column(JsonType, default=dict, nullable=False)
    total_observed_duration = Column(Float, nullable=False, default=0.0)
    segment_count = Column(Integer, nullable=False, default=0)
    temporal_coverage = Column(Float, nullable=False, default=0.0)


class BehaviourTransition(Base):
    __tablename__ = "behaviour_transitions"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    track_id = Column(Integer, nullable=True, index=True)
    from_behaviour = Column(String(50), nullable=False)
    to_behaviour = Column(String(50), nullable=False)
    count = Column(Integer, nullable=False, default=0)
    probability = Column(Float, nullable=False, default=0.0)


class ClassroomTemporalState(Base):
    __tablename__ = "classroom_temporal_states"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    state = Column(String(80), nullable=False)
    behaviour_distribution_json = Column(JsonType, default=dict, nullable=False)
    students_contributing = Column(Integer, nullable=False, default=0)


class ClassroomEntropy(Base):
    __tablename__ = "classroom_entropy"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    entropy = Column(Float, nullable=False)
    behaviour_distribution_json = Column(JsonType, default=dict, nullable=False)


class ChangePoint(Base):
    __tablename__ = "classroom_change_points"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)
    previous_distribution_json = Column(JsonType, default=dict, nullable=False)
    new_distribution_json = Column(JsonType, default=dict, nullable=False)
    change_score = Column(Float, nullable=False)
    previous_state = Column(String(80), nullable=False)
    new_state = Column(String(80), nullable=False)


class CoverageMetric(Base):
    __tablename__ = "coverage_metrics"
    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False, unique=True)
    manual_reference_student_count = Column(Integer, nullable=True)
    detected_student_count = Column(Integer, nullable=False, default=0)
    tracked_student_count = Column(Integer, nullable=False, default=0)
    temporal_ready_track_count = Column(Integer, nullable=False, default=0)
    detection_coverage = Column(Float, nullable=True)
    tracking_coverage = Column(Float, nullable=False, default=0.0)
    temporal_coverage = Column(Float, nullable=False, default=0.0)


class FacultyInsight(Base):
    __tablename__ = "faculty_insights"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    job_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    category = Column(String(50), nullable=False, index=True)  # PACING, VARIETY, INTERACTION, ATTENTION_PATTERNS, COVERAGE
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    observation = Column(Text, nullable=False)
    pedagogical_context = Column(Text, nullable=False)
    suggested_action = Column(Text, nullable=False)
    coverage_context = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    job = relationship("AnalysisJob", back_populates="faculty_insights")

    def __repr__(self) -> str:
        return f"<FacultyInsight(id={self.id}, job_id={self.job_id}, category='{self.category}')>"

