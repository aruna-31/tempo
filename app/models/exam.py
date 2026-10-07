import uuid
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON, Uuid, func
)
from sqlalchemy.orm import relationship
from app.db.session import Base


class ExamSession(Base):
    """
    Represents an examination monitoring session in a room.
    Independent from faculty timetables.
    """
    __tablename__ = "exam_sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_code = Column(String(50), nullable=False, unique=True, index=True)
    title = Column(String(150), nullable=False)
    
    room_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("rooms.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    camera_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )

    status = Column(String(50), default="SCHEDULED", nullable=False)  # SCHEDULED, ACTIVE, COMPLETED, CANCELLED
    expected_students = Column(Integer, default=70, nullable=False)
    detected_students_count = Column(Integer, default=0, nullable=False)
    coverage_ratio = Column(Float, default=0.0, nullable=False)
    uncertainty_score = Column(Float, default=0.0, nullable=False)
    room_monitoring_status = Column(String(50), default="IDLE", nullable=False)  # IDLE, MONITORING, DEGRADED_OBSERVATION, OFFLINE

    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    invigilator_name = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    room = relationship("Room", backref="exam_sessions")
    camera = relationship("Camera", backref="exam_sessions")
    review_events = relationship(
        "ExamReviewEvent",
        back_populates="exam_session",
        cascade="all, delete-orphan",
        order_by="desc(ExamReviewEvent.start_time_offset)"
    )

    def __repr__(self) -> str:
        return f"<ExamSession(id={self.id}, code='{self.session_code}', title='{self.title}', status='{self.status}')>"


class ExamReviewEvent(Base):
    """
    Observable temporal review events flagged during an exam session.
    STRICT PRIVACY: Anonymous, session-specific IDs only (e.g. Student 04).
    STRICT COMPLIANCE: NEVER automated cheating accusations. Only observable behaviors.
    HUMAN-IN-THE-LOOP: Requires invigilator review.
    """
    __tablename__ = "exam_review_events"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    exam_session_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("exam_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Strictly anonymous session-specific ID, e.g. "Student 05"
    anonymous_student_id = Column(String(50), nullable=False, index=True)
    track_id = Column(Integer, nullable=True)

    # Observable behavior only (e.g. PEER_FACING_ORIENTATION, PROLONGED_UNUSUAL_ORIENTATION)
    event_type = Column(String(100), nullable=False, index=True)
    event_description = Column(String(255), nullable=True)

    # Temporal persistence and confidence
    confidence = Column(Float, nullable=False, default=0.85)
    start_time_offset = Column(Float, nullable=False)  # seconds from session/video start
    end_time_offset = Column(Float, nullable=False)
    duration_seconds = Column(Float, nullable=False)  # sustained duration

    # Observation quality metrics
    observation_coverage = Column(Float, nullable=False, default=0.90)
    observation_uncertainty = Column(Float, nullable=False, default=0.10)
    spatial_zone = Column(String(50), default="DESK_AREA", nullable=False)  # FRONT_ROW, MID_ROW, BACK_ROW

    # Evidence window (timeline markers and optional clip preview)
    bounding_box = Column(JSON, nullable=True)  # [x1, y1, x2, y2] normalized or pixel
    evidence_window_start = Column(Float, nullable=True)
    evidence_window_end = Column(Float, nullable=True)
    evidence_clip_path = Column(String(255), nullable=True)
    supporting_observations = Column(JSON, nullable=True)  # e.g. head angle, duration, movement variance

    # Invigilator review workflow
    review_status = Column(
        String(50),
        default="PENDING_REVIEW",
        nullable=False,
        index=True
    )  # PENDING_REVIEW, CONFIRMED_OBSERVATION, DISMISSED, ACTION_TAKEN
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    review_notes = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    exam_session = relationship("ExamSession", back_populates="review_events")

    def __repr__(self) -> str:
        return f"<ExamReviewEvent(id={self.id}, student='{self.anonymous_student_id}', type='{self.event_type}', status='{self.review_status}')>"
