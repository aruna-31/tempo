import uuid
from sqlalchemy import Column, String, BigInteger, Float, DateTime, ForeignKey, Uuid, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("class_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    faculty_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("faculties.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    file_path = Column(String(500), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    content_type = Column(String(100), default="video/mp4", nullable=False)
    output_video_path = Column(String(500), nullable=True)
    status = Column(String(50), default="UPLOADED", nullable=False)  # UPLOADED, PROCESSING, ANALYZED, FAILED

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    class_session = relationship("ClassSession", back_populates="videos")
    faculty = relationship("Faculty", back_populates="videos")
    analysis_jobs = relationship(
        "AnalysisJob",
        back_populates="video",
        cascade="all, delete-orphan",
        order_by="desc(AnalysisJob.created_at)"
    )

    def __repr__(self) -> str:
        return f"<Video(id={self.id}, filename='{self.original_filename}', status='{self.status}')>"
