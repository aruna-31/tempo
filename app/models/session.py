import uuid
from sqlalchemy import Column, String, Date, Time, DateTime, ForeignKey, Uuid, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class ClassSession(Base):
    __tablename__ = "class_sessions"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    faculty_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("faculties.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    section_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("sections.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    room_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("rooms.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    title = Column(String(200), nullable=False)
    session_date = Column(Date, nullable=False, index=True)
    day_of_week = Column(String(20), nullable=True)  # MONDAY, TUESDAY, etc.
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    room_number = Column(String(50), nullable=True)  # Backward compatible string / override
    status = Column(String(50), default="SCHEDULED", nullable=False)  # SCHEDULED, IN_PROGRESS, COMPLETED, CANCELLED
    is_recurring = Column(String(10), default="false", nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    faculty = relationship("Faculty", back_populates="class_sessions")
    section = relationship("Section", back_populates="class_sessions")
    room = relationship("Room", back_populates="class_sessions")
    videos = relationship("Video", back_populates="class_session", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ClassSession(id={self.id}, title='{self.title}', date={self.session_date})>"
