import uuid
from sqlalchemy import Column, String, Time, Boolean, DateTime, ForeignKey, Uuid, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class FacultySchedule(Base):
    __tablename__ = "faculty_schedules"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    faculty_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("faculties.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    subject_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
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
    # Day of week: MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY
    day_of_week = Column(String(20), nullable=False, index=True)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    academic_year = Column(String(20), nullable=False, default="2025-2026")
    semester = Column(String(20), nullable=False, default="ODD")
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    faculty = relationship("Faculty", back_populates="schedules")
    subject = relationship("Subject")
    section = relationship("Section")
    room = relationship("Room", back_populates="faculty_schedules")

    def __repr__(self) -> str:
        return (
            f"<FacultySchedule(id={self.id}, day='{self.day_of_week}', "
            f"{self.start_time}-{self.end_time}, room_id={self.room_id})>"
        )
