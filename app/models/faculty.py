import uuid
from sqlalchemy import Column, String, Boolean, DateTime, Uuid, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Faculty(Base):
    __tablename__ = "faculties"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    department = Column(String(100), nullable=False)
    designation = Column(String(100), nullable=True)
    role = Column(String(50), default="FACULTY", nullable=False)  # FACULTY, ADMIN, DEAN, HOD
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    subjects = relationship("Subject", back_populates="faculty", cascade="all, delete-orphan")
    class_sessions = relationship("ClassSession", back_populates="faculty", cascade="all, delete-orphan")
    schedules = relationship("FacultySchedule", back_populates="faculty", cascade="all, delete-orphan")
    videos = relationship("Video", back_populates="faculty", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Faculty(id={self.id}, email='{self.email}', name='{self.full_name}')>"
