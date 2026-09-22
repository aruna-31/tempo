import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Uuid, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    faculty_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("faculties.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    code = Column(String(50), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    faculty = relationship("Faculty", back_populates="subjects")
    sections = relationship("Section", back_populates="subject", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Subject(id={self.id}, code='{self.code}', name='{self.name}')>"


class Section(Base):
    __tablename__ = "sections"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    subject_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    name = Column(String(50), nullable=False)
    academic_year = Column(String(20), nullable=False)
    semester = Column(String(20), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    subject = relationship("Subject", back_populates="sections")
    students = relationship("Student", back_populates="section", cascade="all, delete-orphan")
    class_sessions = relationship("ClassSession", back_populates="section", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Section(id={self.id}, name='{self.name}', subject_id={self.subject_id})>"


class Student(Base):
    __tablename__ = "students"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    section_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("sections.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    roll_number = Column(String(50), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    email = Column(String(255), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    section = relationship("Section", back_populates="students")
    track_results = relationship("StudentTrackResult", back_populates="student")
    behaviour_results = relationship("BehaviourResult", back_populates="student")

    def __repr__(self) -> str:
        return f"<Student(id={self.id}, roll_number='{self.roll_number}', name='{self.name}')>"
