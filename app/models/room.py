import uuid
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Uuid, func
from sqlalchemy.orm import relationship
from app.db.session import Base


class Room(Base):
    __tablename__ = "rooms"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    room_number = Column(String(50), nullable=False, unique=True, index=True)
    building = Column(String(100), nullable=False, default="Academic Block A")
    floor = Column(Integer, nullable=False, default=1)
    capacity = Column(Integer, nullable=False, default=60)
    is_active = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    cameras = relationship("Camera", back_populates="room", cascade="all, delete-orphan")
    class_sessions = relationship("ClassSession", back_populates="room")
    faculty_schedules = relationship("FacultySchedule", back_populates="room")

    def __repr__(self) -> str:
        return f"<Room(id={self.id}, number='{self.room_number}', building='{self.building}')>"


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    room_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("rooms.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    camera_name = Column(String(100), nullable=False)
    device_code = Column(String(50), nullable=False, unique=True, index=True)  # e.g., CAM-301-FRONT
    stream_url_or_channel = Column(String(255), nullable=True)  # RTSP/NVR channel
    ip_address = Column(String(50), nullable=True)
    nvr_channel_id = Column(Integer, nullable=True)
    location_in_room = Column(String(50), default="FRONT", nullable=False)  # FRONT, REAR, CEILING
    status = Column(String(50), default="ACTIVE", nullable=False)  # ACTIVE, INACTIVE, MAINTENANCE

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    # Relationships
    room = relationship("Room", back_populates="cameras")

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, code='{self.device_code}', status='{self.status}')>"
