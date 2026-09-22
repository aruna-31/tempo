import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Uuid, JSON, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from app.db.session import Base

JsonType = JSON().with_variant(JSONB, "postgresql")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("faculties.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    user_email = Column(String(255), nullable=True, index=True)
    action = Column(String(100), nullable=False, index=True)  # LOGIN, UPLOAD_VIDEO, NVR_INGESTION, EXPORT_REPORT, RETRY_JOB, DELETE_VIDEO
    resource_type = Column(String(50), nullable=False)  # VIDEO, SESSION, JOB, REPORT, ROOM, SCHEDULE
    resource_id = Column(String(100), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(255), nullable=True)
    details_json = Column(JsonType, default=dict, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    faculty = relationship("Faculty")

    def __repr__(self) -> str:
        return f"<AuditLog(id={self.id}, action='{self.action}', user='{self.user_email}')>"
