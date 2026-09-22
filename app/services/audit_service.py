import logging
import uuid
from typing import Any, Dict, Optional
from sqlalchemy.orm import Session
from app.models.audit import AuditLog
from app.models.faculty import Faculty

logger = logging.getLogger("tempo.services.audit")


class AuditService:
    @staticmethod
    def log_event(
        db: Session,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user: Optional[Faculty] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        """
        Records an audit event to the database for security compliance and tracking.
        """
        try:
            audit_entry = AuditLog(
                user_id=user.id if user else None,
                user_email=user.email if user else "SYSTEM",
                action=action,
                resource_type=resource_type,
                resource_id=str(resource_id) if resource_id else None,
                ip_address=ip_address,
                user_agent=user_agent,
                details_json=details or {}
            )
            db.add(audit_entry)
            db.commit()
            db.refresh(audit_entry)
            logger.info(f"Audit log recorded: {action} on {resource_type}:{resource_id} by {user.email if user else 'SYSTEM'}")
            return audit_entry
        except Exception as e:
            logger.error(f"Failed to record audit log: {str(e)}")
            db.rollback()
            # Non-blocking fallback
            return None
