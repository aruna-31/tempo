import uuid
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.dependencies import get_current_active_faculty, require_admin
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.audit import AuditLogResponse
from app.schemas.ingest import DropzoneScanResult, IngestJobResponse, NVRWebhookPayload
from app.services.audit_service import AuditService
from app.services.ingestion_service import IngestionService
from app.services.storage_manager import StorageManagerService
from app.models.audit import AuditLog

router = APIRouter(prefix="/ingest", tags=["Ingestion & Campus Integration"])


@router.post(
    "/nvr-webhook",
    response_model=IngestJobResponse,
    status_code=status.HTTP_200_OK,
    summary="NVR / Camera Server Recording Webhook",
    description="Authorized endpoint for campus NVR/camera recording server notifications."
)
def nvr_webhook_listener(
    payload: NVRWebhookPayload,
    request: Request,
    x_nvr_token: str = Header(default="", alias="X-NVR-Token"),
    db: Session = Depends(get_db)
):
    """
    Receives post-class recording event from classroom camera/NVR infrastructure,
    matches the recording to the scheduled session, creates Video/AnalysisJob records,
    and dispatches background ML analysis.
    """
    # Token validation (if configured in production)
    if settings.NVR_WEBHOOK_SECRET and x_nvr_token and x_nvr_token != settings.NVR_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid NVR webhook authentication token."
        )

    client_ip = request.client.host if request.client else None
    result = IngestionService.handle_nvr_webhook(db=db, payload=payload, client_ip=client_ip)
    return result


@router.post(
    "/scan-dropzone",
    response_model=DropzoneScanResult,
    status_code=status.HTTP_200_OK,
    summary="Scan Storage Dropzone for Deposited Recordings",
    description="Scans the university dropzone folder, matches deposited video files to scheduled sessions, and triggers ML analysis."
)
def scan_dropzone_endpoint(
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Admin-triggered scan of campus storage dropzone folder."""
    result = IngestionService.scan_dropzone(db=db)
    AuditService.log_event(
        db=db,
        action="SCAN_DROPZONE",
        resource_type="STORAGE",
        user=admin,
        details={
            "files_scanned": result.files_scanned,
            "matched_and_ingested": result.matched_and_ingested
        }
    )
    return result


@router.get(
    "/storage-stats",
    summary="Get Campus Video Storage & Retention Stats"
)
def get_storage_stats(
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Returns disk usage and storage retention statistics."""
    return StorageManagerService.get_storage_stats()


@router.post(
    "/enforce-retention",
    summary="Trigger Video Retention Policy Cleanup"
)
def enforce_retention(
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Enforces college video retention policy (Admin only)."""
    res = StorageManagerService.enforce_retention_policy(db=db)
    AuditService.log_event(
        db=db,
        action="ENFORCE_RETENTION",
        resource_type="STORAGE",
        user=admin,
        details=res
    )
    return {
        "message": f"Retention policy executed. Purged {res['purged_count']} raw files.",
        "details": res
    }


@router.get(
    "/audit-logs",
    response_model=List[AuditLogResponse],
    summary="View System Compliance & Security Audit Logs"
)
def get_audit_logs(
    limit: int = 50,
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Returns recent system audit events (Admin only)."""
    logs = db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()
    return logs
