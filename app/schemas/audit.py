import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    user_email: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details_json: Dict[str, Any] = {}
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
