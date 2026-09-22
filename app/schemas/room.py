import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CameraBase(BaseModel):
    camera_name: str = Field(..., max_length=100)
    device_code: str = Field(..., max_length=50)
    stream_url_or_channel: Optional[str] = None
    ip_address: Optional[str] = None
    nvr_channel_id: Optional[int] = None
    location_in_room: str = Field(default="FRONT", max_length=50)
    status: str = Field(default="ACTIVE", max_length=50)


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: uuid.UUID
    room_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RoomBase(BaseModel):
    room_number: str = Field(..., max_length=50)
    building: str = Field(default="Academic Block A", max_length=100)
    floor: int = Field(default=1, ge=0)
    capacity: int = Field(default=60, gt=0)
    is_active: bool = True


class RoomCreate(RoomBase):
    pass


class RoomUpdate(BaseModel):
    room_number: Optional[str] = None
    building: Optional[str] = None
    floor: Optional[int] = None
    capacity: Optional[int] = None
    is_active: Optional[bool] = None


class RoomResponse(RoomBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    cameras: List[CameraResponse] = []

    model_config = ConfigDict(from_attributes=True)
