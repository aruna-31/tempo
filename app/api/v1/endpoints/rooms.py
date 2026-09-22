import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_faculty, require_admin
from app.db.session import get_db
from app.models.faculty import Faculty
from app.models.room import Camera, Room
from app.schemas.room import CameraCreate, CameraResponse, RoomCreate, RoomResponse, RoomUpdate
from app.services.audit_service import AuditService

router = APIRouter(prefix="/rooms", tags=["Rooms & Cameras"])


@router.get("", response_model=List[RoomResponse])
def get_rooms(
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """List all classroom rooms and their registered cameras."""
    rooms = db.query(Room).filter(Room.is_active == True).all()
    return rooms


@router.post("", response_model=RoomResponse, status_code=status.HTTP_201_CREATED)
def create_room(
    data: RoomCreate,
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Create a new classroom room (Admin/Dean only)."""
    existing = db.query(Room).filter(Room.room_number == data.room_number).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Room with number '{data.room_number}' already exists."
        )
    room = Room(**data.model_dump())
    db.add(room)
    db.commit()
    db.refresh(room)

    AuditService.log_event(
        db=db,
        action="CREATE_ROOM",
        resource_type="ROOM",
        resource_id=str(room.id),
        user=admin,
        details={"room_number": room.room_number, "building": room.building}
    )
    return room


@router.get("/{room_id}", response_model=RoomResponse)
def get_room(
    room_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """Get single room details."""
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Room not found")
    return room


@router.put("/{room_id}", response_model=RoomResponse)
def update_room(
    room_id: uuid.UUID,
    data: RoomUpdate,
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Update room details (Admin only)."""
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Room not found")

    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(room, k, v)

    db.commit()
    db.refresh(room)
    return room


@router.post("/{room_id}/cameras", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
def add_camera_to_room(
    room_id: uuid.UUID,
    data: CameraCreate,
    db: Session = Depends(get_db),
    admin: Faculty = Depends(require_admin)
):
    """Register an existing camera / NVR channel to a classroom (Admin only)."""
    room = db.query(Room).filter(Room.id == room_id).first()
    if not room:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Room not found")

    existing = db.query(Camera).filter(Camera.device_code == data.device_code).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera with device code '{data.device_code}' already exists."
        )

    camera = Camera(room_id=room.id, **data.model_dump())
    db.add(camera)
    db.commit()
    db.refresh(camera)

    AuditService.log_event(
        db=db,
        action="ADD_CAMERA",
        resource_type="ROOM",
        resource_id=str(room.id),
        user=admin,
        details={"camera_id": str(camera.id), "device_code": camera.device_code}
    )
    return camera
