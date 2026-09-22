from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.academic import SectionCreate, SectionResponse, SectionUpdate
from app.services.academic_service import AcademicService

router = APIRouter(prefix="/sections", tags=["Academic - Sections"])


@router.post(
    "",
    response_model=SectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a section under a subject"
)
def create_section(
    data: SectionCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.create_section(db=db, faculty_id=current_faculty.id, data=data)


@router.get(
    "/subject/{subject_id}",
    response_model=List[SectionResponse],
    status_code=status.HTTP_200_OK,
    summary="List sections for a subject"
)
def list_sections_for_subject(
    subject_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_sections_by_subject(
        db=db, faculty_id=current_faculty.id, subject_id=subject_id
    )


@router.get(
    "/{section_id}",
    response_model=SectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get section by ID"
)
def get_section(
    section_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_section_by_id(
        db=db, faculty_id=current_faculty.id, section_id=section_id
    )


@router.put(
    "/{section_id}",
    response_model=SectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update section"
)
def update_section(
    section_id: UUID,
    data: SectionUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.update_section(
        db=db, faculty_id=current_faculty.id, section_id=section_id, data=data
    )


@router.delete(
    "/{section_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete section"
)
def delete_section(
    section_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    AcademicService.delete_section(
        db=db, faculty_id=current_faculty.id, section_id=section_id
    )
