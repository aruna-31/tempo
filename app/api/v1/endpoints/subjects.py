from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.academic import (
    SectionCreate,
    SectionResponse,
    SubjectCreate,
    SubjectResponse,
    SubjectUpdate,
)
from app.services.academic_service import AcademicService

router = APIRouter(prefix="/subjects", tags=["Academic - Subjects"])


@router.post(
    "",
    response_model=SubjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new subject"
)
def create_subject(
    data: SubjectCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.create_subject(db=db, faculty_id=current_faculty.id, data=data)


@router.get(
    "",
    response_model=List[SubjectResponse],
    status_code=status.HTTP_200_OK,
    summary="List faculty's subjects"
)
def list_subjects(
    skip: int = 0,
    limit: int = 100,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_faculty_subjects(
        db=db, faculty_id=current_faculty.id, skip=skip, limit=limit
    )


@router.get(
    "/{subject_id}/sections",
    response_model=List[SectionResponse],
    status_code=status.HTTP_200_OK,
    summary="List sections for a subject (nested alias of /sections/subject/{id})"
)
def list_sections_nested(
    subject_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_sections_by_subject(
        db=db, faculty_id=current_faculty.id, subject_id=subject_id
    )


@router.post(
    "/{subject_id}/sections",
    response_model=SectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a section under a subject (nested alias of POST /sections)"
)
def create_section_nested(
    subject_id: UUID,
    data: dict,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    section_data = SectionCreate(
        subject_id=subject_id,
        name=str(data.get("name", "")).strip(),
        academic_year=str(data.get("academic_year", "2025-2026")).strip(),
        semester=str(data.get("semester", "EVEN")).strip(),
    )
    return AcademicService.create_section(db=db, faculty_id=current_faculty.id, data=section_data)


@router.get(
    "/{subject_id}",
    response_model=SubjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Get subject by ID"
)
def get_subject(
    subject_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_subject_by_id(
        db=db, faculty_id=current_faculty.id, subject_id=subject_id
    )


@router.put(
    "/{subject_id}",
    response_model=SubjectResponse,
    status_code=status.HTTP_200_OK,
    summary="Update subject"
)
def update_subject(
    subject_id: UUID,
    data: SubjectUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.update_subject(
        db=db, faculty_id=current_faculty.id, subject_id=subject_id, data=data
    )


@router.delete(
    "/{subject_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete subject"
)
def delete_subject(
    subject_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    AcademicService.delete_subject(
        db=db, faculty_id=current_faculty.id, subject_id=subject_id
    )
