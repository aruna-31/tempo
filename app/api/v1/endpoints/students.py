from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.academic import (
    StudentBulkCreate,
    StudentCreate,
    StudentResponse,
    StudentUpdate,
)
from app.services.academic_service import AcademicService

router = APIRouter(prefix="/students", tags=["Academic - Students"])


@router.post(
    "",
    response_model=StudentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Enroll a student in a section"
)
def create_student(
    data: StudentCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.create_student(db=db, faculty_id=current_faculty.id, data=data)


@router.post(
    "/bulk",
    response_model=List[StudentResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Bulk enroll students in a section"
)
def bulk_create_students(
    data: StudentBulkCreate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.bulk_create_students(
        db=db, faculty_id=current_faculty.id, data=data
    )


@router.get(
    "/section/{section_id}",
    response_model=List[StudentResponse],
    status_code=status.HTTP_200_OK,
    summary="List students in a section"
)
def list_students_for_section(
    section_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_students_by_section(
        db=db, faculty_id=current_faculty.id, section_id=section_id
    )


@router.get(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get student details"
)
def get_student(
    student_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.get_student_by_id(
        db=db, faculty_id=current_faculty.id, student_id=student_id
    )


@router.put(
    "/{student_id}",
    response_model=StudentResponse,
    status_code=status.HTTP_200_OK,
    summary="Update student record"
)
def update_student(
    student_id: UUID,
    data: StudentUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    return AcademicService.update_student(
        db=db, faculty_id=current_faculty.id, student_id=student_id, data=data
    )


@router.delete(
    "/{student_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete student record"
)
def delete_student(
    student_id: UUID,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    AcademicService.delete_student(
        db=db, faculty_id=current_faculty.id, student_id=student_id
    )
