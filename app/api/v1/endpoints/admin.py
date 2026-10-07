"""
Admin / HOD Academic Analytics API Endpoints.
==============================================
Provides hierarchical section drill-down and faculty cross-section
teaching delivery analysis.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.admin import (
    DepartmentTimetableOverview,
    FacultyCrossSectionComparison,
    FacultyItem,
    FacultyTeachingSummary,
    SectionSubjectDetail,
    SectionSummary,
    ThreeClassSummaryResponse,
    TimetableUploadResponse,
)
from app.services.admin_service import AdminService

router = APIRouter(prefix="/admin", tags=["HOD & Academic Administration"])



@router.get(
    "/sections",
    response_model=List[SectionSummary],
    summary="Get all departmental sections",
    description="Returns all sections in the department with enrollment, subject counts, and active telemetry."
)
def get_sections(
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    return AdminService.get_sections(db=db)


@router.get(
    "/sections/{section_id}/subjects",
    response_model=List[SectionSubjectDetail],
    summary="Get subjects and faculty for a section",
    description="Returns all subjects taught in the specified section along with assigned faculty and teaching style badges."
)
def get_section_subjects(
    section_id: UUID,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    return AdminService.get_section_subjects(db=db, section_id=section_id)


@router.get(
    "/sections/{section_id}/subjects/{subject_id}/teaching-summary",
    response_model=FacultyTeachingSummary,
    summary="Get faculty teaching and interaction dossier",
    description="Returns full pedagogical summary, teaching delivery modes (lectern vs aisle circulation vs board), and longitudinal class history."
)
def get_faculty_teaching_summary(
    section_id: UUID,
    subject_id: UUID,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    summary = AdminService.get_subject_faculty_teaching_summary(
        db=db, section_id=section_id, subject_id=subject_id
    )
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Section or Subject record not found"
        )
    return summary


@router.get(
    "/faculties",
    response_model=List[FacultyItem],
    summary="Get all faculty members with assigned sections",
    description="Returns all department faculty with a list of sections and subjects they teach."
)
def get_all_faculties(
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    return AdminService.get_faculties_with_assignments(db=db)


@router.get(
    "/faculties/{faculty_id}/cross-section-comparison",
    response_model=FacultyCrossSectionComparison,
    summary="Get faculty cross-section comparative analytics",
    description="Compares an instructor's pedagogical delivery, mobility, and student interaction across multiple sections (e.g. Section A vs Section B)."
)
def get_faculty_cross_section_comparison(
    faculty_id: UUID,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    comparison = AdminService.get_faculty_cross_section_comparison(
        db=db, faculty_id=faculty_id
    )
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Faculty record not found or lacks assigned sections"
        )
    return comparison


@router.post(
    "/upload-department-timetable",
    response_model=TimetableUploadResponse,
    summary="Upload Department Timetable PDF (aSc Timetable)",
    description="Parses the official faculty timetable PDF, extracting all faculties, sections, subjects, rooms, and weekly schedule slots into the database."
)
async def upload_department_timetable(
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    pdf_bytes = None
    if file:
        pdf_bytes = await file.read()
    try:
        return AdminService.parse_and_ingest_department_timetable(
            db=db,
            pdf_bytes=pdf_bytes,
            pdf_path=r"c:\Users\aruna\Downloads\Faculty-TT-03-07-26.pdf" if not pdf_bytes else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process timetable PDF: {str(e)}"
        )


@router.get(
    "/timetable-overview",
    response_model=DepartmentTimetableOverview,
    summary="Get today's segregated department schedule overview",
    description="Returns current date, day of week, active periods, and teaching matrix for all faculty and sections."
)
def get_department_timetable_overview(
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    return AdminService.get_department_timetable_overview(db=db)


@router.post(
    "/sections/{section_id}/subjects/{subject_id}/analyze-three-classes",
    response_model=ThreeClassSummaryResponse,
    summary="Upload minimum of 3 videos and generate multi-class teaching & interaction summary",
    description="Analyzes the last 3 classes for a faculty member handling a specific section and subject. Measures mobility, lectern vs circulation, mobile phone idling, and student focus."
)
def analyze_three_classes(
    section_id: UUID,
    subject_id: UUID,
    use_benchmarks: bool = True,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    try:
        return AdminService.analyze_three_classes(
            db=db,
            section_id=section_id,
            subject_id=subject_id,
            use_benchmarks=use_benchmarks
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

