"""
Pydantic Schemas for HOD / Admin Academic & Teaching Delivery Analytics.
========================================================================
Focuses on Departmental Hierarchy (Sections → Subjects → Faculty Teaching Summary)
and Faculty Cross-Section Comparative Profiling.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class SectionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    academic_year: str
    semester: str
    subject_id: Optional[UUID] = None
    subject_code: Optional[str] = None
    subject_name: Optional[str] = None
    student_count: int
    subjects_count: int
    total_sessions_conducted: int
    avg_engagement_pct: float
    faculty_mobility_pct: float


class SectionSubjectDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    subject_id: UUID
    subject_code: str
    subject_name: str
    section_id: UUID
    section_name: str
    faculty_id: UUID
    faculty_name: str
    faculty_email: str
    faculty_designation: Optional[str] = None
    total_classes_conducted: int
    avg_student_engagement: float
    faculty_mobility_score: float
    dominant_teaching_mode: str


class TeachingDeliveryBreakdown(BaseModel):
    didactic_lectern_pct: float
    active_aisle_circulation_pct: float
    board_projection_presentation_pct: float
    desk_consultation_facilitation_pct: float


class SessionDeliveryRecord(BaseModel):
    session_id: UUID
    title: str
    session_date: str
    start_time: str
    end_time: str
    duration_minutes: float
    dominant_mode: str
    faculty_mobility_score: float
    student_attention_pct: float
    shannon_entropy: float
    pedagogical_note: str


class FacultyTeachingSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    subject_id: UUID
    subject_code: str
    subject_name: str
    section_id: UUID
    section_name: str
    faculty_id: UUID
    faculty_name: str
    faculty_email: str
    faculty_designation: Optional[str] = None
    total_classes_conducted: int
    total_instructional_hours: float
    teaching_delivery: TeachingDeliveryBreakdown
    interaction_density_score: float
    classroom_mobility_index: str
    student_synchronous_attention_pct: float
    student_active_collaboration_pct: float
    average_shannon_entropy: float
    pedagogical_summary: str
    session_history: List[SessionDeliveryRecord]


class FacultySectionAssignment(BaseModel):
    section_id: UUID
    section_name: str
    subject_id: UUID
    subject_code: str
    subject_name: str
    total_classes: int


class FacultyItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_name: str
    email: str
    department: str
    designation: Optional[str] = None
    role: str
    sections_count: int
    assignments: List[FacultySectionAssignment]


class CrossSectionItem(BaseModel):
    section_id: UUID
    section_name: str
    subject_code: str
    subject_name: str
    total_classes: int
    avg_student_attention_pct: float
    faculty_mobility_pct: float
    interactive_discussion_pct: float
    shannon_entropy: float
    dominant_style: str
    teaching_breakdown: TeachingDeliveryBreakdown


class FacultyCrossSectionComparison(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    faculty_id: UUID
    faculty_name: str
    faculty_email: str
    department: str
    designation: Optional[str] = None
    total_sections_taught: int
    sections: List[CrossSectionItem]
    cross_section_consistency_score: float
    comparative_analysis: str
    hod_actionable_insight: str


class TimetableUploadResponse(BaseModel):
    message: str
    faculties_count: int
    sections_count: int
    subjects_count: int
    rooms_count: int
    slots_count: int
    date_processed: str
    status: str = "SUCCESS"


class VideoSourceInfo(BaseModel):
    class_number: int
    label: str
    file_path: str
    file_size_mb: float
    description: str


class ClassSessionAnalysisDetail(BaseModel):
    class_number: int
    session_id: UUID
    title: str
    session_date: str
    duration_minutes: float
    video_filename: str
    faculty_mobility_pct: float
    didactic_lectern_pct: float
    aisle_circulation_pct: float
    board_exposition_pct: float
    desk_consultation_pct: float
    student_focus_pct: float
    cell_phone_distraction_pct: float
    teacher_idle_stationary_pct: float
    shannon_entropy: float
    dominant_mode: str
    fiac_category: str
    pedagogical_notes: str


class ThreeClassSummaryResponse(BaseModel):
    faculty_id: UUID
    faculty_name: str
    faculty_email: str
    faculty_designation: Optional[str] = None
    section_id: UUID
    section_name: str
    subject_id: UUID
    subject_code: str
    subject_name: str
    classes: List[ClassSessionAnalysisDetail]
    aggregate_mobility_index: str
    aggregate_student_focus_pct: float
    aggregate_teacher_active_pct: float
    cell_phone_idle_alert: str
    longitudinal_trend_summary: str
    fiac_matrix_summary: str
    video_sources: List[VideoSourceInfo]


class TodayScheduleSlot(BaseModel):
    period: int
    time_window: str
    day_of_week: str
    faculty_name: str
    section_name: str
    subject_code: str
    subject_name: str
    room_number: str


class DepartmentTimetableOverview(BaseModel):
    current_date: str
    day_of_week: str
    total_faculties: int
    total_sections: int
    total_subjects: int
    today_active_classes_count: int
    today_schedule: List[TodayScheduleSlot]

