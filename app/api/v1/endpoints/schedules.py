from datetime import date, datetime, time, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.academic import Section, Subject
from app.models.faculty import Faculty
from app.models.room import Room
from app.models.schedule import FacultySchedule
from app.models.session import ClassSession
from app.schemas.schedule import (
    FacultyScheduleCreate,
    FacultyScheduleResponse,
    SessionGenerateRequest,
    TodayClassResponse,
    TimetableConfirmRequest,
    TimetableConfirmResponse,
    TimetableExtractResponse,
    TimetableExtractedSlot,
)
from app.services.audit_service import AuditService
from app.services.timetable_ocr_service import TimetableOCRService

router = APIRouter(prefix="/schedules", tags=["Faculty Schedules & Timetable"])


@router.get("/today", response_model=List[TodayClassResponse], summary="Fetch today's scheduled classes for current faculty")
@router.get("/my-classes-today", response_model=List[TodayClassResponse], summary="Fetch today's scheduled classes for current faculty (alias)")
def get_today_classes(
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """
    Returns only the authenticated faculty's classes scheduled for TODAY from their timetable,
    ensuring each class has a corresponding ClassSession for video upload & ML analysis.
    """
    today = date.today()
    day_name = today.strftime("%A").upper()

    # 1. Fetch recurring timetable slots for today
    schedules = db.query(FacultySchedule).filter(
        FacultySchedule.faculty_id == current_faculty.id,
        FacultySchedule.day_of_week == day_name,
        FacultySchedule.is_active == True
    ).all()

    today_classes: List[TodayClassResponse] = []
    seen_session_ids = set()

    for s in schedules:
        # Check if ClassSession already exists for today's slot
        session = db.query(ClassSession).filter(
            ClassSession.faculty_id == current_faculty.id,
            ClassSession.section_id == s.section_id,
            ClassSession.session_date == today,
            ClassSession.start_time == s.start_time
        ).first()

        subject = db.query(Subject).filter(Subject.id == s.subject_id).first()
        section = db.query(Section).filter(Section.id == s.section_id).first()
        room = db.query(Room).filter(Room.id == s.room_id).first() if s.room_id else None

        room_str = room.name if room else (s.room_id and str(s.room_id)[:8]) or "Room TBD"
        subj_code = subject.code if subject else "N/A"
        subj_name = subject.name if subject else "Class Session"
        sec_name = section.name if section else "Section A"
        title = f"{subj_code}: {subj_name} - {sec_name}"

        if not session:
            session = ClassSession(
                faculty_id=current_faculty.id,
                section_id=s.section_id,
                room_id=s.room_id,
                room_number=room_str,
                title=title,
                session_date=today,
                day_of_week=day_name,
                start_time=s.start_time,
                end_time=s.end_time,
                status="SCHEDULED",
                is_recurring="true"
            )
            db.add(session)
            db.commit()
            db.refresh(session)

        seen_session_ids.add(session.id)

        from app.models.video import Video
        from app.models.analysis import AnalysisJob
        videos = db.query(Video).filter(Video.session_id == session.id).all()
        latest_video = videos[-1] if videos else None
        latest_job = None
        if latest_video:
            latest_job = db.query(AnalysisJob).filter(AnalysisJob.video_id == latest_video.id).order_by(AnalysisJob.created_at.desc()).first()

        today_classes.append(TodayClassResponse(
            session_id=session.id,
            schedule_id=s.id,
            title=session.title or title,
            subject_code=subj_code,
            subject_name=subj_name,
            section_name=sec_name,
            room_number=session.room_number or room_str,
            start_time=s.start_time.strftime("%H:%M") if hasattr(s.start_time, "strftime") else str(s.start_time),
            end_time=s.end_time.strftime("%H:%M") if hasattr(s.end_time, "strftime") else str(s.end_time),
            day_of_week=day_name,
            session_date=today,
            status=session.status,
            videos_count=len(videos),
            latest_video_id=latest_video.id if latest_video else None,
            latest_job_id=latest_job.id if latest_job else None,
            latest_job_status=latest_job.status if latest_job else None,
            has_analysis=latest_job is not None and latest_job.status == "COMPLETED"
        ))

    # 2. Also query any non-recurring sessions explicitly created for today
    from app.models.video import Video
    from app.models.analysis import AnalysisJob
    explicit_sessions = db.query(ClassSession).filter(
        ClassSession.faculty_id == current_faculty.id,
        ClassSession.session_date == today
    ).all()

    for sess in explicit_sessions:
        if sess.id in seen_session_ids:
            continue
        seen_session_ids.add(sess.id)
        section = db.query(Section).filter(Section.id == sess.section_id).first()
        subject = db.query(Subject).filter(Subject.id == section.subject_id).first() if section else None
        
        videos = db.query(Video).filter(Video.session_id == sess.id).all()
        latest_video = videos[-1] if videos else None
        latest_job = None
        if latest_video:
            latest_job = db.query(AnalysisJob).filter(AnalysisJob.video_id == latest_video.id).order_by(AnalysisJob.created_at.desc()).first()

        today_classes.append(TodayClassResponse(
            session_id=sess.id,
            schedule_id=None,
            title=sess.title,
            subject_code=subject.code if subject else "N/A",
            subject_name=subject.name if subject else "Class Session",
            section_name=section.name if section else "Section",
            room_number=sess.room_number or "Room TBD",
            start_time=sess.start_time.strftime("%H:%M") if hasattr(sess.start_time, "strftime") else str(sess.start_time),
            end_time=sess.end_time.strftime("%H:%M") if hasattr(sess.end_time, "strftime") else str(sess.end_time),
            day_of_week=day_name,
            session_date=today,
            status=sess.status,
            videos_count=len(videos),
            latest_video_id=latest_video.id if latest_video else None,
            latest_job_id=latest_job.id if latest_job else None,
            latest_job_status=latest_job.status if latest_job else None,
            has_analysis=latest_job is not None and latest_job.status == "COMPLETED"
        ))

    today_classes.sort(key=lambda x: x.start_time)
    return today_classes


@router.get("", response_model=List[FacultyScheduleResponse])
def get_schedules(
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """List recurring weekly timetable schedules for current faculty (or all if admin)."""
    if current_faculty.role in ("ADMIN", "DEAN") or current_faculty.is_superuser:
        schedules = db.query(FacultySchedule).filter(FacultySchedule.is_active == True).all()
    else:
        schedules = db.query(FacultySchedule).filter(
            FacultySchedule.faculty_id == current_faculty.id,
            FacultySchedule.is_active == True
        ).all()
    return schedules


@router.post("", response_model=FacultyScheduleResponse, status_code=status.HTTP_201_CREATED)
def create_schedule_slot(
    data: FacultyScheduleCreate,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """Create a new weekly recurring timetable schedule slot for a course/section."""
    # Verify subject and section exist
    subject = db.query(Subject).filter(Subject.id == data.subject_id).first()
    if not subject:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subject not found")

    section = db.query(Section).filter(Section.id == data.section_id).first()
    if not section:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section not found")

    if data.room_id:
        room = db.query(Room).filter(Room.id == data.room_id).first()
        if not room:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Room not found")

    day_clean = data.day_of_week.strip().upper()
    valid_days = {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"}
    if day_clean not in valid_days:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid day_of_week '{data.day_of_week}'. Must be one of: {', '.join(sorted(valid_days))}"
        )

    schedule = FacultySchedule(
        faculty_id=current_faculty.id,
        subject_id=data.subject_id,
        section_id=data.section_id,
        room_id=data.room_id,
        day_of_week=day_clean,
        start_time=data.start_time,
        end_time=data.end_time,
        academic_year=data.academic_year,
        semester=data.semester
    )
    db.add(schedule)
    db.commit()
    db.refresh(schedule)

    AuditService.log_event(
        db=db,
        action="CREATE_SCHEDULE",
        resource_type="SCHEDULE",
        resource_id=str(schedule.id),
        user=current_faculty,
        details={"day": day_clean, "start_time": str(data.start_time), "room_id": str(data.room_id)}
    )
    return schedule


@router.post("/generate-sessions")
def generate_sessions_from_schedule(
    data: SessionGenerateRequest,
    db: Session = Depends(get_db),
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    """
    Instantiates calendar ClassSessions between start_date and end_date based on weekly recurring schedules.
    """
    if data.start_date > data.end_date:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="start_date cannot be after end_date")

    target_faculty_id = current_faculty.id
    if (current_faculty.role in ("ADMIN", "DEAN") or current_faculty.is_superuser) and data.faculty_id:
        target_faculty_id = data.faculty_id

    schedules = db.query(FacultySchedule).filter(
        FacultySchedule.faculty_id == target_faculty_id,
        FacultySchedule.is_active == True
    ).all()

    if not schedules:
        return {"message": "No active recurring schedules found to generate sessions from.", "generated_count": 0}

    sched_by_day = {}
    for s in schedules:
        sched_by_day.setdefault(s.day_of_week.upper(), []).append(s)

    created_sessions = []
    curr_date = data.start_date
    delta = timedelta(days=1)

    while curr_date <= data.end_date:
        day_name = curr_date.strftime("%A").upper()
        if day_name in sched_by_day:
            for s in sched_by_day[day_name]:
                # Check if session already exists for this slot
                existing = db.query(ClassSession).filter(
                    ClassSession.faculty_id == target_faculty_id,
                    ClassSession.section_id == s.section_id,
                    ClassSession.session_date == curr_date,
                    ClassSession.start_time == s.start_time
                ).first()

                if not existing:
                    subject = db.query(Subject).filter(Subject.id == s.subject_id).first()
                    section = db.query(Section).filter(Section.id == s.section_id).first()
                    room = db.query(Room).filter(Room.id == s.room_id).first() if s.room_id else None

                    title = f"{subject.name if subject else 'Class'} - {section.name if section else ''}".strip(" -")
                    sess = ClassSession(
                        faculty_id=target_faculty_id,
                        section_id=s.section_id,
                        room_id=s.room_id,
                        room_number=room.room_number if room else None,
                        title=title,
                        session_date=curr_date,
                        day_of_week=day_name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        status="SCHEDULED",
                        is_recurring="true"
                    )
                    db.add(sess)
                    created_sessions.append(sess)

        curr_date += delta

    db.commit()
    return {
        "message": f"Successfully generated {len(created_sessions)} calendar sessions from weekly timetable.",
        "generated_count": len(created_sessions)
    }


# =====================================================================
# TIMETABLE IMAGE OCR EXTRACTION & FACULTY CONFIRMATION WORKFLOW
# =====================================================================

@router.post(
    "/extract-timetable",
    response_model=TimetableExtractResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload Timetable Image or PDF and extract schedule entries via OCR"
)
async def extract_timetable(
    file: UploadFile = File(..., description="Timetable image (.png, .jpg, .jpeg, .webp) or PDF (.pdf)"),
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    """
    Parses an uploaded timetable image or PDF using OCR and table extraction heuristics.
    Returns the extracted schedule slots (Day, Period, Times, Subject, Section, Room)
    for faculty confirmation/editing before committing to PostgreSQL.
    """
    allowed_exts = (".png", ".jpg", ".jpeg", ".webp", ".pdf")
    filename = file.filename or "timetable.png"
    if not any(filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Please upload an image ({', '.join(allowed_exts[:4])}) or PDF (.pdf)."
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    extracted = TimetableOCRService.extract_from_file_bytes(content, filename)

    # No defaults are applied here: only values actually read from the document are returned.
    # Empty fields must be completed by the faculty during confirmation/editing.
    slot_models = [
        TimetableExtractedSlot(
            id=f"slot_{idx}",
            day_of_week=s.get("day_of_week", ""),
            period_name=s.get("period_name"),
            start_time=s.get("start_time", ""),
            end_time=s.get("end_time", ""),
            subject_code=s.get("subject_code", ""),
            subject_name=s.get("subject_name", ""),
            section_name=s.get("section_name", ""),
            room_number=s.get("room_number", ""),
            confidence=float(s.get("confidence", 0.0))
        )
        for idx, s in enumerate(extracted)
    ]

    if not slot_models:
        message = (
            "No timetable entries could be extracted from this file (OCR engine unavailable or no "
            "readable table detected). Enter your timetable entries manually below and confirm - "
            "nothing has been invented or pre-filled."
        )
    else:
        incomplete = sum(
            1 for s in slot_models
            if not (s.day_of_week and s.start_time and s.end_time and (s.subject_code or s.subject_name))
        )
        message = (
            f"Extracted {len(slot_models)} candidate timetable slots. "
            + (
                f"{incomplete} slot(s) have missing fields that could not be read - "
                "complete them manually before saving."
                if incomplete
                else "Please review and confirm."
            )
        )

    return TimetableExtractResponse(
        filename=filename,
        total_slots_extracted=len(slot_models),
        slots=slot_models,
        message=message
    )


@router.post(
    "/confirm-timetable",
    response_model=TimetableConfirmResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save faculty-confirmed timetable slots into PostgreSQL"
)
def confirm_timetable(
    data: TimetableConfirmRequest,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    """
    Commits faculty-confirmed/edited timetable slots to PostgreSQL:
    - Creates or updates Subject and Section entities for current faculty
    - Stores FacultySchedule weekly recurring slots
    - Pre-provisions ClassSession for today's classes so Dashboard displays them immediately
    """
    if not data.slots:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot confirm empty timetable. At least one class slot is required."
        )

    if data.clear_existing:
        # Deactivate or remove existing recurring schedules for this faculty
        db.query(FacultySchedule).filter(
            FacultySchedule.faculty_id == current_faculty.id
        ).update({"is_active": False})
        db.commit()

    created_subjects = 0
    created_sections = 0
    saved_slots = 0
    today = date.today()
    today_day_name = today.strftime("%A").upper()

    for s in data.slots:
        day_str = s.day_of_week.strip().upper()
        valid_days = {"MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"}
        if day_str not in valid_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid day_of_week '{s.day_of_week}'. Must be one of: {', '.join(sorted(valid_days))}"
            )
        # Required fields must be present - the prototype never invents timetable values.
        # The subject identity may arrive as a classic course code (23CS301) or, per the
        # faculty's KLU convention, as the subject token (3-CSE-DL / PG2) in subject_name.
        code_clean_req = s.subject_code.strip().upper()
        name_identity = s.subject_name.strip().upper()
        if not code_clean_req and not name_identity:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Every confirmed timetable slot requires a subject (course code or subject)."
            )
        if not s.start_time or not s.end_time:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Every confirmed timetable slot requires start and end times."
            )

        # Parse start and end time
        def parse_t(t_str: str) -> time:
            parts = [int(p) for p in t_str.split(":")[:2]]
            return time(parts[0], parts[1])

        start_t = parse_t(s.start_time)
        end_t = parse_t(s.end_time)

        # 1. Subject: the identity is the course code when present, otherwise the
        #    subject token read from the timetable (3-CSE-DL / PG2). Nothing is invented.
        code_clean = code_clean_req or name_identity
        name_clean = s.subject_name.strip() or code_clean
        subject = db.query(Subject).filter(
            Subject.faculty_id == current_faculty.id,
            Subject.code == code_clean
        ).first()

        if not subject:
            subject = Subject(
                faculty_id=current_faculty.id,
                code=code_clean,
                name=name_clean,
                description=f"Auto-created from verified timetable: {code_clean}"
            )
            db.add(subject)
            db.commit()
            db.refresh(subject)
            created_subjects += 1
        elif subject.name != name_clean and name_clean:
            subject.name = name_clean
            db.commit()

        # 2. Section (as read from the timetable; may be a batch name like PG2 / 3-CSE-DL)
        sec_name = s.section_name.strip()
        if not sec_name:
            sec_name = code_clean  # fall back to the course code itself as the section identity
        section = db.query(Section).filter(
            Section.subject_id == subject.id,
            Section.name == sec_name
        ).first()

        if not section:
            section = Section(
                subject_id=subject.id,
                name=sec_name,
                academic_year=s.academic_year or "2025-2026",
                semester=s.semester or "EVEN"
            )
            db.add(section)
            db.commit()
            db.refresh(section)
            created_sections += 1

        # 3. FacultySchedule
        schedule = FacultySchedule(
            faculty_id=current_faculty.id,
            subject_id=subject.id,
            section_id=section.id,
            day_of_week=day_str,
            start_time=start_t,
            end_time=end_t,
            academic_year=s.academic_year or "2025-2026",
            semester=s.semester or "EVEN",
            is_active=True
        )
        db.add(schedule)
        saved_slots += 1

        # 4. If scheduled for today, ensure ClassSession exists
        if day_str == today_day_name:
            sess = db.query(ClassSession).filter(
                ClassSession.faculty_id == current_faculty.id,
                ClassSession.section_id == section.id,
                ClassSession.session_date == today,
                ClassSession.start_time == start_t
            ).first()
            if not sess:
                title = f"{subject.code}: {subject.name or subject.code} - {section.name}"
                sess = ClassSession(
                    faculty_id=current_faculty.id,
                    section_id=section.id,
                    room_number=s.room_number or None,
                    title=title,
                    session_date=today,
                    day_of_week=day_str,
                    start_time=start_t,
                    end_time=end_t,
                    status="SCHEDULED",
                    is_recurring="true"
                )
                db.add(sess)

    db.commit()

    AuditService.log_event(
        db=db,
        action="TIMETABLE_CONFIRMED",
        resource_type="FACULTY_SCHEDULE",
        user=current_faculty,
        details={
            "saved_slots_count": saved_slots,
            "created_subjects_count": created_subjects,
            "created_sections_count": created_sections
        }
    )

    return TimetableConfirmResponse(
        saved_slots_count=saved_slots,
        created_subjects_count=created_subjects,
        created_sections_count=created_sections,
        message=f"Timetable saved successfully! {saved_slots} recurring classes registered in PostgreSQL."
    )

