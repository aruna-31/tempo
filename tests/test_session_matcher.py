import uuid
from datetime import date, time
import pytest
from sqlalchemy.orm import Session

from app.models.academic import Section, Subject
from app.models.faculty import Faculty
from app.models.room import Camera, Room
from app.models.schedule import FacultySchedule
from app.models.session import ClassSession
from app.services.session_matcher import SessionMatcherService


def test_session_matcher_exact_match(db_session: Session, faculty_a: Faculty):
    # Setup Room, Subject, Section, and scheduled ClassSession
    room = Room(room_number="301", building="Block A", floor=3, capacity=60)
    db_session.add(room)
    db_session.commit()

    subject = Subject(code="CS101", name="Data Structures", faculty_id=faculty_a.id)
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Section A", academic_year="2025-2026", semester="ODD")
    db_session.add(section)
    db_session.commit()

    today = date.today()
    session = ClassSession(
        faculty_id=faculty_a.id,
        section_id=section.id,
        room_id=room.id,
        room_number="301",
        title="CS101 - Lecture 05",
        session_date=today,
        start_time=time(10, 0, 0),
        end_time=time(10, 50, 0),
        status="SCHEDULED"
    )
    db_session.add(session)
    db_session.commit()

    # Match recording arriving at 10:01:00 for Room 301
    matched, match_type, conf, exp = SessionMatcherService.match_recording_to_session(
        db=db_session,
        recording_date=today,
        recording_start=time(10, 1, 0),
        room_number="301",
        tolerance_minutes=15
    )

    assert matched is not None
    assert matched.id == session.id
    assert match_type == "EXACT_MATCH"
    assert conf >= 0.90


def test_session_matcher_tolerance_match(db_session: Session, faculty_a: Faculty):
    room = Room(room_number="402", building="Block B", floor=4, capacity=50)
    db_session.add(room)
    db_session.commit()

    camera = Camera(room_id=room.id, camera_name="Front Cam", device_code="CAM-402-FRONT")
    db_session.add(camera)
    db_session.commit()

    subject = Subject(code="CS102", name="Algorithms", faculty_id=faculty_a.id)
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Section B", academic_year="2025-2026", semester="ODD")
    db_session.add(section)
    db_session.commit()

    today = date.today()
    session = ClassSession(
        faculty_id=faculty_a.id,
        section_id=section.id,
        room_id=room.id,
        room_number="402",
        title="CS102 - Algorithms",
        session_date=today,
        start_time=time(14, 0, 0),
        end_time=time(14, 50, 0),
        status="SCHEDULED"
    )
    db_session.add(session)
    db_session.commit()

    # Recording starts 8 minutes late (14:08:00) using camera device_code
    matched, match_type, conf, exp = SessionMatcherService.match_recording_to_session(
        db=db_session,
        recording_date=today,
        recording_start=time(14, 8, 0),
        device_code="CAM-402-FRONT",
        tolerance_minutes=15
    )

    assert matched is not None
    assert matched.id == session.id
    assert match_type == "TOLERANCE_MATCH"


def test_session_matcher_recurring_schedule_fallback(db_session: Session, faculty_a: Faculty):
    room = Room(room_number="505", building="Block C", floor=5, capacity=70)
    db_session.add(room)
    db_session.commit()

    subject = Subject(code="CS201", name="Operating Systems", faculty_id=faculty_a.id)
    db_session.add(subject)
    db_session.commit()

    section = Section(subject_id=subject.id, name="Section C", academic_year="2025-2026", semester="ODD")
    db_session.add(section)
    db_session.commit()

    today = date.today()
    day_name = today.strftime("%A").upper()

    schedule = FacultySchedule(
        faculty_id=faculty_a.id,
        subject_id=subject.id,
        section_id=section.id,
        room_id=room.id,
        day_of_week=day_name,
        start_time=time(11, 0, 0),
        end_time=time(11, 50, 0),
        is_active=True
    )
    db_session.add(schedule)
    db_session.commit()

    # No ClassSession previously existed in DB for today
    matched, match_type, conf, exp = SessionMatcherService.match_recording_to_session(
        db=db_session,
        recording_date=today,
        recording_start=time(11, 2, 0),
        room_number="505",
        tolerance_minutes=15
    )

    assert matched is not None
    assert match_type == "SCHEDULE_INSTANTIATED"
    assert matched.faculty_id == faculty_a.id
    assert matched.section_id == section.id


def test_session_matcher_out_of_tolerance(db_session: Session, faculty_a: Faculty):
    today = date.today()
    # Query non-existent room/session with 40 mins gap
    matched, match_type, conf, exp = SessionMatcherService.match_recording_to_session(
        db=db_session,
        recording_date=today,
        recording_start=time(23, 0, 0),
        room_number="999",
        tolerance_minutes=15
    )

    assert matched is None
    assert match_type == "UNMATCHED"
