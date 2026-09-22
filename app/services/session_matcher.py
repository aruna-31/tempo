import logging
import uuid
from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple
from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.academic import Section, Subject
from app.models.faculty import Faculty
from app.models.room import Camera, Room
from app.models.schedule import FacultySchedule
from app.models.session import ClassSession

logger = logging.getLogger("tempo.services.session_matcher")


class SessionMatcherService:
    @staticmethod
    def match_recording_to_session(
        db: Session,
        recording_date: Optional[date] = None,
        recording_start: Optional[time] = None,
        recording_end: Optional[time] = None,
        room_number: Optional[str] = None,
        device_code: Optional[str] = None,
        faculty_email: Optional[str] = None,
        tolerance_minutes: Optional[int] = None
    ) -> Tuple[Optional[ClassSession], str, float, str]:
        """
        Matches a completed video recording to the corresponding scheduled ClassSession.
        
        Returns:
            (session, match_type, confidence_score, explanation)
            match_type: "EXACT_MATCH", "TOLERANCE_MATCH", "SCHEDULE_INSTANTIATED", "UNMATCHED"
        """
        if tolerance_minutes is None:
            tolerance_minutes = settings.SESSION_MATCH_TOLERANCE_MINUTES

        rec_date = recording_date or date.today()
        rec_start = recording_start or datetime.now().time()
        
        # 1. Resolve room if device_code or room_number provided
        room: Optional[Room] = None
        if device_code:
            camera = db.query(Camera).filter(Camera.device_code == device_code).first()
            if camera and camera.room:
                room = camera.room
        
        if not room and room_number:
            clean_num = room_number.strip().replace("Room ", "").replace("room ", "")
            room = db.query(Room).filter(
                or_(
                    Room.room_number == room_number,
                    Room.room_number == clean_num,
                    Room.room_number.ilike(f"%{clean_num}%")
                )
            ).first()

        # 2. Query existing scheduled ClassSessions for that date
        query = db.query(ClassSession).filter(ClassSession.session_date == rec_date)
        if room:
            query = query.filter(
                or_(
                    ClassSession.room_id == room.id,
                    ClassSession.room_number == room.room_number
                )
            )
        elif room_number:
            query = query.filter(ClassSession.room_number == room_number)

        candidate_sessions = query.all()

        best_session: Optional[ClassSession] = None
        best_diff_minutes = float("inf")
        
        rec_start_dt = datetime.combine(rec_date, rec_start)

        for sess in candidate_sessions:
            sess_start_dt = datetime.combine(sess.session_date, sess.start_time)
            diff_mins = abs((sess_start_dt - rec_start_dt).total_seconds()) / 60.0
            
            if diff_mins < best_diff_minutes:
                best_diff_minutes = diff_mins
                best_session = sess

        # Check if match is within tolerance window
        if best_session and best_diff_minutes <= tolerance_minutes:
            match_type = "EXACT_MATCH" if best_diff_minutes <= 2.0 else "TOLERANCE_MATCH"
            confidence = max(0.5, round(1.0 - (best_diff_minutes / (tolerance_minutes * 2)), 2))
            explanation = (
                f"Matched session '{best_session.title}' (ID: {best_session.id}) "
                f"scheduled at {best_session.start_time} (Delta: {best_diff_minutes:.1f} mins, Room: {best_session.room_number or (room.room_number if room else 'N/A')})"
            )
            return best_session, match_type, confidence, explanation

        # 3. Fallback: Check weekly recurring FacultySchedule if no session exists for today
        day_name = rec_date.strftime("%A").upper()  # MONDAY, TUESDAY...
        sched_query = db.query(FacultySchedule).filter(
            FacultySchedule.day_of_week == day_name,
            FacultySchedule.is_active == True
        )
        if room:
            sched_query = sched_query.filter(FacultySchedule.room_id == room.id)
            
        candidate_schedules = sched_query.all()
        best_sched = None
        best_sched_diff = float("inf")
        
        for sched in candidate_schedules:
            sched_dt = datetime.combine(rec_date, sched.start_time)
            diff_mins = abs((sched_dt - rec_start_dt).total_seconds()) / 60.0
            if diff_mins < best_sched_diff:
                best_sched_diff = diff_mins
                best_sched = sched

        if best_sched and best_sched_diff <= tolerance_minutes:
            # Automatically instantiate the ClassSession from schedule
            subject = db.query(Subject).filter(Subject.id == best_sched.subject_id).first()
            section = db.query(Section).filter(Section.id == best_sched.section_id).first()
            sub_name = subject.name if subject else "Class Session"
            sec_name = section.name if section else ""
            
            new_session = ClassSession(
                faculty_id=best_sched.faculty_id,
                section_id=best_sched.section_id,
                room_id=best_sched.room_id,
                room_number=room.room_number if room else None,
                title=f"{sub_name} - {sec_name}".strip(" -"),
                session_date=rec_date,
                day_of_week=day_name,
                start_time=best_sched.start_time,
                end_time=best_sched.end_time,
                status="SCHEDULED"
            )
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            
            confidence = max(0.6, round(1.0 - (best_sched_diff / (tolerance_minutes * 2)), 2))
            explanation = (
                f"Auto-instantiated ClassSession from recurring timetable schedule '{best_sched.day_of_week} "
                f"{best_sched.start_time}' for faculty {best_sched.faculty_id}"
            )
            return new_session, "SCHEDULE_INSTANTIATED", confidence, explanation

        return None, "UNMATCHED", 0.0, f"No matching scheduled session or timetable found for date {rec_date} at {rec_start} in room {room_number or (room.room_number if room else 'N/A')}"
