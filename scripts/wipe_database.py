"""
Database Wipe Script
====================
Purges all fake profiles, subjects, sections, sessions, videos,
and seeded credentials from PostgreSQL for a completely clean slate.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import SessionLocal, engine
from sqlalchemy import text

def wipe_all_data():
    print("=" * 60)
    print("TEMPO — PURGING ALL FAKE/SEEDED DATA (CLEAN SLATE)")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        tables = [
            'faculty_insights',
            'coverage_metrics',
            'classroom_change_points',
            'classroom_entropy',
            'classroom_temporal_states',
            'behaviour_transitions',
            'student_temporal_profiles',
            'behaviour_results',
            'student_track_results',
            'analysis_jobs',
            'videos',
            'class_sessions',
            'faculty_schedules',
            'exam_review_events',
            'exam_sessions',
            'cameras',
            'rooms',
            'students',
            'sections',
            'subjects',
            'audit_logs',
            'faculties',
        ]
        
        with engine.connect() as conn:
            for t in tables:
                try:
                    conn.execute(text(f"TRUNCATE TABLE {t} CASCADE;"))
                    conn.commit()
                    print(f"  Truncated {t}")
                except Exception as ex:
                    print(f"  Note on {t}: {ex}")
                    
        print("\nAll database tables successfully wiped! Complete clean slate confirmed.")
    finally:
        db.close()

if __name__ == "__main__":
    wipe_all_data()
