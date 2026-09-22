"""Initial PostgreSQL schema for Classroom Temporal Behaviour Analytics

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-09-15 22:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. faculties table
    op.create_table(
        "faculties",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=150), nullable=False),
        sa.Column("department", sa.String(length=100), nullable=False),
        sa.Column("designation", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_faculties_id", "faculties", ["id"], unique=False)
    op.create_index("ix_faculties_email", "faculties", ["email"], unique=True)

    # 2. subjects table
    op.create_table(
        "subjects",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("faculty_id", sa.Uuid(), sa.ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False),
        sa.Column("code", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_subjects_id", "subjects", ["id"], unique=False)
    op.create_index("ix_subjects_faculty_id", "subjects", ["faculty_id"], unique=False)
    op.create_index("ix_subjects_code", "subjects", ["code"], unique=False)

    # 3. sections table
    op.create_table(
        "sections",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("subject_id", sa.Uuid(), sa.ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("academic_year", sa.String(length=20), nullable=False),
        sa.Column("semester", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_sections_id", "sections", ["id"], unique=False)
    op.create_index("ix_sections_subject_id", "sections", ["subject_id"], unique=False)

    # 4. students table
    op.create_table(
        "students",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("section_id", sa.Uuid(), sa.ForeignKey("sections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("roll_number", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=150), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_students_id", "students", ["id"], unique=False)
    op.create_index("ix_students_section_id", "students", ["section_id"], unique=False)
    op.create_index("ix_students_roll_number", "students", ["roll_number"], unique=False)

    # 5. class_sessions table
    op.create_table(
        "class_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("faculty_id", sa.Uuid(), sa.ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False),
        sa.Column("section_id", sa.Uuid(), sa.ForeignKey("sections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("start_time", sa.Time(), nullable=False),
        sa.Column("end_time", sa.Time(), nullable=False),
        sa.Column("room_number", sa.String(length=50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_class_sessions_id", "class_sessions", ["id"], unique=False)
    op.create_index("ix_class_sessions_faculty_id", "class_sessions", ["faculty_id"], unique=False)
    op.create_index("ix_class_sessions_section_id", "class_sessions", ["section_id"], unique=False)
    op.create_index("ix_class_sessions_session_date", "class_sessions", ["session_date"], unique=False)

    # 6. videos table
    op.create_table(
        "videos",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("session_id", sa.Uuid(), sa.ForeignKey("class_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("faculty_id", sa.Uuid(), sa.ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("content_type", sa.String(length=100), server_default="video/mp4", nullable=False),
        sa.Column("output_video_path", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=50), server_default="UPLOADED", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_videos_id", "videos", ["id"], unique=False)
    op.create_index("ix_videos_session_id", "videos", ["session_id"], unique=False)
    op.create_index("ix_videos_faculty_id", "videos", ["faculty_id"], unique=False)

    # 7. analysis_jobs table
    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("video_id", sa.Uuid(), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=50), server_default="PENDING", nullable=False),
        sa.Column("progress_pct", sa.Integer(), server_default="0", nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("config_json", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_analysis_jobs_id", "analysis_jobs", ["id"], unique=False)
    op.create_index("ix_analysis_jobs_video_id", "analysis_jobs", ["video_id"], unique=False)

    # 8. student_track_results table
    op.create_table(
        "student_track_results",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("job_id", sa.Uuid(), sa.ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("student_id", sa.Uuid(), sa.ForeignKey("students.id", ondelete="SET NULL"), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("bounding_box_history", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.Column("start_frame", sa.Integer(), nullable=False),
        sa.Column("end_frame", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_student_track_results_id", "student_track_results", ["id"], unique=False)
    op.create_index("ix_student_track_results_job_id", "student_track_results", ["job_id"], unique=False)
    op.create_index("ix_student_track_results_student_id", "student_track_results", ["student_id"], unique=False)
    op.create_index("ix_student_track_results_track_id", "student_track_results", ["track_id"], unique=False)

    # 9. behaviour_results table
    op.create_table(
        "behaviour_results",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("job_id", sa.Uuid(), sa.ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("student_id", sa.Uuid(), sa.ForeignKey("students.id", ondelete="SET NULL"), nullable=True),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("frame_number", sa.Integer(), nullable=False),
        sa.Column("timestamp_seconds", sa.Float(), nullable=False),
        sa.Column("behaviour_type", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("metadata_json", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_behaviour_results_id", "behaviour_results", ["id"], unique=False)
    op.create_index("ix_behaviour_results_job_id", "behaviour_results", ["job_id"], unique=False)
    op.create_index("ix_behaviour_results_student_id", "behaviour_results", ["student_id"], unique=False)
    op.create_index("ix_behaviour_results_track_id", "behaviour_results", ["track_id"], unique=False)
    op.create_index("ix_behaviour_results_frame_number", "behaviour_results", ["frame_number"], unique=False)
    op.create_index("ix_behaviour_results_timestamp_seconds", "behaviour_results", ["timestamp_seconds"], unique=False)
    op.create_index("ix_behaviour_results_behaviour_type", "behaviour_results", ["behaviour_type"], unique=False)


def downgrade() -> None:
    op.drop_table("behaviour_results")
    op.drop_table("student_track_results")
    op.drop_table("analysis_jobs")
    op.drop_table("videos")
    op.drop_table("class_sessions")
    op.drop_table("students")
    op.drop_table("sections")
    op.drop_table("subjects")
    op.drop_table("faculties")
