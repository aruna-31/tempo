"""Add exam_sessions, exam_review_events tables and camera source fields."""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "007_exam_mode_cameras"
down_revision: Union[str, None] = "006_add_faculty_insights"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add source_type and demo_video_path to cameras table
    op.add_column("cameras", sa.Column("source_type", sa.String(50), nullable=False, server_default="MP4_DEMO"))
    op.add_column("cameras", sa.Column("demo_video_path", sa.String(255), nullable=True))

    # 2. Create exam_sessions table
    op.create_table(
        "exam_sessions",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("session_code", sa.String(50), nullable=False, unique=True),
        sa.Column("title", sa.String(150), nullable=False),
        sa.Column("room_id", sa.Uuid(as_uuid=True), sa.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False),
        sa.Column("camera_id", sa.Uuid(as_uuid=True), sa.ForeignKey("cameras.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="SCHEDULED"),
        sa.Column("expected_students", sa.Integer(), nullable=False, server_default="70"),
        sa.Column("detected_students_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("coverage_ratio", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("uncertainty_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("room_monitoring_status", sa.String(50), nullable=False, server_default="IDLE"),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("invigilator_name", sa.String(100), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_exam_sessions_session_code", "exam_sessions", ["session_code"])
    op.create_index("ix_exam_sessions_room_id", "exam_sessions", ["room_id"])
    op.create_index("ix_exam_sessions_camera_id", "exam_sessions", ["camera_id"])

    # 3. Create exam_review_events table
    op.create_table(
        "exam_review_events",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("exam_session_id", sa.Uuid(as_uuid=True), sa.ForeignKey("exam_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("anonymous_student_id", sa.String(50), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("event_description", sa.String(255), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.85"),
        sa.Column("start_time_offset", sa.Float(), nullable=False),
        sa.Column("end_time_offset", sa.Float(), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("observation_coverage", sa.Float(), nullable=False, server_default="0.90"),
        sa.Column("observation_uncertainty", sa.Float(), nullable=False, server_default="0.10"),
        sa.Column("spatial_zone", sa.String(50), nullable=False, server_default="DESK_AREA"),
        sa.Column("bounding_box", sa.JSON(), nullable=True),
        sa.Column("evidence_window_start", sa.Float(), nullable=True),
        sa.Column("evidence_window_end", sa.Float(), nullable=True),
        sa.Column("evidence_clip_path", sa.String(255), nullable=True),
        sa.Column("supporting_observations", sa.JSON(), nullable=True),
        sa.Column("review_status", sa.String(50), nullable=False, server_default="PENDING_REVIEW"),
        sa.Column("reviewed_by", sa.String(100), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_exam_review_events_exam_session_id", "exam_review_events", ["exam_session_id"])
    op.create_index("ix_exam_review_events_anonymous_student_id", "exam_review_events", ["anonymous_student_id"])
    op.create_index("ix_exam_review_events_event_type", "exam_review_events", ["event_type"])
    op.create_index("ix_exam_review_events_review_status", "exam_review_events", ["review_status"])


def downgrade() -> None:
    op.drop_index("ix_exam_review_events_review_status", table_name="exam_review_events")
    op.drop_index("ix_exam_review_events_event_type", table_name="exam_review_events")
    op.drop_index("ix_exam_review_events_anonymous_student_id", table_name="exam_review_events")
    op.drop_index("ix_exam_review_events_exam_session_id", table_name="exam_review_events")
    op.drop_table("exam_review_events")

    op.drop_index("ix_exam_sessions_camera_id", table_name="exam_sessions")
    op.drop_index("ix_exam_sessions_room_id", table_name="exam_sessions")
    op.drop_index("ix_exam_sessions_session_code", table_name="exam_sessions")
    op.drop_table("exam_sessions")

    op.drop_column("cameras", "demo_video_path")
    op.drop_column("cameras", "source_type")
