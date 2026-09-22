"""Add missing columns to class_sessions and analysis_jobs

Revision ID: 003_session_job_cols
Revises: 002_enterprise_roles
Create Date: 2026-09-16 11:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "003_session_job_cols"
down_revision: Union[str, None] = "002_enterprise_roles"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    # 1. class_sessions columns
    session_cols = {c["name"] for c in inspector.get_columns("class_sessions")}
    if "room_id" not in session_cols:
        op.add_column("class_sessions", sa.Column("room_id", sa.Uuid(), sa.ForeignKey("rooms.id", ondelete="SET NULL"), nullable=True))
        op.create_index("ix_class_sessions_room_id", "class_sessions", ["room_id"], unique=False)
    if "day_of_week" not in session_cols:
        op.add_column("class_sessions", sa.Column("day_of_week", sa.String(length=20), nullable=True))
    if "status" not in session_cols:
        op.add_column("class_sessions", sa.Column("status", sa.String(length=50), server_default="SCHEDULED", nullable=False))
    if "is_recurring" not in session_cols:
        op.add_column("class_sessions", sa.Column("is_recurring", sa.String(length=10), server_default="false", nullable=False))

    # 2. analysis_jobs columns
    job_cols = {c["name"] for c in inspector.get_columns("analysis_jobs")}
    if "source_type" not in job_cols:
        op.add_column("analysis_jobs", sa.Column("source_type", sa.String(length=50), server_default="MANUAL_UPLOAD", nullable=False))
    if "current_stage" not in job_cols:
        op.add_column("analysis_jobs", sa.Column("current_stage", sa.String(length=50), server_default="INITIALIZING", nullable=False))
    if "retry_count" not in job_cols:
        op.add_column("analysis_jobs", sa.Column("retry_count", sa.Integer(), server_default="0", nullable=False))
    if "max_retries" not in job_cols:
        op.add_column("analysis_jobs", sa.Column("max_retries", sa.Integer(), server_default="3", nullable=False))
    if "execution_time_seconds" not in job_cols:
        op.add_column("analysis_jobs", sa.Column("execution_time_seconds", sa.Float(), nullable=True))
    if "processing_logs" not in job_cols:
        op.add_column("analysis_jobs", sa.Column("processing_logs", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), server_default="[]", nullable=False))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "processing_logs")
    op.drop_column("analysis_jobs", "execution_time_seconds")
    op.drop_column("analysis_jobs", "max_retries")
    op.drop_column("analysis_jobs", "retry_count")
    op.drop_column("analysis_jobs", "current_stage")
    op.drop_column("analysis_jobs", "source_type")

    op.drop_index("ix_class_sessions_room_id", table_name="class_sessions")
    op.drop_column("class_sessions", "is_recurring")
    op.drop_column("class_sessions", "status")
    op.drop_column("class_sessions", "day_of_week")
    op.drop_column("class_sessions", "room_id")
