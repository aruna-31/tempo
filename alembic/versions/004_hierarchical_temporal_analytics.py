"""Add hierarchical temporal analytics tables.

Revision ID: 004_hierarchical_temporal_analytics
Revises: 003_session_job_cols
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "004_hierarchical_temporal"
down_revision: Union[str, None] = "003_session_job_cols"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None
json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def _table(name, columns):
    op.create_table(name, *columns)
    op.create_index(f"ix_{name}_job_id", name, ["job_id"])


def upgrade() -> None:
    fk = lambda: sa.ForeignKey("analysis_jobs.id", ondelete="CASCADE")
    _table("student_temporal_profiles", [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), fk(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("timeline_json", json_type, nullable=False),
        sa.Column("behaviour_distribution_json", json_type, nullable=False),
        sa.Column("transition_matrix_json", json_type, nullable=False),
        sa.Column("behaviour_duration_json", json_type, nullable=False),
        sa.Column("total_observed_duration", sa.Float(), nullable=False),
        sa.Column("segment_count", sa.Integer(), nullable=False),
        sa.Column("temporal_coverage", sa.Float(), nullable=False),
    ])
    _table("behaviour_transitions", [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), fk(), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=True),
        sa.Column("from_behaviour", sa.String(50), nullable=False),
        sa.Column("to_behaviour", sa.String(50), nullable=False),
        sa.Column("count", sa.Integer(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
    ])
    _table("classroom_temporal_states", [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), fk(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("state", sa.String(80), nullable=False),
        sa.Column("behaviour_distribution_json", json_type, nullable=False),
        sa.Column("students_contributing", sa.Integer(), nullable=False),
    ])
    _table("classroom_entropy", [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), fk(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("entropy", sa.Float(), nullable=False),
        sa.Column("behaviour_distribution_json", json_type, nullable=False),
    ])
    _table("classroom_change_points", [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), fk(), nullable=False),
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("previous_distribution_json", json_type, nullable=False),
        sa.Column("new_distribution_json", json_type, nullable=False),
        sa.Column("change_score", sa.Float(), nullable=False),
        sa.Column("previous_state", sa.String(80), nullable=False),
        sa.Column("new_state", sa.String(80), nullable=False),
    ])
    _table("coverage_metrics", [
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), fk(), nullable=False, unique=True),
        sa.Column("manual_reference_student_count", sa.Integer(), nullable=True),
        sa.Column("detected_student_count", sa.Integer(), nullable=False),
        sa.Column("tracked_student_count", sa.Integer(), nullable=False),
        sa.Column("temporal_ready_track_count", sa.Integer(), nullable=False),
        sa.Column("detection_coverage", sa.Float(), nullable=True),
        sa.Column("tracking_coverage", sa.Float(), nullable=False),
        sa.Column("temporal_coverage", sa.Float(), nullable=False),
    ])


def downgrade() -> None:
    for name in ("coverage_metrics", "classroom_change_points", "classroom_entropy", "classroom_temporal_states", "behaviour_transitions", "student_temporal_profiles"):
        op.drop_index(f"ix_{name}_job_id", table_name=name)
        op.drop_table(name)