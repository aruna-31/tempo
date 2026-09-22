"""Add faculty_insights table for post-class faculty instructional recommendations."""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "006_add_faculty_insights"
down_revision: Union[str, None] = "005_optional_manual_coverage"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "faculty_insights",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column("job_id", sa.Uuid(as_uuid=True), sa.ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("observation", sa.Text(), nullable=False),
        sa.Column("pedagogical_context", sa.Text(), nullable=False),
        sa.Column("suggested_action", sa.Text(), nullable=False),
        sa.Column("coverage_context", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_faculty_insights_job_id", "faculty_insights", ["job_id"])
    op.create_index("ix_faculty_insights_category", "faculty_insights", ["category"])


def downgrade() -> None:
    op.drop_index("ix_faculty_insights_category", table_name="faculty_insights")
    op.drop_index("ix_faculty_insights_job_id", table_name="faculty_insights")
    op.drop_table("faculty_insights")
