"""Add enterprise and role columns to PostgreSQL schema

Revision ID: 002_add_enterprise_and_role_columns
Revises: 001_initial_schema
Create Date: 2026-09-16 10:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "002_enterprise_roles"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = set(inspector.get_table_names())
    faculty_cols = {c["name"] for c in inspector.get_columns("faculties")}

    # 1. Add missing columns to faculties table
    if "role" not in faculty_cols:
        op.add_column("faculties", sa.Column("role", sa.String(length=50), server_default="FACULTY", nullable=False))
    if "is_superuser" not in faculty_cols:
        op.add_column("faculties", sa.Column("is_superuser", sa.Boolean(), server_default=sa.text("false"), nullable=False))

    # 2. rooms table
    if "rooms" not in existing_tables:
        op.create_table(
            "rooms",
            sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
            sa.Column("name", sa.String(length=100), nullable=False),
            sa.Column("building", sa.String(length=100), nullable=False),
            sa.Column("floor", sa.Integer(), nullable=True),
            sa.Column("capacity", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_rooms_id", "rooms", ["id"], unique=False)
        op.create_index("ix_rooms_name", "rooms", ["name"], unique=True)

    # 3. cameras table
    if "cameras" not in existing_tables:
        op.create_table(
            "cameras",
            sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
            sa.Column("room_id", sa.Uuid(), sa.ForeignKey("rooms.id", ondelete="CASCADE"), nullable=False),
            sa.Column("camera_name", sa.String(length=100), nullable=False),
            sa.Column("ip_address", sa.String(length=50), nullable=True),
            sa.Column("stream_url", sa.String(length=500), nullable=True),
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_cameras_id", "cameras", ["id"], unique=False)
        op.create_index("ix_cameras_room_id", "cameras", ["room_id"], unique=False)

    # 4. faculty_schedules table
    if "faculty_schedules" not in existing_tables:
        op.create_table(
            "faculty_schedules",
            sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
            sa.Column("faculty_id", sa.Uuid(), sa.ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False),
            sa.Column("subject_id", sa.Uuid(), sa.ForeignKey("subjects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("section_id", sa.Uuid(), sa.ForeignKey("sections.id", ondelete="CASCADE"), nullable=False),
            sa.Column("room_id", sa.Uuid(), sa.ForeignKey("rooms.id", ondelete="SET NULL"), nullable=True),
            sa.Column("day_of_week", sa.Integer(), nullable=False),
            sa.Column("start_time", sa.Time(), nullable=False),
            sa.Column("end_time", sa.Time(), nullable=False),
            sa.Column("academic_year", sa.String(length=20), nullable=False),
            sa.Column("semester", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_faculty_schedules_id", "faculty_schedules", ["id"], unique=False)
        op.create_index("ix_faculty_schedules_faculty_id", "faculty_schedules", ["faculty_id"], unique=False)

    # 5. audit_logs table
    if "audit_logs" not in existing_tables:
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
            sa.Column("faculty_id", sa.Uuid(), sa.ForeignKey("faculties.id", ondelete="SET NULL"), nullable=True),
            sa.Column("action", sa.String(length=100), nullable=False),
            sa.Column("entity_type", sa.String(length=100), nullable=False),
            sa.Column("entity_id", sa.String(length=100), nullable=True),
            sa.Column("details", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), server_default="{}", nullable=False),
            sa.Column("ip_address", sa.String(length=50), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        )
        op.create_index("ix_audit_logs_id", "audit_logs", ["id"], unique=False)
        op.create_index("ix_audit_logs_faculty_id", "audit_logs", ["faculty_id"], unique=False)


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("faculty_schedules")
    op.drop_table("cameras")
    op.drop_table("rooms")
    op.drop_column("faculties", "is_superuser")
    op.drop_column("faculties", "role")
