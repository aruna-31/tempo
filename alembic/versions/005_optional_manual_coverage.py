"""Allow coverage metrics without a supplied manual reference."""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "005_optional_manual_coverage"
down_revision: Union[str, None] = "004_hierarchical_temporal"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("coverage_metrics", "detection_coverage", existing_type=sa.Float(), nullable=True)


def downgrade() -> None:
    op.alter_column("coverage_metrics", "detection_coverage", existing_type=sa.Float(), nullable=False)