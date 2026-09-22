# Import all models so Alembic can see them through Base.metadata
from app.db.session import Base  # noqa
from app.models.faculty import Faculty  # noqa
from app.models.academic import Subject, Section, Student  # noqa
from app.models.session import ClassSession  # noqa
from app.models.video import Video  # noqa
from app.models.analysis import AnalysisJob, StudentTrackResult, BehaviourResult  # noqa
