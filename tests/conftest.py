import uuid
import pytest
from typing import Generator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.security import create_access_token, get_password_hash
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.faculty import Faculty
from app.services.ml_connector import set_session_factory
from app.services.job_manager import set_session_factory as set_job_session_factory

# Disable auto background ML dispatch during synchronous API unit testing.
settings.AUTO_DISPATCH_ML_JOB = False

# ---------------------------------------------------------------------------
# Tests run against a REAL PostgreSQL database (mandatory in this project).
# No SQLite fallback anywhere: a reachable PostgreSQL server is required to run
# the test suite, exactly as in production.
# ---------------------------------------------------------------------------
TEST_DB_NAME = "tempo_test_db"

_admin_url = settings.DATABASE_URL.rsplit("/", 1)[0] + "/postgres"
_admin_engine = create_engine(_admin_url, poolclass=NullPool, isolation_level="AUTOCOMMIT")

with _admin_engine.connect() as conn:
    exists = conn.execute(
        text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DB_NAME}
    ).scalar()
    if not exists:
        conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))

# Recreate from scratch on every test run for a deterministic state
_test_url = settings.DATABASE_URL.rsplit("/", 1)[0] + "/" + TEST_DB_NAME
_cleanup_engine = create_engine(_test_url, poolclass=NullPool, isolation_level="AUTOCOMMIT")
with _cleanup_engine.connect() as conn:
    conn.execute(text(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = :name AND pid <> pg_backend_pid()"
    ), {"name": TEST_DB_NAME})
    conn.execute(text('DROP SCHEMA public CASCADE; CREATE SCHEMA public;'))
_cleanup_engine.dispose()

test_engine = create_engine(_test_url, pool_pre_ping=True, poolclass=NullPool)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)

set_session_factory(TestingSessionLocal)
set_job_session_factory(TestingSessionLocal)


@pytest.fixture(autouse=True)
def clean_db():
    """Ensure clean tables before every test."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def faculty_a(db_session: Session) -> Faculty:
    faculty = Faculty(
        id=uuid.uuid4(),
        email="prof.sharma@klu.ac.in",
        hashed_password=get_password_hash("SecretPassword123!"),
        full_name="Dr. Sharma",
        department="Computer Science and Engineering",
        designation="Professor",
        is_active=True
    )
    db_session.add(faculty)
    db_session.commit()
    db_session.refresh(faculty)
    return faculty


@pytest.fixture
def faculty_b(db_session: Session) -> Faculty:
    faculty = Faculty(
        id=uuid.uuid4(),
        email="prof.verma@klu.ac.in",
        hashed_password=get_password_hash("SecretPassword123!"),
        full_name="Dr. Verma",
        department="Electronics and Communication",
        designation="Associate Professor",
        is_active=True
    )
    db_session.add(faculty)
    db_session.commit()
    db_session.refresh(faculty)
    return faculty


@pytest.fixture
def auth_headers_faculty_a(faculty_a: Faculty) -> dict:
    token = create_access_token(
        subject=str(faculty_a.id),
        extra_claims={"email": faculty_a.email, "name": faculty_a.full_name}
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_faculty_b(faculty_b: Faculty) -> dict:
    token = create_access_token(
        subject=str(faculty_b.id),
        extra_claims={"email": faculty_b.email, "name": faculty_b.full_name}
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_faculty(db_session: Session) -> Faculty:
    faculty = Faculty(
        id=uuid.uuid4(),
        email="admin.dean@klu.ac.in",
        hashed_password=get_password_hash("AdminPass123!"),
        full_name="Dean of Academic Affairs",
        department="Administration",
        designation="Dean",
        role="ADMIN",
        is_superuser=True,
        is_active=True
    )
    db_session.add(faculty)
    db_session.commit()
    db_session.refresh(faculty)
    return faculty


@pytest.fixture
def auth_headers_admin(admin_faculty: Faculty) -> dict:
    token = create_access_token(
        subject=str(admin_faculty.id),
        extra_claims={"email": admin_faculty.email, "name": admin_faculty.full_name}
    )
    return {"Authorization": f"Bearer {token}"}
