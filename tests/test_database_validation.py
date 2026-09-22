import pytest
from app.core.config import Settings


def test_postgresql_validation_success():
    valid_urls = [
        "postgresql://postgres:password@localhost:5432/tempo_db",
        "postgresql+psycopg2://admin:secret@db.internal:5432/analytics",
        "postgresql+asyncpg://user:pass@127.0.0.1:5432/tempo",
        "postgres://user:pass@localhost:5432/tempo"  # Cloud standardized
    ]
    for url in valid_urls:
        s = Settings(DATABASE_URL=url)
        assert s.DATABASE_URL.startswith("postgresql")


def test_sqlite_and_invalid_database_url_strictly_fails():
    invalid_urls = [
        "sqlite:///./test.db",
        "sqlite:///:memory:",
        "mysql://user:pass@localhost/db",
        "mongodb://localhost:27017",
        "",
    ]
    for invalid_url in invalid_urls:
        with pytest.raises(ValueError) as excinfo:
            Settings(DATABASE_URL=invalid_url)
        assert "PostgreSQL is" in str(excinfo.value) or "DATABASE_URL is required" in str(excinfo.value)
