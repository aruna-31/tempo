from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.core.config import settings

# Enforce PostgreSQL engine
DATABASE_URL = settings.DATABASE_URL

# SQLAlchemy PostgreSQL Engine with connection pooling and pre-ping
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    echo=False
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a transactional database session.
    Closes automatically after request lifecycle.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
