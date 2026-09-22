import os
from typing import List, Set, Union
from pydantic import field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    APP_NAME: str = "Classroom Temporal Behaviour Analytics API"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # Strict PostgreSQL Database URL
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/tempo_db"

    # Security & JWT
    JWT_SECRET_KEY: str = "temporary_secret_key_change_in_production_abcdef1234567890"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Faculty Domain Restriction
    ALLOWED_EMAIL_DOMAIN: str = "klu.ac.in"

    # File & Object Storage Configuration
    STORAGE_BASE_DIR: str = "./storage"
    UPLOAD_RAW_DIR: str = "./storage/raw_videos"
    UPLOAD_DIR: str = "./storage/raw_videos"  # Backward-compatible alias
    OUTPUT_VIDEO_DIR: str = "./storage/output_videos"
    MAX_UPLOAD_SIZE_BYTES: int = 1073741824  # 1GB

    # Allowed Video Types
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".mov", ".webm"}
    ALLOWED_VIDEO_MIME_TYPES: Set[str] = {
        "video/mp4",
        "video/quicktime",
        "video/webm",
        "application/octet-stream"
    }

    # ML Pipeline & Model Configuration
    MODEL_WEIGHTS_PATH: str = "./models/classroom_temporal_model.pth"
    SPATIAL_BACKBONE: str = "resnet18"
    TEMPORAL_MODEL_TYPE: str = "GRU"  # Choices: GRU, LSTM, RNN
    TEMPORAL_SEQUENCE_LENGTH: int = 16  # Frames per temporal sequence window
    TEMPORAL_HIDDEN_DIM: int = 256
    TEMPORAL_NUM_LAYERS: int = 2
    SAMPLING_FPS: float = 2.0  # Extract 2 frames per second
    ML_DEVICE: str = "auto"  # auto, cuda, or cpu
    AUTO_DISPATCH_ML_JOB: bool = True  # Automatically run background ML analysis on upload
    CLASSROOM_DETECTOR_MODE: str = "full"  # full or tiled; tiled is opt-in for dense classrooms
    CLASSROOM_DETECTOR_WEIGHTS: str = "./yolov8s.pt"
    CLASSROOM_DETECTOR_CONFIDENCE: float = 0.15
    CLASSROOM_DETECTOR_IMAGE_SIZE: int = 1280
    CLASSROOM_DETECTOR_IOU: float = 0.55
    CLASSROOM_TILE_SIZE: int = 640
    CLASSROOM_TILE_OVERLAP: float = 0.25
    CLASSROOM_NMS_IOU: float = 0.55
    CLASSROOM_TRACKER_MODE: str = "baseline"  # baseline or dense
    CLASSROOM_TRACKER_CONFIRMATION_HITS: int = 2
    CLASSROOM_TRACKER_CENTER_GATE: float = 2.5
    CLASSROOM_TRACKER_LOST_BUFFER: int = 12
    CLASSROOM_TRACKER_APPEARANCE_WEIGHT: float = 0.25
    CLASSROOM_TRACKER_APPEARANCE_GATE: float = 0.65
    # High-recall detector (opt-in: CLASSROOM_DETECTOR_MODE=highrecall)
    CLASSROOM_HIGHRECALL_FINE_TILE: int = 480
    CLASSROOM_HIGHRECALL_FINE_OVERLAP: float = 0.40
    CLASSROOM_HIGHRECALL_MERGE_IOU: float = 0.40
    CLASSROOM_HIGHRECALL_CONTAINMENT: float = 0.85
    CLASSROOM_HIGHRECALL_CONFIDENCE: float = 0.23
    CLASSROOM_HIGHRECALL_USE_TILES: bool = False
    MAX_JOB_RETRIES: int = 3
    RETRY_BACKOFF_SECONDS: int = 5

    # Campus Ingestion, Matching & Retention Settings
    AUTO_INGEST_DROPZONE_DIR: str = "./storage/dropzone"
    SESSION_MATCH_TOLERANCE_MINUTES: int = 15
    NVR_WEBHOOK_SECRET: str = "tempo_nvr_secure_webhook_key_2026"
    STORAGE_RETENTION_DAYS: int = 30
    PURGE_RAW_VIDEO_AFTER_ANALYSIS: bool = False  # College privacy toggle

    # Security & CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    ENABLE_HTTPS_REDIRECT: bool = False

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """
        Enforce strict PostgreSQL requirement.
        Fail clearly if DATABASE_URL is missing or uses SQLite / other non-PostgreSQL dialects.
        """
        if not v or not isinstance(v, str):
            raise ValueError(
                "DATABASE_URL is required. PostgreSQL is mandatory for this application."
            )
        
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql://", 1)

        allowed_prefixes = ("postgresql://", "postgresql+psycopg2://", "postgresql+asyncpg://")
        if not any(v.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(
                f"Invalid DATABASE_URL '{v}'. PostgreSQL is strictly mandatory! "
                f"URL must start with one of: {allowed_prefixes}. SQLite and other databases are not permitted."
            )
        return v

    @field_validator("ALLOWED_EMAIL_DOMAIN")
    @classmethod
    def validate_email_domain(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("ALLOWED_EMAIL_DOMAIN cannot be empty.")
        if v.startswith("@"):
            v = v[1:]
        return v

    @field_validator("TEMPORAL_MODEL_TYPE")
    @classmethod
    def validate_temporal_model(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if v_upper not in ("GRU", "LSTM", "RNN"):
            raise ValueError(f"TEMPORAL_MODEL_TYPE must be 'GRU', 'LSTM', or 'RNN', got: {v}")
        return v_upper


settings = Settings()
