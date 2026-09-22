# Classroom Temporal Behaviour Analytics Backend & ML Service

Production-ready backend foundation and integrated Machine Learning service for the **Classroom Temporal Behaviour Analytics System** built using **FastAPI**, **PostgreSQL**, **SQLAlchemy 2.0**, **PyTorch (ResNet-18 + GRU/LSTM/RNN)**, **Alembic**, and **Pydantic v2**.

---

## 🌟 Core Highlights

- **Mandatory PostgreSQL**: Strictly enforces PostgreSQL connection strings with fail-fast startup validation (`postgresql://` or `postgresql+psycopg2://`). No silent SQLite fallbacks.
- **Strict Faculty Domain Restriction**: Faculty registration and authentication are strictly restricted to official `@klu.ac.in` email addresses.
- **Faculty Multi-Tenancy & Data Isolation**: Strict tenant scoping across all levels (`Faculty → Subject → Section → ClassSession → Video → AnalysisJob`). A faculty member can **only** access and manage their own data.
- **Secure Video Upload & Storage**: Multipart video uploads (`.mp4`, `.mov`, `.webm`) stored on local/object file storage with collision-free filenames (PostgreSQL stores only metadata and file paths).
- **Integrated Classroom Temporal Behaviour ML Model**:
  - **Spatial Feature Extractor**: ResNet-18 (512-dim frame embedding).
  - **Temporal Sequence Modeler**: Recurrent sequence modeling with configurable `GRU`, `LSTM`, or `RNN` architectures.
  - **Frame Preprocessing**: ImageNet standard normalization and $(224, 224)$ resizing.
  - **Temporal Frame Sampling**: Extracts video frames at configurable FPS rate and constructs sliding sequence windows.
- **Five Observable Behaviour Classes**:
  1. `Looking_Toward_Instruction`
  2. `Reading`
  3. `Writing`
  4. `Peer_Interaction`
  5. `Looking_Away`
- **Asynchronous Background Processing**: Video inference executes as a non-blocking background job with real-time progress updates (`0% -> 100%`) and state transitions (`PENDING` -> `PROCESSING` -> `COMPLETED` / `FAILED`).
- **PostgreSQL Prediction Persistence**: Stores timestamped behaviour classifications, confidence values, and model metadata via `BehaviourResult`.
- **HTTP 206 Partial Content Video Streaming**: Full support for `Range` byte-seeking playback for both raw uploaded videos and annotated output videos.

---

## 🏗️ Architecture Overview

```
TEMPO/
├── alembic/                      # Database migrations
│   ├── versions/                 # Versioned migration scripts
│   │   └── 001_initial_schema.py # Initial PostgreSQL schema
│   ├── env.py                    # Alembic environment config
│   └── script.py.mako            # Migration template
├── app/
│   ├── api/                      # REST API Endpoints (v1)
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── auth.py       # Login, Register (@klu.ac.in), Refresh, Profile
│   │   │   │   ├── subjects.py   # Subject management
│   │   │   │   ├── sections.py   # Section management
│   │   │   │   ├── students.py   # Student enrollment & roster
│   │   │   │   ├── sessions.py   # Class session scheduling
│   │   │   │   ├── videos.py     # Video upload, status & HTTP 206 streaming
│   │   │   │   └── analysis.py   # Analysis jobs & temporal metrics
│   │   │   └── router.py         # Router aggregation
│   ├── core/                     # Core configs & security
│   │   ├── config.py             # BaseSettings with ML & strict PostgreSQL settings
│   │   ├── security.py           # Bcrypt hashing & JWT management
│   │   ├── storage.py            # Disk file storage & HTTP 206 Range streaming
│   │   └── dependencies.py       # Auth guards & faculty scoping
│   ├── db/                       # Database engine & session
│   │   ├── base.py               # Declarative Base & metadata aggregator
│   │   └── session.py            # Engine & SessionLocal factory
│   ├── ml/                       # Integrated Temporal ML Service
│   │   ├── __init__.py           # ML module exports
│   │   ├── model.py              # ResNet-18 + GRU/LSTM/RNN PyTorch Model
│   │   ├── preprocessor.py       # ImageNet transforms & tensor batching
│   │   ├── sampler.py            # Video frame extraction & sliding window generator
│   │   └── service.py            # MLInferenceService orchestrator
│   ├── models/                   # SQLAlchemy 2.0 ORM models
│   │   ├── faculty.py            # Faculty table
│   │   ├── academic.py           # Subject, Section, Student tables
│   │   ├── session.py            # ClassSession table
│   │   ├── video.py              # Video table (file paths, content_type, output_video_path)
│   │   └── analysis.py           # AnalysisJob, TrackResult, BehaviourResult
│   ├── schemas/                  # Pydantic v2 schemas
│   │   ├── auth.py               # Auth & @klu.ac.in validator
│   │   ├── faculty.py            # Faculty schemas
│   │   ├── academic.py           # Academic hierarchy schemas
│   │   ├── session.py            # Session schemas
│   │   ├── video.py              # Video upload & streaming schemas
│   │   └── analysis.py           # 5-class Behaviour Enum & Track ID schemas
│   ├── services/                 # Layered business logic & tenancy enforcement
│   │   ├── auth_service.py       # Auth & credential services
│   │   ├── academic_service.py   # Subject/Section/Student operations
│   │   ├── session_service.py    # ClassSession & Video upload operations
│   │   ├── analysis_service.py   # Analysis & timeline aggregations
│   │   └── ml_connector.py       # Asynchronous background ML worker & connector
│   └── main.py                   # FastAPI app entrypoint & lifespan
├── tests/                        # Automated Pytest suite (28 passing tests)
│   ├── conftest.py               # Test fixtures & tokens
│   ├── test_auth.py              # Auth & domain validation tests
│   ├── test_academic.py          # Scoping & academic CRUD tests
│   ├── test_ml_pipeline.py       # ResNet-18 + GRU/LSTM/RNN inference tests
│   ├── test_sessions_videos.py   # Session & video tests
│   ├── test_video_upload_and_streaming.py # Upload, streaming & 5-class behaviour tests
│   ├── test_analysis.py          # Analysis & temporal summary tests
│   └── test_database_validation.py # Strict Postgres validation tests
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git exclusions
├── alembic.ini                   # Alembic configuration
└── requirements.txt              # Production & test dependencies
```

---

## ⚙️ Environment Variables & ML Configurations

Configure in `.env`:
```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/tempo_db

# Security & JWT
JWT_SECRET_KEY=generate_a_secure_random_key_here
ALLOWED_EMAIL_DOMAIN=klu.ac.in

# ML Model Settings
MODEL_WEIGHTS_PATH=./models/classroom_temporal_model.pth
SPATIAL_BACKBONE=resnet18
TEMPORAL_MODEL_TYPE=GRU # Choices: GRU, LSTM, RNN
TEMPORAL_SEQUENCE_LENGTH=16
TEMPORAL_HIDDEN_DIM=256
TEMPORAL_NUM_LAYERS=2
SAMPLING_FPS=2.0
ML_DEVICE=auto # auto, cuda, or cpu
```

---

## 📡 API Endpoints Reference

### Authentication (`/api/v1/auth`)
| Method | Path | Description | Access |
|---|---|---|---|
| `POST` | `/api/v1/auth/register` | Register faculty (must be `@klu.ac.in`) | Public |
| `POST` | `/api/v1/auth/login` | Login with `@klu.ac.in` email & password | Public |
| `POST` | `/api/v1/auth/refresh` | Issue new access token using refresh token | Public |
| `GET` | `/api/v1/auth/me` | Get current logged-in faculty profile | Faculty Auth |
| `PUT` | `/api/v1/auth/me` | Update faculty profile / change password | Faculty Auth |

### Academic Hierarchy (`/api/v1/subjects`, `/sections`, `/students`)
| Method | Path | Description | Access |
|---|---|---|---|
| `POST` | `/api/v1/subjects/` | Create a subject | Faculty Auth |
| `GET` | `/api/v1/subjects/` | List faculty's subjects | Faculty Auth |
| `GET` | `/api/v1/subjects/{id}` | Get subject details | Faculty Auth |
| `PUT` | `/api/v1/subjects/{id}` | Update subject | Faculty Auth |
| `DELETE` | `/api/v1/subjects/{id}` | Delete subject (cascades) | Faculty Auth |
| `POST` | `/api/v1/sections/` | Create section under a subject | Faculty Auth |
| `GET` | `/api/v1/sections/subject/{subject_id}` | List sections for a subject | Faculty Auth |
| `POST` | `/api/v1/students/` | Enroll student in section | Faculty Auth |
| `POST` | `/api/v1/students/bulk` | Bulk enroll students | Faculty Auth |
| `GET` | `/api/v1/students/section/{section_id}` | List students in section | Faculty Auth |

### Class Sessions, Video Uploads & Streaming (`/api/v1/sessions`, `/videos`)
| Method | Path | Description | Access |
|---|---|---|---|
| `POST` | `/api/v1/sessions/` | Schedule a class session | Faculty Auth |
| `GET` | `/api/v1/sessions/` | List sessions (with filters) | Faculty Auth |
| `GET` | `/api/v1/sessions/{id}` | Get session details | Faculty Auth |
| `POST` | `/api/v1/videos/upload` | Multipart video upload + automatic ML analysis job creation | Faculty Auth |
| `GET` | `/api/v1/videos/` | List faculty's uploaded videos | Faculty Auth |
| `GET` | `/api/v1/videos/{id}` | Get video metadata | Faculty Auth |
| `GET` | `/api/v1/videos/{id}/status` | Check processing & analysis status | Faculty Auth |
| `GET` | `/api/v1/videos/{id}/stream` | Stream raw video (HTTP 206 Partial Content) | Faculty Auth |
| `GET` | `/api/v1/videos/{id}/output-stream` | Stream annotated output video (HTTP 206) | Faculty Auth |
| `DELETE` | `/api/v1/videos/{id}` | Delete video record and storage files | Faculty Auth |

### Behaviour Analytics (`/api/v1/analysis`)
| Method | Path | Description | Access |
|---|---|---|---|
| `POST` | `/api/v1/analysis/jobs` | Create analysis job and start background ML inference | Faculty Auth |
| `GET` | `/api/v1/analysis/jobs/{id}` | Get job status & progress | Faculty Auth |
| `PATCH` | `/api/v1/analysis/jobs/{id}/status` | Update job status & progress | Faculty Auth |
| `POST` | `/api/v1/analysis/jobs/{id}/tracks` | Save per-session student track result | Faculty Auth |
| `POST` | `/api/v1/analysis/jobs/{id}/behaviours` | Batch save behaviour detections (5 classes) | Faculty Auth |
| `GET` | `/api/v1/analysis/jobs/{id}/behaviours` | Query behaviour predictions | Faculty Auth |
| `GET` | `/api/v1/analysis/jobs/{id}/summary` | Get aggregated timeline & track engagement metrics | Faculty Auth |

---

## 🧪 Automated Test Suite

```bash
pytest -v
# 28 passed in 27.19s
```
