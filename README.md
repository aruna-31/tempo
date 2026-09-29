# TEMPO — Classroom Temporal Behaviour Analytics

> AI-powered classroom video analytics prototype combining computer vision, temporal modelling, FastAPI, PostgreSQL, and a faculty-facing workflow.

## Problem

Classroom video contains temporal behavioural information that is difficult to analyze manually at scale. TEMPO explores an end-to-end pipeline for detecting and classifying observable student behaviours from video while connecting model output to a usable backend and analytics workflow.

## Prototype workflow

```
Faculty login
    ↓
Today's classes
    ↓
Select class session
    ↓
Upload MP4
    ↓
Video processing
    ↓
YOLO → tracking → ResNet-18 features → temporal model
    ↓
Behaviour predictions
    ↓
PostgreSQL results + annotated video
```

## Observable behaviour classes

1. Looking toward instruction
2. Reading
3. Writing
4. Peer interaction
5. Looking away

## ML pipeline

- **Object detection:** YOLO
- **Tracking:** ByteTrack
- **Spatial features:** ResNet-18
- **Temporal modelling:** configurable RNN / GRU / LSTM
- **Sequence length:** configurable sliding temporal windows
- **Output:** timestamped behaviour predictions, confidence values, tracking results, and analysis summaries

## Backend architecture

```
Faculty
  ↓
FastAPI + JWT
  ↓
Faculty → Subject → Section → ClassSession → Video → AnalysisJob
  ↓
PostgreSQL
  ↓
Background ML inference
```

The system enforces faculty-level data isolation and uses PostgreSQL as the required database rather than silently falling back to SQLite.

## Tech stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic |
| Database | PostgreSQL, SQLAlchemy 2.0 |
| Migrations | Alembic |
| Authentication | JWT + bcrypt |
| ML | PyTorch, ResNet-18, RNN/GRU/LSTM |
| Video | OpenCV / video processing pipeline |
| Testing | Pytest |

## API areas

- Faculty authentication
- Subjects, sections, and students
- Class sessions
- Video upload and streaming
- Analysis job management
- Track results
- Behaviour predictions
- Timeline and engagement summaries

## Validation

The repository includes an automated Pytest suite covering authentication, academic scoping, ML integration, video workflows, analysis, and database validation.

Run:

```bash
pytest -v
```

## Current ML status

The temporal model is a prototype and its current evaluation should be interpreted as an engineering baseline rather than a production-grade accuracy claim. The repository documents the model configuration and evaluation work so improvements can be measured transparently.

## Future production direction

A production deployment would integrate with an institutional LMS/video source, stronger student identity mapping, scalable object storage, GPU-backed inference, monitoring, and stricter privacy/governance controls.

## Author

**Lavanuru Aruna** · https://github.com/aruna-31
