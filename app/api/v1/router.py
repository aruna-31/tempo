from fastapi import APIRouter
from app.api.v1.endpoints import (
    analysis,
    auth,
    ingest,
    schedules,
    sections,
    sessions,
    students,
    subjects,
    videos,
)

# PROTOTYPE SCOPE: Rooms & Cameras are excluded from the prototype. The routes and
# models remain available for the future production LMS/NVR integration.
api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth.router)
api_router.include_router(schedules.router)
api_router.include_router(subjects.router)
api_router.include_router(sections.router)
api_router.include_router(students.router)
api_router.include_router(sessions.router)
api_router.include_router(videos.router)
api_router.include_router(analysis.router)
api_router.include_router(ingest.router)
