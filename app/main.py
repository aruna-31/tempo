import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.storage import StorageService
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure upload & storage directories exist
    StorageService.ensure_directories_exist()
    
    # Startup: Verify PostgreSQL connection can be established
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as e:
        print(f"[STARTUP WARNING] Could not verify PostgreSQL connection on startup: {e}")
        print("[STARTUP WARNING] Ensure PostgreSQL service is active and DATABASE_URL is reachable.")
    
    yield
    
    # Shutdown
    engine.dispose()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Production-ready backend foundation for Classroom Temporal Behaviour Analytics. "
        "Strictly restricted to KL University faculty (@klu.ac.in) with multi-tenant data isolation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


# Include API Routers
app.include_router(api_router)


@app.get("/", tags=["General"])
def root():
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "allowed_email_domain": f"@{settings.ALLOWED_EMAIL_DOMAIN}",
        "database_type": "PostgreSQL (Mandatory)"
    }


@app.get("/health", tags=["General"])
def health_check():
    db_status = "unhealthy"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            db_status = "healthy"
    except Exception as e:
        db_status = f"disconnected: {str(e)}"

    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "database": db_status,
        "environment": settings.APP_ENV
    }
