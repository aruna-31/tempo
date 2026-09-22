from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from app.core.dependencies import get_current_active_faculty
from app.db.session import get_db
from app.models.faculty import Faculty
from app.schemas.auth import (
    FacultyLogin,
    FacultyRegister,
    FacultyResponse,
    FacultyUpdate,
    Token,
    TokenRefreshRequest,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=FacultyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new faculty account",
    description="Registers a faculty member. Email must belong to the official university domain (@klu.ac.in)."
)
def register(
    data: FacultyRegister,
    db: Session = Depends(get_db)
):
    faculty = AuthService.register_faculty(db=db, data=data)
    return faculty


@router.post(
    "/login",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Faculty login",
    description="Authenticates faculty using @klu.ac.in email and password, returning JWT access and refresh tokens."
)
def login(
    data: FacultyLogin,
    db: Session = Depends(get_db)
):
    _, tokens = AuthService.authenticate_faculty(db=db, data=data)
    return tokens


@router.post(
    "/refresh",
    response_model=Token,
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
    description="Issues a fresh JWT access token using a valid refresh token."
)
def refresh(
    data: TokenRefreshRequest,
    db: Session = Depends(get_db)
):
    tokens = AuthService.refresh_token(db=db, refresh_token_str=data.refresh_token)
    return tokens


@router.get(
    "/me",
    response_model=FacultyResponse,
    status_code=status.HTTP_200_OK,
    summary="Get current faculty profile",
    description="Retrieves the profile of the currently authenticated faculty member."
)
def get_me(
    current_faculty: Faculty = Depends(get_current_active_faculty)
):
    return current_faculty


@router.put(
    "/me",
    response_model=FacultyResponse,
    status_code=status.HTTP_200_OK,
    summary="Update current faculty profile",
    description="Updates faculty details (full name, department, designation, or password)."
)
def update_me(
    data: FacultyUpdate,
    current_faculty: Faculty = Depends(get_current_active_faculty),
    db: Session = Depends(get_db)
):
    updated = AuthService.update_profile(db=db, faculty=current_faculty, data=data)
    return updated
