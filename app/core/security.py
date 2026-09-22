from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
import bcrypt
from jose import JWTError, jwt
from app.core.config import settings

# Dedicated token type constants
TOKEN_TYPE_ACCESS = "access"
TOKEN_TYPE_REFRESH = "refresh"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Generate a bcrypt password hash."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def create_jwt_token(
    subject: Union[str, Any],
    token_type: str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None
) -> str:
    """Create a signed JWT token with claims and expiration."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        if token_type == TOKEN_TYPE_ACCESS:
            expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        else:
            expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    to_encode: Dict[str, Any] = {
        "sub": str(subject),
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp())
    }
    if extra_claims:
        to_encode.update(extra_claims)

    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None
) -> str:
    return create_jwt_token(
        subject=subject,
        token_type=TOKEN_TYPE_ACCESS,
        expires_delta=expires_delta,
        extra_claims=extra_claims
    )


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None
) -> str:
    return create_jwt_token(
        subject=subject,
        token_type=TOKEN_TYPE_REFRESH,
        expires_delta=expires_delta,
        extra_claims=extra_claims
    )


def decode_jwt_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token signature and expiration."""
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError as e:
        raise ValueError(f"Invalid or expired token: {str(e)}") from e
