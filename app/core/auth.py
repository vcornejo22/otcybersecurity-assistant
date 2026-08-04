"""Authentication: API Key and JWT token management."""

import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta

import jwt

from app.config import Settings


def verify_api_key(api_key: str, settings: Settings | None = None) -> bool:
    """Verify an API key using constant-time comparison."""
    settings = settings or Settings()
    if not settings.API_KEY:
        return False
    return hmac.compare_digest(api_key, settings.API_KEY)


def create_jwt(
    user: str,
    expires_delta: timedelta = timedelta(hours=24),
    settings: Settings | None = None,
) -> str:
    """Create a JWT token for the given user."""
    settings = settings or Settings()
    now = datetime.now(UTC)
    payload = {
        "sub": user,
        "iat": now,
        "exp": now + expires_delta,
        "scope": "query",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def decode_jwt(token: str, settings: Settings | None = None) -> dict:
    """Decode and validate a JWT token. Returns the payload."""
    settings = settings or Settings()
    return jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return f"sk-otcs-{secrets.token_urlsafe(32)}"


def hash_token(token: str) -> str:
    """Return a SHA-256 hash of a token for safe logging."""
    return hashlib.sha256(token.encode()).hexdigest()[:16]
