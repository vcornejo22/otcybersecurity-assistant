"""FastAPI dependency injection for authentication."""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.core.auth import decode_jwt, verify_api_key

security = HTTPBearer(auto_error=False)


async def verify_credentials(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify the request credentials (API Key or JWT).

    Returns the authenticated principal (user ID or "api_key").
    Raises HTTP 401 if credentials are missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Se requiere autenticación. Usá el header Authorization: Bearer <token>",
        )

    token = credentials.credentials

    # Try API Key first
    if verify_api_key(token, settings):
        return "api_key"

    # Try JWT
    try:
        payload = decode_jwt(token, settings)
        return payload.get("sub", "unknown")
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas o expiradas",
        )


def verify_admin(
    credentials: HTTPAuthorizationCredentials = Security(security),
    settings: Settings = Depends(get_settings),
) -> str:
    """Require a valid API Key (admin scope). No JWT allowed."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Se requiere API Key de administrador",
        )
    token = credentials.credentials
    if not verify_api_key(token, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key inválida",
        )
    return "admin"
