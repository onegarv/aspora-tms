"""
JWT bearer authentication dependency for the Dashboard API.
"""

from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

from config.settings import settings

security = HTTPBearer()


async def require_auth(credentials=Depends(security)):
    try:
        jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
