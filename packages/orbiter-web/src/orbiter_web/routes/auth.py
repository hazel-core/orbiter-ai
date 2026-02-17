"""Authentication endpoints and middleware."""

from __future__ import annotations

import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from orbiter_web.config import settings
from orbiter_web.database import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])

SESSION_COOKIE = "orbiter_session"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a bcrypt hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# Protected route dependency
# ---------------------------------------------------------------------------


async def get_current_user(
    orbiter_session: str | None = Cookie(None),
) -> dict[str, Any]:
    """Extract the current user from the session cookie, or raise 401."""
    if not orbiter_session:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (orbiter_session,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/login", response_model=UserResponse)
async def login(body: LoginRequest, response: Response) -> dict[str, Any]:
    """Authenticate with email + password and set a session cookie."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (body.email,),
        )
        user = await cursor.fetchone()

    if user is None or not _verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create session with CSRF token.
    session_id = str(uuid.uuid4())
    csrf_token = secrets.token_urlsafe(32)
    expires_at = (
        datetime.now(UTC) + timedelta(hours=settings.session_expiry_hours)
    ).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        await db.execute(
            "INSERT INTO sessions (id, user_id, expires_at, csrf_token) VALUES (?, ?, ?, ?)",
            (session_id, user["id"], expires_at, csrf_token),
        )
        await db.commit()

    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        samesite="lax",
        max_age=settings.session_expiry_hours * 3600,
        path="/",
    )

    return {"id": user["id"], "email": user["email"], "created_at": user["created_at"]}


@router.post("/logout", status_code=204)
async def logout(
    response: Response,
    orbiter_session: str | None = Cookie(None),
) -> None:
    """Clear the session cookie and delete the session."""
    if orbiter_session:
        async with get_db() as db:
            await db.execute("DELETE FROM sessions WHERE id = ?", (orbiter_session,))
            await db.commit()

    response.delete_cookie(key=SESSION_COOKIE, path="/")


@router.get("/me", response_model=UserResponse)
async def me(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return the current authenticated user."""
    return user


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8)


class MessageResponse(BaseModel):
    message: str


@router.put("/password", response_model=MessageResponse)
async def change_password(
    body: PasswordChangeRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
    orbiter_session: str | None = Cookie(None),
) -> dict[str, str]:
    """Change the current user's password."""
    user_id = user["id"]

    # Fetch current password hash.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not _verify_password(body.current_password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect")

    # Hash and store new password, then invalidate other sessions.
    new_hash = _hash_password(body.new_password)
    async with get_db() as db:
        await db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user_id),
        )
        # Invalidate all sessions except the current one.
        await db.execute(
            "DELETE FROM sessions WHERE user_id = ? AND id != ?",
            (user_id, orbiter_session),
        )
        await db.commit()

    return {"message": "Password updated"}


class CsrfResponse(BaseModel):
    token: str


@router.get("/csrf", response_model=CsrfResponse)
async def get_csrf_token(
    orbiter_session: str | None = Cookie(None),
) -> dict[str, str]:
    """Return the CSRF token for the current session."""
    if not orbiter_session:
        raise HTTPException(status_code=401, detail="Not authenticated")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT csrf_token FROM sessions WHERE id = ? AND expires_at > datetime('now')",
            (orbiter_session,),
        )
        row = await cursor.fetchone()

    if row is None or row["csrf_token"] is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {"token": row["csrf_token"]}
