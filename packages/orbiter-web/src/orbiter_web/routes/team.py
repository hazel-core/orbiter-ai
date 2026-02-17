"""Team management endpoints — admin-only user CRUD."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import _hash_password, require_role
from orbiter_web.sanitize import sanitize_html
from orbiter_web.services.audit import audit_log

router = APIRouter(prefix="/api/v1/settings/team", tags=["team"])

_VALID_ROLES = {"viewer", "developer", "admin"}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class InviteRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=8)
    role: str = Field(default="developer")


class TeamMemberResponse(BaseModel):
    id: str
    email: str
    role: str
    created_at: str


class RoleUpdateRequest(BaseModel):
    role: str = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[TeamMemberResponse])
async def list_team_members(
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all users (admin only)."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, email, role, created_at FROM users ORDER BY created_at",
        )
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


@router.post("/invite", response_model=TeamMemberResponse, status_code=201)
async def invite_member(
    body: InviteRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Create a new user (admin only)."""
    if body.role not in _VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.role}")

    email = sanitize_html(body.email.strip().lower())

    async with get_db() as db:
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (email,))
        if await cursor.fetchone():
            raise HTTPException(status_code=409, detail="Email already registered")

        new_id = str(uuid.uuid4())
        password_hash = _hash_password(body.password)
        await db.execute(
            "INSERT INTO users (id, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (new_id, email, password_hash, body.role),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT id, email, role, created_at FROM users WHERE id = ?",
            (new_id,),
        )
        row = await cursor.fetchone()

    ip = request.client.host if request.client else None
    await audit_log(
        user["id"],
        "invite_user",
        "user",
        new_id,
        details={"email": email, "role": body.role},
        ip_address=ip,
    )

    return dict(row)  # type: ignore[arg-type]


@router.put("/{user_id}", response_model=TeamMemberResponse)
async def update_member_role(
    user_id: str,
    body: RoleUpdateRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> dict[str, Any]:
    """Update a user's role (admin only)."""
    if body.role not in _VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.role}")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        target = await cursor.fetchone()

    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent demoting the last admin.
    if dict(target)["role"] == "admin" and body.role != "admin":
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin'",
            )
            row = await cursor.fetchone()
        if row and row["cnt"] <= 1:  # type: ignore[index]
            raise HTTPException(status_code=400, detail="Cannot demote the last admin")

    async with get_db() as db:
        await db.execute(
            "UPDATE users SET role = ? WHERE id = ?",
            (body.role, user_id),
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT id, email, role, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        updated = await cursor.fetchone()

    ip = request.client.host if request.client else None
    await audit_log(
        user["id"],
        "update_role",
        "user",
        user_id,
        details={"old_role": dict(target)["role"], "new_role": body.role},
        ip_address=ip,
    )

    return dict(updated)  # type: ignore[arg-type]


@router.delete("/{user_id}", status_code=204)
async def deactivate_member(
    user_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> None:
    """Deactivate a user and revoke all their sessions (admin only)."""
    # Cannot delete yourself.
    if user_id == user["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id, role FROM users WHERE id = ?",
            (user_id,),
        )
        target = await cursor.fetchone()

    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Prevent deleting the last admin.
    if dict(target)["role"] == "admin":
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT COUNT(*) AS cnt FROM users WHERE role = 'admin'",
            )
            row = await cursor.fetchone()
        if row and row["cnt"] <= 1:  # type: ignore[index]
            raise HTTPException(status_code=400, detail="Cannot delete the last admin")

    async with get_db() as db:
        # Revoke all sessions first.
        await db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        # Delete the user (cascade will clean up related data).
        await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
        await db.commit()

    ip = request.client.host if request.client else None
    await audit_log(
        user["id"],
        "delete_user",
        "user",
        user_id,
        ip_address=ip,
    )
