"""Audit log endpoints — admin-only access to security event history."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from orbiter_web.database import get_db
from orbiter_web.pagination import PaginatedResponse, paginate
from orbiter_web.routes.auth import require_role

router = APIRouter(prefix="/api/audit-log", tags=["audit-log"])


class AuditLogEntry(BaseModel):
    id: str
    user_id: str
    action: str
    entity_type: str | None
    entity_id: str | None
    details: dict[str, Any] | None
    ip_address: str | None
    created_at: str


def _map_row(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite Row to a dict, parsing details_json."""
    d = dict(row)
    raw = d.pop("details_json", None)
    d["details"] = json.loads(raw) if raw else None
    return d


@router.get("")
async def list_audit_log(
    user_id: str | None = None,
    action: str | None = None,
    entity_type: str | None = None,
    date_from: str | None = Query(None, description="ISO date lower bound (inclusive)"),
    date_to: str | None = Query(None, description="ISO date upper bound (inclusive)"),
    cursor: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    _user: dict[str, Any] = Depends(require_role("admin")),  # noqa: B008
) -> PaginatedResponse:
    """Return paginated audit log entries with optional filters (admin only)."""
    conditions: list[str] = []
    params: list[Any] = []

    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)
    if action:
        conditions.append("action = ?")
        params.append(action)
    if entity_type:
        conditions.append("entity_type = ?")
        params.append(entity_type)
    if date_from:
        conditions.append("created_at >= ?")
        params.append(date_from)
    if date_to:
        conditions.append("created_at <= datetime(?, '+1 day')")
        params.append(date_to)

    async with get_db() as db:
        return await paginate(
            db,
            table="audit_log",
            conditions=conditions,
            params=params,
            cursor=cursor,
            limit=limit,
            row_mapper=_map_row,
        )
