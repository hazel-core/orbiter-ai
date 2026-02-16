"""Tools REST API.

Manages user-defined tools stored in the database.
Also exposes built-in tools merged into listings.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/tools", tags=["tools"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"search", "code", "file", "data", "communication", "custom"}
VALID_TOOL_TYPES = {"function", "http", "schema", "mcp"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ToolResponse(BaseModel):
    id: str
    name: str
    description: str
    category: str
    schema_json: str
    code: str
    tool_type: str
    usage_count: int
    project_id: str
    user_id: str
    created_at: str


class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    description: str = ""
    category: str = "custom"
    schema_json: str = "{}"
    code: str = ""
    tool_type: str = "function"


class ToolUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    category: str | None = None
    schema_json: str | None = None
    code: str | None = None
    tool_type: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, tool_id: str, user_id: str) -> dict[str, Any]:
    """Verify tool exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM tools WHERE id = ? AND user_id = ?",
        (tool_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Tool not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ToolResponse])
async def list_tools(
    category: str | None = Query(None),
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all tools for the current user, optionally filtered by category and/or project."""
    async with get_db() as db:
        conditions = ["user_id = ?"]
        params: list[str] = [user["id"]]

        if category:
            conditions.append("category = ?")
            params.append(category)
        if project_id:
            conditions.append("project_id = ?")
            params.append(project_id)

        where = " AND ".join(conditions)
        cursor = await db.execute(
            f"SELECT * FROM tools WHERE {where} ORDER BY created_at DESC",
            params,
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=ToolResponse, status_code=201)
async def create_tool(
    body: ToolCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new tool."""
    if body.category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )
    if body.tool_type not in VALID_TOOL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tool_type. Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}",
        )

    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        tool_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO tools (
                id, name, description, category, schema_json, code,
                tool_type, usage_count, project_id, user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                tool_id,
                body.name,
                body.description,
                body.category,
                body.schema_json,
                body.code,
                body.tool_type,
                body.project_id,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single tool by ID with full detail including schema."""
    async with get_db() as db:
        return await _verify_ownership(db, tool_id, user["id"])


@router.put("/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: str,
    body: ToolUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a tool's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    if "category" in updates and updates["category"] not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid category. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}",
        )
    if "tool_type" in updates and updates["tool_type"] not in VALID_TOOL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid tool_type. Must be one of: {', '.join(sorted(VALID_TOOL_TYPES))}",
        )

    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), tool_id]

        await db.execute(
            f"UPDATE tools SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{tool_id}", status_code=204)
async def delete_tool(
    tool_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a tool."""
    async with get_db() as db:
        await _verify_ownership(db, tool_id, user["id"])
        await db.execute("DELETE FROM tools WHERE id = ?", (tool_id,))
        await db.commit()
