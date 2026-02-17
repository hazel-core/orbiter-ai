"""Workflows CRUD REST API with canvas state persistence."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user
from orbiter_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/workflows", tags=["workflows"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    description: str = ""
    nodes_json: str = "[]"
    edges_json: str = "[]"
    viewport_json: str = '{"x":0,"y":0,"zoom":1}'


class WorkflowUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    nodes_json: str | None = None
    edges_json: str | None = None
    viewport_json: str | None = None
    status: str | None = None


class WorkflowImport(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    description: str = ""
    nodes_json: str = "[]"
    edges_json: str = "[]"
    viewport_json: str = '{"x":0,"y":0,"zoom":1}'


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str
    project_id: str
    nodes_json: str
    edges_json: str
    viewport_json: str
    version: int
    status: str
    last_run_at: str | None
    user_id: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(db: Any, workflow_id: str, user_id: str) -> dict[str, Any]:
    """Verify workflow exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM workflows WHERE id = ? AND user_id = ?",
        (workflow_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[WorkflowResponse])
async def list_workflows(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all workflows for the current user, optionally filtered by project."""
    async with get_db() as db:
        if project_id:
            cursor = await db.execute(
                "SELECT * FROM workflows WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                (user["id"], project_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM workflows WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    body: WorkflowCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new workflow."""
    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workflow_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO workflows (id, name, description, project_id, nodes_json, edges_json, viewport_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.project_id,
                body.nodes_json,
                body.edges_json,
                body.viewport_json,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.post("/import", response_model=WorkflowResponse, status_code=201)
async def import_workflow(
    body: WorkflowImport,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Import and create a workflow from JSON data."""
    # Validate that JSON fields are valid JSON.
    for field_name in ("nodes_json", "edges_json", "viewport_json"):
        value = getattr(body, field_name)
        try:
            json.loads(value)
        except (json.JSONDecodeError, TypeError):
            raise HTTPException(  # noqa: B904
                status_code=422,
                detail=f"Invalid JSON in {field_name}",
            )

    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        workflow_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO workflows (id, name, description, project_id, nodes_json, edges_json, viewport_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                sanitize_html(body.name),
                sanitize_html(body.description),
                body.project_id,
                body.nodes_json,
                body.edges_json,
                body.viewport_json,
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single workflow by ID (including canvas state)."""
    async with get_db() as db:
        return await _verify_ownership(db, workflow_id, user["id"])


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a workflow's fields (including canvas state)."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "description"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        await _verify_ownership(db, workflow_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), workflow_id]

        await db.execute(
            f"UPDATE workflows SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a workflow."""
    async with get_db() as db:
        await _verify_ownership(db, workflow_id, user["id"])
        await db.execute("DELETE FROM workflows WHERE id = ?", (workflow_id,))
        await db.commit()


@router.post("/{workflow_id}/export")
async def export_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> JSONResponse:
    """Export a workflow as downloadable JSON."""
    async with get_db() as db:
        data = await _verify_ownership(db, workflow_id, user["id"])

    export_data = {
        "name": data["name"],
        "description": data["description"],
        "version": data["version"],
        "nodes_json": data["nodes_json"],
        "edges_json": data["edges_json"],
        "viewport_json": data["viewport_json"],
    }

    safe_name = data["name"].replace(" ", "_").lower()
    return JSONResponse(
        content=export_data,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.json"',
        },
    )


@router.post("/{workflow_id}/duplicate", response_model=WorkflowResponse, status_code=201)
async def duplicate_workflow(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Duplicate a workflow with '(Copy)' suffix."""
    async with get_db() as db:
        data = await _verify_ownership(db, workflow_id, user["id"])

        new_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO workflows (id, name, description, project_id, nodes_json, edges_json, viewport_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id,
                data["name"] + " (Copy)",
                data["description"],
                data["project_id"],
                data["nodes_json"],
                data["edges_json"],
                data["viewport_json"],
                user["id"],
                now,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM workflows WHERE id = ?", (new_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)
