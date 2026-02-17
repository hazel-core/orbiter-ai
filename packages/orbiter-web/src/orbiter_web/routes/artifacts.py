"""Artifact storage and retrieval REST API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from orbiter_web.config import settings
from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ArtifactResponse(BaseModel):
    id: str
    run_id: str | None = None
    agent_id: str | None = None
    filename: str
    file_type: str
    file_size: int
    storage_path: str
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact_dir() -> Path:
    """Return the configured artifact storage directory, creating it if needed."""
    p = Path(settings.artifact_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[ArtifactResponse])
async def list_artifacts(
    file_type: str | None = Query(None),
    agent_id: str | None = Query(None),
    run_id: str | None = Query(None),
    date_from: str | None = Query(None, description="ISO date, e.g. 2026-01-01"),
    date_to: str | None = Query(None, description="ISO date, e.g. 2026-12-31"),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List artifacts with optional filters."""
    query = "SELECT * FROM artifacts WHERE user_id = ?"
    params: list[Any] = [user["id"]]

    if file_type is not None:
        query += " AND file_type = ?"
        params.append(file_type)
    if agent_id is not None:
        query += " AND agent_id = ?"
        params.append(agent_id)
    if run_id is not None:
        query += " AND run_id = ?"
        params.append(run_id)
    if date_from is not None:
        query += " AND created_at >= ?"
        params.append(date_from)
    if date_to is not None:
        query += " AND created_at <= ?"
        params.append(date_to)

    query += " ORDER BY created_at DESC"

    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


@router.get("/{artifact_id}", response_model=ArtifactResponse)
async def get_artifact(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return metadata for a single artifact."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return dict(row)


@router.get("/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> FileResponse:
    """Download the artifact file content."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    artifact = dict(row)
    file_path = Path(artifact["storage_path"])

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found on disk")

    return FileResponse(
        path=str(file_path),
        filename=artifact["filename"],
        media_type="application/octet-stream",
    )


@router.delete("/{artifact_id}", status_code=204)
async def delete_artifact(
    artifact_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an artifact record and its file from disk."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM artifacts WHERE id = ? AND user_id = ?",
            (artifact_id, user["id"]),
        )
        row = await cursor.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Artifact not found")

        artifact = dict(row)

        # Remove from database
        await db.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        await db.commit()

    # Remove file from disk (best-effort)
    file_path = Path(artifact["storage_path"])
    if file_path.is_file():
        os.remove(file_path)
