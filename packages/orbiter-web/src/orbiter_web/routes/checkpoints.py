"""Checkpoint save/restore for long-running runs."""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/runs", tags=["checkpoints"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CheckpointCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    step_number: int = Field(default=0, ge=0)
    state_blob: dict[str, Any] | list[Any] | str = Field(
        ..., description="Execution state to persist (JSON-serialisable)"
    )


class CheckpointOut(BaseModel):
    id: str
    run_id: str
    name: str
    step_number: int
    state_blob: Any
    created_at: str


class CheckpointListOut(BaseModel):
    checkpoints: list[CheckpointOut]
    total: int


class RestoreOut(BaseModel):
    checkpoint: CheckpointOut
    message: str = "Run status reset to 'pending' for resume"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _verify_run_ownership(db: Any, run_id: str, user_id: str) -> dict[str, Any]:
    """Return workflow_runs row or raise 404."""
    cursor = await db.execute(
        "SELECT * FROM workflow_runs WHERE id = ? AND user_id = ?",
        (run_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return dict(row)


# ---------------------------------------------------------------------------
# POST /api/runs/:runId/checkpoints — save checkpoint
# ---------------------------------------------------------------------------


@router.post("/{run_id}/checkpoints", status_code=201)
async def create_checkpoint(
    run_id: str,
    body: CheckpointCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> CheckpointOut:
    async with get_db() as db:
        await _verify_run_ownership(db, run_id, user["id"])

        cp_id = str(uuid.uuid4())
        blob = json.dumps(body.state_blob) if not isinstance(body.state_blob, str) else body.state_blob
        now = datetime.now(UTC).isoformat()

        await db.execute(
            """
            INSERT INTO checkpoints (id, run_id, name, step_number, state_blob, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cp_id, run_id, body.name, body.step_number, blob, now),
        )
        await db.commit()

    return CheckpointOut(
        id=cp_id,
        run_id=run_id,
        name=body.name,
        step_number=body.step_number,
        state_blob=body.state_blob,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# GET /api/runs/:runId/checkpoints — list checkpoints for a run
# ---------------------------------------------------------------------------


@router.get("/{run_id}/checkpoints")
async def list_checkpoints(
    run_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> CheckpointListOut:
    async with get_db() as db:
        await _verify_run_ownership(db, run_id, user["id"])

        cursor = await db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE run_id = ?", (run_id,)
        )
        total = (await cursor.fetchone())[0]

        cursor = await db.execute(
            """
            SELECT id, run_id, name, step_number, state_blob, created_at
            FROM checkpoints
            WHERE run_id = ?
            ORDER BY step_number DESC, created_at DESC
            LIMIT ? OFFSET ?
            """,
            (run_id, limit, offset),
        )
        rows = await cursor.fetchall()

    checkpoints = []
    for r in rows:
        row = dict(r)
        blob = row["state_blob"]
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            blob = json.loads(blob)
        checkpoints.append(
            CheckpointOut(
                id=row["id"],
                run_id=row["run_id"],
                name=row["name"],
                step_number=row["step_number"],
                state_blob=blob,
                created_at=row["created_at"],
            )
        )

    return CheckpointListOut(checkpoints=checkpoints, total=total)


# ---------------------------------------------------------------------------
# POST /api/runs/:runId/checkpoints/:cpId/restore — restore from checkpoint
# ---------------------------------------------------------------------------


@router.post("/{run_id}/checkpoints/{cp_id}/restore")
async def restore_checkpoint(
    run_id: str,
    cp_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> RestoreOut:
    async with get_db() as db:
        await _verify_run_ownership(db, run_id, user["id"])

        cursor = await db.execute(
            "SELECT * FROM checkpoints WHERE id = ? AND run_id = ?",
            (cp_id, run_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        cp = dict(row)

        # Reset the run to pending so it can be resumed from this checkpoint.
        await db.execute(
            "UPDATE workflow_runs SET status = 'pending' WHERE id = ?",
            (run_id,),
        )
        await db.commit()

    blob = cp["state_blob"]
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        blob = json.loads(blob)

    return RestoreOut(
        checkpoint=CheckpointOut(
            id=cp["id"],
            run_id=cp["run_id"],
            name=cp["name"],
            step_number=cp["step_number"],
            state_blob=blob,
            created_at=cp["created_at"],
        ),
    )


# ---------------------------------------------------------------------------
# Auto-checkpoint & retention helpers (used by engine / background tasks)
# ---------------------------------------------------------------------------

# Default configuration — can be overridden per run via run config.
DEFAULT_AUTO_CHECKPOINT_INTERVAL = 5  # every N steps
DEFAULT_RETENTION_DAYS = 30  # delete checkpoints older than N days


async def maybe_auto_checkpoint(
    run_id: str,
    step_number: int,
    state_blob: dict[str, Any] | str,
    interval: int = DEFAULT_AUTO_CHECKPOINT_INTERVAL,
) -> str | None:
    """Create an auto-checkpoint if step_number is a multiple of *interval*.

    Returns the checkpoint id if created, else None.
    """
    if interval <= 0 or step_number % interval != 0:
        return None

    cp_id = str(uuid.uuid4())
    blob = json.dumps(state_blob) if not isinstance(state_blob, str) else state_blob
    now = datetime.now(UTC).isoformat()

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO checkpoints (id, run_id, name, step_number, state_blob, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cp_id, run_id, f"auto-step-{step_number}", step_number, blob, now),
        )
        await db.commit()

    return cp_id


async def cleanup_old_checkpoints(retention_days: int = DEFAULT_RETENTION_DAYS) -> int:
    """Delete checkpoints older than *retention_days*. Returns count deleted."""
    async with get_db() as db:
        cursor = await db.execute(
            """
            DELETE FROM checkpoints
            WHERE created_at < datetime('now', ? || ' days')
            """,
            (f"-{retention_days}",),
        )
        await db.commit()
        return cursor.rowcount
