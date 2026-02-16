"""Workflow execution REST + WebSocket endpoints."""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from orbiter_web.database import get_db
from orbiter_web.engine import cancel_run, execute_workflow
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/workflows", tags=["workflow-runs"])

# Keep references to background tasks so they aren't garbage-collected.
_background_tasks: set[asyncio.Task[Any]] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_user_from_cookie(websocket: WebSocket) -> dict[str, Any] | None:
    """Extract user from session cookie on the WebSocket connection."""
    session_id = websocket.cookies.get("orbiter_session")
    if not session_id:
        return None

    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT u.id, u.email, u.created_at
            FROM sessions s
            JOIN users u ON u.id = s.user_id
            WHERE s.id = ? AND s.expires_at > datetime('now')
            """,
            (session_id,),
        )
        row = await cursor.fetchone()

    return dict(row) if row else None


async def _verify_workflow_ownership(
    db: Any, workflow_id: str, user_id: str
) -> dict[str, Any]:
    """Return workflow row or raise 404."""
    cursor = await db.execute(
        "SELECT * FROM workflows WHERE id = ? AND user_id = ?",
        (workflow_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return dict(row)


# ---------------------------------------------------------------------------
# POST /api/workflows/:id/run — start execution
# ---------------------------------------------------------------------------


@router.post("/{workflow_id}/run")
async def start_workflow_run(
    workflow_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Start executing a workflow. Returns the run_id immediately."""
    async with get_db() as db:
        wf = await _verify_workflow_ownership(db, workflow_id, user["id"])

        nodes = json.loads(wf["nodes_json"] or "[]")
        edges = json.loads(wf["edges_json"] or "[]")

        if not nodes:
            raise HTTPException(status_code=422, detail="Workflow has no nodes")

        run_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            "INSERT INTO workflow_runs (id, workflow_id, status, user_id, created_at) VALUES (?, ?, 'pending', ?, ?)",
            (run_id, workflow_id, user["id"], now),
        )
        await db.commit()

    # Fire-and-forget: run the engine in the background.
    task = asyncio.create_task(
        execute_workflow(run_id, workflow_id, user["id"], nodes, edges)
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return {"run_id": run_id, "status": "pending"}


# ---------------------------------------------------------------------------
# DELETE /api/workflows/:id/runs/:runId — cancel execution
# ---------------------------------------------------------------------------


@router.delete("/{workflow_id}/runs/{run_id}")
async def cancel_workflow_run(
    workflow_id: str,
    run_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, str]:
    """Cancel a running workflow execution."""
    async with get_db() as db:
        await _verify_workflow_ownership(db, workflow_id, user["id"])

        cursor = await db.execute(
            "SELECT status FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")

        if row["status"] not in ("pending", "running"):
            raise HTTPException(
                status_code=422,
                detail=f"Run is already {row['status']}",
            )

    cancelled = cancel_run(run_id)
    if not cancelled:
        # Run might have finished between the check and cancel.
        async with get_db() as db:
            await db.execute(
                "UPDATE workflow_runs SET status = 'cancelled', completed_at = datetime('now') WHERE id = ?",
                (run_id,),
            )
            await db.commit()

    return {"status": "cancelled"}


# ---------------------------------------------------------------------------
# GET /api/workflows/:id/runs/:runId/nodes/:nodeId — node execution data
# ---------------------------------------------------------------------------


@router.get("/{workflow_id}/runs/{run_id}/nodes/{node_id}")
async def get_node_execution(
    workflow_id: str,
    run_id: str,
    node_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return execution data for a specific node within a run."""
    async with get_db() as db:
        await _verify_workflow_ownership(db, workflow_id, user["id"])

        # Verify the run belongs to this workflow.
        cursor = await db.execute(
            "SELECT id FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Run not found")

        cursor = await db.execute(
            "SELECT id, run_id, node_id, status, input_json, output_json, logs_text, started_at, completed_at, error, token_usage_json FROM workflow_run_logs WHERE run_id = ? AND node_id = ?",
            (run_id, node_id),
        )
        row = await cursor.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Node execution not found")

    result = dict(row)
    # Parse JSON fields for the response.
    for field in ("input_json", "output_json", "token_usage_json"):
        if result.get(field):
            result[field] = json.loads(result[field])

    return result


# ---------------------------------------------------------------------------
# WebSocket /api/workflows/:id/runs/:runId/stream — live execution events
# ---------------------------------------------------------------------------


@router.websocket("/{workflow_id}/runs/{run_id}/stream")
async def stream_workflow_run(
    websocket: WebSocket,
    workflow_id: str,
    run_id: str,
) -> None:
    """WebSocket endpoint for streaming workflow execution events."""
    user = await _get_user_from_cookie(websocket)
    if user is None:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    # Verify ownership.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM workflows WHERE id = ? AND user_id = ?",
            (workflow_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            await websocket.close(code=4004, reason="Workflow not found")
            return

        cursor = await db.execute(
            "SELECT status FROM workflow_runs WHERE id = ? AND workflow_id = ?",
            (run_id, workflow_id),
        )
        run_row = await cursor.fetchone()
        if run_row is None:
            await websocket.close(code=4004, reason="Run not found")
            return

    await websocket.accept()

    # If run is already done, send the final status and close.
    if run_row["status"] in ("completed", "failed", "cancelled"):
        await websocket.send_json(
            {"type": "execution_completed", "status": run_row["status"]}
        )
        await websocket.close()
        return

    # Create an event queue to receive events from the engine.
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def event_callback(event: dict[str, Any]) -> None:
        await event_queue.put(event)

    # If the run hasn't started yet, start it with our callback.
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT status FROM workflow_runs WHERE id = ?", (run_id,)
        )
        current = await cursor.fetchone()

    if current and current["status"] == "pending":
        async with get_db() as db:
            cursor = await db.execute(
                "SELECT nodes_json, edges_json FROM workflows WHERE id = ?",
                (workflow_id,),
            )
            wf = await cursor.fetchone()

        if wf:
            nodes = json.loads(wf["nodes_json"] or "[]")
            edges = json.loads(wf["edges_json"] or "[]")
            task = asyncio.create_task(
                execute_workflow(
                    run_id, workflow_id, user["id"], nodes, edges, event_callback
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    else:
        # Run is already in-progress (started via POST). We can't retroactively
        # attach a callback, so poll the DB for updates instead.
        task = asyncio.create_task(_poll_run_status(run_id, event_queue))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    try:
        while True:
            event = await event_queue.get()
            await websocket.send_json(event)
            if event.get("type") == "execution_completed":
                break
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


async def _poll_run_status(
    run_id: str, queue: asyncio.Queue[dict[str, Any]]
) -> None:
    """Fallback poller for runs started before the WS connected."""
    seen_nodes: set[str] = set()

    while True:
        await asyncio.sleep(0.3)

        async with get_db() as db:
            cursor = await db.execute(
                "SELECT status FROM workflow_runs WHERE id = ?", (run_id,)
            )
            run_row = await cursor.fetchone()
            if run_row is None:
                await queue.put({"type": "execution_completed", "status": "failed", "error": "Run not found"})
                return

            # Check for new node completions.
            cursor = await db.execute(
                "SELECT node_id, status, output_json, error FROM workflow_run_logs WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            )
            logs = await cursor.fetchall()

        for log in logs:
            nid = log["node_id"]
            if nid not in seen_nodes and log["status"] != "pending":
                if log["status"] == "running":
                    await queue.put({"type": "node_started", "node_id": nid})
                elif log["status"] == "completed":
                    output = json.loads(log["output_json"]) if log["output_json"] else {}
                    await queue.put({"type": "node_completed", "node_id": nid, "output": output})
                    seen_nodes.add(nid)
                elif log["status"] == "failed":
                    await queue.put({"type": "node_failed", "node_id": nid, "error": log["error"] or ""})
                    seen_nodes.add(nid)

        if run_row["status"] in ("completed", "failed", "cancelled"):
            await queue.put({"type": "execution_completed", "status": run_row["status"]})
            return
