"""Crews CRUD REST API and crew execution."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/crews", tags=["crews"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CrewTaskCreate(BaseModel):
    agent_id: str = Field(..., min_length=1)
    task_description: str = ""
    expected_output: str = ""
    task_order: int = 0
    dependencies_json: str = "[]"


class CrewTaskUpdate(BaseModel):
    agent_id: str | None = Field(None, min_length=1)
    task_description: str | None = None
    expected_output: str | None = None
    task_order: int | None = None
    dependencies_json: str | None = None


class CrewTaskResponse(BaseModel):
    id: str
    crew_id: str
    agent_id: str
    task_description: str
    expected_output: str
    task_order: int
    dependencies_json: str
    created_at: str
    updated_at: str


class CrewCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    description: str = ""
    process_type: str = Field("sequential", pattern=r"^(sequential|parallel)$")
    config_json: str = "{}"


class CrewUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    process_type: str | None = Field(None, pattern=r"^(sequential|parallel)$")
    config_json: str | None = None


class CrewResponse(BaseModel):
    id: str
    name: str
    description: str
    process_type: str
    config_json: str
    project_id: str
    user_id: str
    created_at: str
    updated_at: str
    tasks: list[CrewTaskResponse] = []


class CrewRunRequest(BaseModel):
    input: str = ""


class CrewRunTaskResult(BaseModel):
    task_id: str
    agent_id: str
    task_description: str
    status: str
    output: str = ""
    error: str = ""


class CrewRunResponse(BaseModel):
    crew_id: str
    status: str
    process_type: str
    results: list[CrewRunTaskResult]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_crew_ownership(db: Any, crew_id: str, user_id: str) -> dict[str, Any]:
    """Verify crew exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM crews WHERE id = ? AND user_id = ?",
        (crew_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Crew not found")
    return _row_to_dict(row)


async def _get_crew_tasks(db: Any, crew_id: str) -> list[dict[str, Any]]:
    """Return tasks for a crew, ordered by task_order."""
    cursor = await db.execute(
        "SELECT * FROM crew_tasks WHERE crew_id = ? ORDER BY task_order ASC",
        (crew_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


async def _crew_with_tasks(db: Any, crew_id: str) -> dict[str, Any]:
    """Return crew dict with nested tasks list."""
    cursor = await db.execute("SELECT * FROM crews WHERE id = ?", (crew_id,))
    row = await cursor.fetchone()
    crew = _row_to_dict(row)
    crew["tasks"] = await _get_crew_tasks(db, crew_id)
    return crew


async def _execute_single_task(
    task: dict[str, Any], crew_input: str, previous_outputs: list[str]
) -> dict[str, Any]:
    """Execute a single crew task by calling the agent's configured model.

    Returns a result dict with status/output/error.
    """
    import time

    import httpx

    from orbiter_web.crypto import decrypt_api_key

    agent_id = task["agent_id"]
    task_description = task["task_description"] or "Complete the assigned task."
    expected_output = task["expected_output"]

    async with get_db() as db:
        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        agent_row = await cursor.fetchone()
        if agent_row is None:
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "Agent not found",
            }
        agent = _row_to_dict(agent_row)

        # Resolve provider and key
        provider_id = agent.get("model_provider", "")
        model_name = agent.get("model_name", "")
        if not provider_id or not model_name:
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "Agent has no model configured",
            }

        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ?",
            (provider_id,),
        )
        provider_row = await cursor.fetchone()
        if provider_row is None:
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "Provider not found",
            }
        provider = _row_to_dict(provider_row)

        api_key = ""
        if provider.get("encrypted_api_key"):
            api_key = decrypt_api_key(provider["encrypted_api_key"])
        else:
            cursor = await db.execute(
                "SELECT encrypted_api_key FROM provider_keys WHERE provider_id = ? AND status = 'active' LIMIT 1",
                (provider_id,),
            )
            key_row = await cursor.fetchone()
            if key_row:
                api_key = decrypt_api_key(key_row["encrypted_api_key"])

        if not api_key:
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "failed",
                "output": "",
                "error": "No API key configured for provider",
            }

    # Build the prompt
    context_parts: list[str] = []
    if agent.get("instructions"):
        context_parts.append(f"System instructions: {agent['instructions']}")
    if crew_input:
        context_parts.append(f"Crew input: {crew_input}")
    if previous_outputs:
        context_parts.append(
            "Previous task outputs:\n" + "\n---\n".join(previous_outputs)
        )

    prompt = f"{task_description}"
    if expected_output:
        prompt += f"\n\nExpected output: {expected_output}"
    if context_parts:
        prompt = "\n".join(context_parts) + "\n\n" + prompt

    provider_type = provider["provider_type"]
    base_url = provider.get("base_url") or ""
    start_time = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if provider_type in ("openai", "custom"):
                url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
                )
            elif provider_type == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                resp = await client.post(
                    url,
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
                )
            elif provider_type == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                )
            elif provider_type == "ollama":
                url = (base_url or "http://localhost:11434") + "/api/generate"
                resp = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
            else:
                return {
                    "task_id": task["id"],
                    "agent_id": agent_id,
                    "task_description": task_description,
                    "status": "failed",
                    "output": "",
                    "error": f"Unsupported provider type: {provider_type}",
                }

            elapsed_ms = int((time.monotonic() - start_time) * 1000)  # noqa: F841

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                return {
                    "task_id": task["id"],
                    "agent_id": agent_id,
                    "task_description": task_description,
                    "status": "failed",
                    "output": "",
                    "error": f"API error ({resp.status_code}): {error_text}",
                }

            data = resp.json()

            output = ""
            if provider_type in ("openai", "custom"):
                output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif provider_type == "anthropic":
                content_blocks = data.get("content", [])
                output = "".join(
                    b.get("text", "") for b in content_blocks if b.get("type") == "text"
                )
            elif provider_type == "gemini":
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    output = "".join(p.get("text", "") for p in parts)
            elif provider_type == "ollama":
                output = data.get("response", "")

            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "task_description": task_description,
                "status": "completed",
                "output": output,
                "error": "",
            }

    except Exception as exc:
        return {
            "task_id": task["id"],
            "agent_id": agent_id,
            "task_description": task_description,
            "status": "failed",
            "output": "",
            "error": f"Connection error: {exc!s}",
        }


# ---------------------------------------------------------------------------
# Crew CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[CrewResponse])
async def list_crews(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all crews for the current user, optionally filtered by project."""
    async with get_db() as db:
        if project_id:
            cursor = await db.execute(
                "SELECT * FROM crews WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                (user["id"], project_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM crews WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            crew = _row_to_dict(r)
            crew["tasks"] = await _get_crew_tasks(db, crew["id"])
            result.append(crew)
        return result


@router.post("", response_model=CrewResponse, status_code=201)
async def create_crew(
    body: CrewCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new crew."""
    async with get_db() as db:
        # Verify project exists and belongs to user
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        crew_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO crews (id, name, description, process_type, config_json, project_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (crew_id, body.name, body.description, body.process_type, body.config_json, body.project_id, user["id"], now, now),
        )
        await db.commit()

        return await _crew_with_tasks(db, crew_id)


@router.get("/{crew_id}", response_model=CrewResponse)
async def get_crew(
    crew_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single crew by ID with its tasks."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        return await _crew_with_tasks(db, crew_id)


@router.put("/{crew_id}", response_model=CrewResponse)
async def update_crew(
    crew_id: str,
    body: CrewUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a crew's fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), crew_id]

        await db.execute(f"UPDATE crews SET {set_clause} WHERE id = ?", values)
        await db.commit()

        return await _crew_with_tasks(db, crew_id)


@router.delete("/{crew_id}", status_code=204)
async def delete_crew(
    crew_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a crew and its tasks (cascade)."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        await db.execute("DELETE FROM crews WHERE id = ?", (crew_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Crew task endpoints
# ---------------------------------------------------------------------------


@router.get("/{crew_id}/tasks", response_model=list[CrewTaskResponse])
async def list_crew_tasks(
    crew_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all tasks in a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        return await _get_crew_tasks(db, crew_id)


@router.post("/{crew_id}/tasks", response_model=CrewTaskResponse, status_code=201)
async def add_crew_task(
    crew_id: str,
    body: CrewTaskCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Add a task to a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        # Verify agent exists
        cursor = await db.execute(
            "SELECT id FROM agents WHERE id = ? AND user_id = ?",
            (body.agent_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        task_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO crew_tasks (id, crew_id, agent_id, task_description, expected_output, task_order, dependencies_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, crew_id, body.agent_id, body.task_description, body.expected_output, body.task_order, body.dependencies_json, now, now),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM crew_tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.put("/{crew_id}/tasks/{task_id}", response_model=CrewTaskResponse)
async def update_crew_task(
    crew_id: str,
    task_id: str,
    body: CrewTaskUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a crew task."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])

        # Verify task belongs to crew
        cursor = await db.execute(
            "SELECT id FROM crew_tasks WHERE id = ? AND crew_id = ?",
            (task_id, crew_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Crew task not found")

        if "agent_id" in updates:
            cursor = await db.execute(
                "SELECT id FROM agents WHERE id = ? AND user_id = ?",
                (updates["agent_id"], user["id"]),
            )
            if await cursor.fetchone() is None:
                raise HTTPException(status_code=404, detail="Agent not found")

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), task_id]

        await db.execute(f"UPDATE crew_tasks SET {set_clause} WHERE id = ?", values)
        await db.commit()

        cursor = await db.execute("SELECT * FROM crew_tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{crew_id}/tasks/{task_id}", status_code=204)
async def delete_crew_task(
    crew_id: str,
    task_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Remove a task from a crew."""
    async with get_db() as db:
        await _verify_crew_ownership(db, crew_id, user["id"])
        cursor = await db.execute(
            "SELECT id FROM crew_tasks WHERE id = ? AND crew_id = ?",
            (task_id, crew_id),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Crew task not found")
        await db.execute("DELETE FROM crew_tasks WHERE id = ?", (task_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Crew execution
# ---------------------------------------------------------------------------


@router.post("/{crew_id}/run", response_model=CrewRunResponse)
async def run_crew(
    crew_id: str,
    body: CrewRunRequest | None = None,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Execute the crew — runs all tasks sequentially or in parallel based on process_type."""
    crew_input = body.input if body else ""

    async with get_db() as db:
        crew = await _verify_crew_ownership(db, crew_id, user["id"])
        tasks = await _get_crew_tasks(db, crew_id)

    if not tasks:
        raise HTTPException(status_code=422, detail="Crew has no tasks to run")

    process_type = crew["process_type"]
    results: list[dict[str, Any]] = []

    if process_type == "parallel":
        # Run all tasks concurrently
        coros = [_execute_single_task(t, crew_input, []) for t in tasks]
        results = await asyncio.gather(*coros)
        results = list(results)
    else:
        # Sequential: feed previous outputs as context
        previous_outputs: list[str] = []
        for t in tasks:
            result = await _execute_single_task(t, crew_input, previous_outputs)
            results.append(result)
            if result["status"] == "completed" and result["output"]:
                previous_outputs.append(result["output"])

    # Determine overall status
    all_completed = all(r["status"] == "completed" for r in results)
    overall_status = "completed" if all_completed else "partial"

    return {
        "crew_id": crew_id,
        "status": overall_status,
        "process_type": process_type,
        "results": results,
    }
