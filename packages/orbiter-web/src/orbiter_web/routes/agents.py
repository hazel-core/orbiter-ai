"""Agents CRUD REST API."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/agents", tags=["agents"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    project_id: str = Field(..., min_length=1)
    description: str = ""
    instructions: str = ""
    model_provider: str = ""
    model_name: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    output_type_json: str = "{}"
    tools_json: str = "[]"
    handoffs_json: str = "[]"
    hooks_json: str = "{}"
    persona_role: str = ""
    persona_goal: str = ""
    persona_backstory: str = ""


class AgentUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    instructions: str | None = None
    model_provider: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int | None = None
    output_type_json: str | None = None
    tools_json: str | None = None
    handoffs_json: str | None = None
    hooks_json: str | None = None
    persona_role: str | None = None
    persona_goal: str | None = None
    persona_backstory: str | None = None


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    instructions: str
    model_provider: str
    model_name: str
    temperature: float | None
    max_tokens: int | None
    max_steps: int | None
    output_type_json: str
    tools_json: str
    handoffs_json: str
    hooks_json: str
    persona_role: str
    persona_goal: str
    persona_backstory: str
    project_id: str
    user_id: str
    created_at: str
    updated_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(
    db: Any, agent_id: str, user_id: str
) -> dict[str, Any]:
    """Verify agent exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM agents WHERE id = ? AND user_id = ?",
        (agent_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    project_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all agents for the current user, optionally filtered by project."""
    async with get_db() as db:
        if project_id:
            cursor = await db.execute(
                "SELECT * FROM agents WHERE user_id = ? AND project_id = ? ORDER BY created_at DESC",
                (user["id"], project_id),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM agents WHERE user_id = ? ORDER BY created_at DESC",
                (user["id"],),
            )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=AgentResponse, status_code=201)
async def create_agent(
    body: AgentCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new agent."""
    async with get_db() as db:
        # Verify project exists and belongs to user.
        cursor = await db.execute(
            "SELECT id FROM projects WHERE id = ? AND user_id = ?",
            (body.project_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Project not found")

        agent_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO agents (
                id, name, description, instructions,
                model_provider, model_name, temperature, max_tokens, max_steps,
                output_type_json, tools_json, handoffs_json, hooks_json,
                persona_role, persona_goal, persona_backstory,
                project_id, user_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id, body.name, body.description, body.instructions,
                body.model_provider, body.model_name, body.temperature,
                body.max_tokens, body.max_steps,
                body.output_type_json, body.tools_json, body.handoffs_json,
                body.hooks_json, body.persona_role, body.persona_goal,
                body.persona_backstory, body.project_id, user["id"], now, now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single agent by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, agent_id, user["id"])


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    body: AgentUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update an agent's editable fields."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    async with get_db() as db:
        await _verify_ownership(db, agent_id, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), agent_id]

        await db.execute(
            f"UPDATE agents SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM agents WHERE id = ?", (agent_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete an agent."""
    async with get_db() as db:
        await _verify_ownership(db, agent_id, user["id"])
        await db.execute("DELETE FROM agents WHERE id = ?", (agent_id,))
        await db.commit()
