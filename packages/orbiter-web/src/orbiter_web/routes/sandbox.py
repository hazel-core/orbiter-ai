"""Sandbox REST API.

Provides endpoints for executing Python code in a sandboxed environment
and managing sandbox configuration.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user
from orbiter_web.services.sandbox import SandboxConfig, execute_code

router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ExecuteRequest(BaseModel):
    code: str = Field(..., min_length=1, description="Python code to execute")
    timeout_seconds: int | None = Field(
        None, ge=1, le=120, description="Execution timeout override"
    )
    allowed_libraries: list[str] | None = Field(
        None, description="Override allowed libraries list"
    )


class GeneratedFile(BaseModel):
    name: str
    size_bytes: int
    content_base64: str | None = None


class ExecuteResponse(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    generated_files: list[GeneratedFile] = []
    execution_time_ms: float = 0.0


class SandboxConfigResponse(BaseModel):
    id: str
    user_id: str
    allowed_libraries: list[str]
    timeout_seconds: int
    memory_limit_mb: int
    created_at: str
    updated_at: str


class SandboxConfigUpdate(BaseModel):
    allowed_libraries: list[str] | None = None
    timeout_seconds: int | None = Field(None, ge=1, le=120)
    memory_limit_mb: int | None = Field(None, ge=64, le=1024)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_or_create_config(
    db: Any, user_id: str
) -> dict[str, Any]:
    """Get sandbox config for user, creating default if not exists."""
    cursor = await db.execute(
        "SELECT * FROM sandbox_configs WHERE user_id = ?", (user_id,)
    )
    row = await cursor.fetchone()
    if row is not None:
        return dict(row)

    config_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    default_libs = json.dumps(SandboxConfig().allowed_libraries)

    await db.execute(
        """INSERT INTO sandbox_configs
           (id, user_id, allowed_libraries, timeout_seconds, memory_limit_mb, created_at, updated_at)
           VALUES (?, ?, ?, 30, 256, ?, ?)""",
        (config_id, user_id, default_libs, now, now),
    )
    await db.commit()

    cursor = await db.execute(
        "SELECT * FROM sandbox_configs WHERE id = ?", (config_id,)
    )
    return dict(await cursor.fetchone())


def _parse_config_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a sandbox_configs row, deserializing JSON fields."""
    result = dict(row)
    try:
        result["allowed_libraries"] = json.loads(result["allowed_libraries"])
    except (json.JSONDecodeError, TypeError):
        result["allowed_libraries"] = SandboxConfig().allowed_libraries
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/execute", response_model=ExecuteResponse)
async def execute_sandbox_code(
    body: ExecuteRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment."""
    # Load user's sandbox config
    async with get_db() as db:
        config_row = await _get_or_create_config(db, user["id"])
    config_data = _parse_config_row(config_row)

    # Build sandbox config with optional overrides
    sandbox_config = SandboxConfig(
        allowed_libraries=body.allowed_libraries or config_data["allowed_libraries"],
        timeout_seconds=body.timeout_seconds or config_data["timeout_seconds"],
        memory_limit_mb=config_data["memory_limit_mb"],
    )

    result = execute_code(body.code, sandbox_config)

    return {
        "success": result.success,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error": result.error,
        "generated_files": result.generated_files,
        "execution_time_ms": result.execution_time_ms,
    }


@router.get("/config", response_model=SandboxConfigResponse)
async def get_sandbox_config(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Get sandbox configuration for the current user."""
    async with get_db() as db:
        config_row = await _get_or_create_config(db, user["id"])
    return _parse_config_row(config_row)


@router.put("/config", response_model=SandboxConfigResponse)
async def update_sandbox_config(
    body: SandboxConfigUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update sandbox configuration for the current user."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    async with get_db() as db:
        await _get_or_create_config(db, user["id"])

        # Serialize allowed_libraries to JSON
        if "allowed_libraries" in updates:
            updates["allowed_libraries"] = json.dumps(updates["allowed_libraries"])

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        updates["updated_at"] = now

        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), user["id"]]

        await db.execute(
            f"UPDATE sandbox_configs SET {set_clause} WHERE user_id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute(
            "SELECT * FROM sandbox_configs WHERE user_id = ?", (user["id"],)
        )
        row = await cursor.fetchone()
    return _parse_config_row(dict(row))
