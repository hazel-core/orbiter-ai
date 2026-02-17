"""Prompt templates CRUD REST API, version history, and test/compare endpoints."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user
from orbiter_web.sanitize import sanitize_html

router = APIRouter(prefix="/api/prompt-templates", tags=["prompt_templates"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class TemplateCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    content: str = ""
    variables_json: str = "{}"


class TemplateUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    content: str | None = None
    variables_json: str | None = None


class TemplateResponse(BaseModel):
    id: str
    name: str
    content: str
    variables_json: str
    user_id: str
    created_at: str
    updated_at: str


class VersionResponse(BaseModel):
    id: str
    template_id: str
    content: str
    variables_json: str
    version_number: int
    user_id: str
    created_at: str


class TestPromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    variables: dict[str, str] = {}
    provider_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)


class TestPromptResponse(BaseModel):
    output: str
    model: str
    tokens_used: int | None = None
    response_time_ms: int | None = None


class CompareModelItem(BaseModel):
    provider_id: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)


class CompareRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    variables: dict[str, str] = {}
    models: list[CompareModelItem] = Field(..., min_length=2, max_length=3)


class CompareResultItem(BaseModel):
    provider_id: str
    model_name: str
    output: str
    tokens_used: int | None = None
    response_time_ms: int | None = None
    error: str | None = None


class CompareResponse(BaseModel):
    results: list[CompareResultItem]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


async def _verify_ownership(
    db: Any, template_id: str, user_id: str
) -> dict[str, Any]:
    """Verify template exists and belongs to user. Returns row dict or raises 404."""
    cursor = await db.execute(
        "SELECT * FROM prompt_templates WHERE id = ? AND user_id = ?",
        (template_id, user_id),
    )
    row = await cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return _row_to_dict(row)


async def _create_version(db: Any, template_id: str, content: str, variables_json: str, user_id: str) -> None:
    """Create a new version entry for a template."""
    # Get next version number
    cursor = await db.execute(
        "SELECT COALESCE(MAX(version_number), 0) + 1 FROM prompt_versions WHERE template_id = ?",
        (template_id,),
    )
    row = await cursor.fetchone()
    next_version = row[0]

    version_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    await db.execute(
        """
        INSERT INTO prompt_versions (id, template_id, content, variables_json, version_number, user_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (version_id, template_id, content, variables_json, next_version, user_id, now),
    )


async def _send_prompt_to_model(
    provider_id: str, model_name: str, prompt: str, user_id: str
) -> dict[str, Any]:
    """Send a prompt to a specific model and return the result dict."""
    import time

    import httpx

    from orbiter_web.crypto import decrypt_api_key

    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM providers WHERE id = ? AND user_id = ?",
            (provider_id, user_id),
        )
        provider_row = await cursor.fetchone()
        if provider_row is None:
            return {"provider_id": provider_id, "model_name": model_name, "output": "", "error": "Provider not found"}
        provider = dict(provider_row)

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
            return {"provider_id": provider_id, "model_name": model_name, "output": "", "error": "No API key configured"}

    provider_type = provider["provider_type"]
    base_url = provider.get("base_url") or ""

    start_time = time.monotonic()
    tokens_used = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if provider_type in ("openai", "custom"):
                url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024},
                )
            elif provider_type == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                resp = await client.post(
                    url,
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
                    json={"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024},
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
                return {"provider_id": provider_id, "model_name": model_name, "output": "", "error": f"Unsupported provider type: {provider_type}"}

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if resp.status_code >= 400:
                error_text = resp.text[:500]
                return {"provider_id": provider_id, "model_name": model_name, "output": "", "error": f"API error ({resp.status_code}): {error_text}", "response_time_ms": elapsed_ms}

            data = resp.json()

            output = ""
            if provider_type in ("openai", "custom"):
                output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens")
            elif provider_type == "anthropic":
                content_blocks = data.get("content", [])
                output = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
                usage = data.get("usage", {})
                tokens_used = (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0)
            elif provider_type == "gemini":
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    output = "".join(p.get("text", "") for p in parts)
            elif provider_type == "ollama":
                output = data.get("response", "")

            return {
                "provider_id": provider_id,
                "model_name": model_name,
                "output": output,
                "tokens_used": tokens_used,
                "response_time_ms": elapsed_ms,
            }

    except httpx.HTTPError as exc:
        return {"provider_id": provider_id, "model_name": model_name, "output": "", "error": f"Connection error: {exc!s}"}


# ---------------------------------------------------------------------------
# Template CRUD Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[TemplateResponse])
async def list_templates(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all prompt templates for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM prompt_templates WHERE user_id = ? ORDER BY updated_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("", response_model=TemplateResponse, status_code=201)
async def create_template(
    body: TemplateCreate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Create a new prompt template and its initial version."""
    async with get_db() as db:
        template_id = str(uuid.uuid4())
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        await db.execute(
            """
            INSERT INTO prompt_templates (id, name, content, variables_json, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (template_id, sanitize_html(body.name), sanitize_html(body.content), body.variables_json, user["id"], now, now),
        )

        # Create initial version
        await _create_version(db, template_id, body.content, body.variables_json, user["id"])

        await db.commit()

        cursor = await db.execute("SELECT * FROM prompt_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single prompt template by ID."""
    async with get_db() as db:
        return await _verify_ownership(db, template_id, user["id"])


@router.put("/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    body: TemplateUpdate,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update a prompt template. Creates a new version if content changes."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    for field in ("name", "content"):
        if field in updates and isinstance(updates[field], str):
            updates[field] = sanitize_html(updates[field])

    async with get_db() as db:
        existing = await _verify_ownership(db, template_id, user["id"])

        # Create version if content or variables changed
        new_content = updates.get("content", existing["content"])
        new_variables = updates.get("variables_json", existing["variables_json"])
        if new_content != existing["content"] or new_variables != existing["variables_json"]:
            await _create_version(db, template_id, new_content, new_variables, user["id"])

        updates["updated_at"] = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = [*list(updates.values()), template_id]

        await db.execute(
            f"UPDATE prompt_templates SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM prompt_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.delete("/{template_id}", status_code=204)
async def delete_template(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Delete a prompt template and all its versions."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        await db.execute("DELETE FROM prompt_versions WHERE template_id = ?", (template_id,))
        await db.execute("DELETE FROM prompt_templates WHERE id = ?", (template_id,))
        await db.commit()


# ---------------------------------------------------------------------------
# Version History Endpoints
# ---------------------------------------------------------------------------


@router.get("/{template_id}/versions", response_model=list[VersionResponse])
async def list_versions(
    template_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all versions of a template, newest first."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM prompt_versions WHERE template_id = ? ORDER BY version_number DESC",
            (template_id,),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.get("/{template_id}/versions/{version_id}", response_model=VersionResponse)
async def get_version(
    template_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single version by ID."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM prompt_versions WHERE id = ? AND template_id = ?",
            (version_id, template_id),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Version not found")
        return _row_to_dict(row)


@router.post("/{template_id}/versions/{version_id}/restore", response_model=TemplateResponse)
async def restore_version(
    template_id: str,
    version_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Restore a previous version as the current template content."""
    async with get_db() as db:
        await _verify_ownership(db, template_id, user["id"])
        cursor = await db.execute(
            "SELECT * FROM prompt_versions WHERE id = ? AND template_id = ?",
            (version_id, template_id),
        )
        version_row = await cursor.fetchone()
        if version_row is None:
            raise HTTPException(status_code=404, detail="Version not found")
        version = _row_to_dict(version_row)

        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

        # Update template with version content
        await db.execute(
            "UPDATE prompt_templates SET content = ?, variables_json = ?, updated_at = ? WHERE id = ?",
            (version["content"], version["variables_json"], now, template_id),
        )

        # Create a new version to record the restore action
        await _create_version(db, template_id, version["content"], version["variables_json"], user["id"])

        await db.commit()

        cursor = await db.execute("SELECT * FROM prompt_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


# ---------------------------------------------------------------------------
# Test Prompt Endpoint
# ---------------------------------------------------------------------------


@router.post("/test", response_model=TestPromptResponse)
async def test_prompt(
    body: TestPromptRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Send a prompt to the specified model and return the response."""
    # Fill in template variables
    prompt = body.prompt
    for var_name, var_value in body.variables.items():
        prompt = prompt.replace("{{" + var_name + "}}", var_value)

    result = await _send_prompt_to_model(body.provider_id, body.model_name, prompt, user["id"])

    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])

    return {
        "output": result["output"],
        "model": body.model_name,
        "tokens_used": result.get("tokens_used"),
        "response_time_ms": result.get("response_time_ms"),
    }


# ---------------------------------------------------------------------------
# Model Comparison Endpoint
# ---------------------------------------------------------------------------


@router.post("/compare", response_model=CompareResponse)
async def compare_models(
    body: CompareRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Send the same prompt to 2-3 models in parallel and return results side-by-side."""
    # Fill in template variables
    prompt = body.prompt
    for var_name, var_value in body.variables.items():
        prompt = prompt.replace("{{" + var_name + "}}", var_value)

    # Run all model calls in parallel
    tasks = [
        _send_prompt_to_model(m.provider_id, m.model_name, prompt, user["id"])
        for m in body.models
    ]
    results = await asyncio.gather(*tasks)

    return {"results": results}
