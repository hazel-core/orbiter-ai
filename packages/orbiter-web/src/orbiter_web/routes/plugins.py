"""Plugins REST API.

Plugin manifest, isolation, install/uninstall, and local-directory loading.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from orbiter_web.database import get_db
from orbiter_web.routes.auth import get_current_user

router = APIRouter(prefix="/api/plugins", tags=["plugins"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_PLUGIN_TYPES = {"model", "tool", "strategy", "extension"}
VALID_STATUSES = {"installed", "enabled", "disabled", "error"}

MANIFEST_REQUIRED_FIELDS = {"name", "version", "type", "entry_point"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PluginManifest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(default="0.1.0")
    type: str = Field(default="extension")
    permissions: list[str] = Field(default_factory=list)
    entry_point: str = Field(default="main.py")
    description: str = ""
    author: str = ""


class PluginInstallRequest(BaseModel):
    manifest: PluginManifest
    directory: str = ""


class PluginLoadDirRequest(BaseModel):
    directory: str = Field(..., min_length=1)


class PluginResponse(BaseModel):
    id: str
    name: str
    version: str
    type: str
    manifest_json: str
    status: str
    entry_point: str
    directory: str
    permissions_json: str
    description: str
    author: str
    user_id: str
    installed_at: str


class PluginIsolationResult(BaseModel):
    success: bool
    output: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert an aiosqlite.Row to a plain dict."""
    return dict(row)


def _validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate that a manifest dict has required fields and valid type."""
    missing = MANIFEST_REQUIRED_FIELDS - set(manifest.keys())
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Manifest missing required fields: {', '.join(sorted(missing))}",
        )
    if manifest.get("type") not in VALID_PLUGIN_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid plugin type. Must be one of: {', '.join(sorted(VALID_PLUGIN_TYPES))}",
        )


async def _run_plugin_subprocess(
    entry_point: str, directory: str, timeout: float = 10.0
) -> PluginIsolationResult:
    """Run a plugin entry point in an isolated subprocess.

    Each plugin runs in its own subprocess with a timeout to prevent runaway
    processes. The subprocess is given only the plugin directory as cwd.
    """
    entry_path = Path(directory) / entry_point if directory else Path(entry_point)
    if not entry_path.exists():
        return PluginIsolationResult(
            success=False,
            error=f"Entry point not found: {entry_path}",
        )

    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            str(entry_path),
            "--validate",
            cwd=directory or None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        if proc.returncode == 0:
            return PluginIsolationResult(
                success=True,
                output=stdout.decode(errors="replace").strip(),
            )
        return PluginIsolationResult(
            success=False,
            output=stdout.decode(errors="replace").strip(),
            error=stderr.decode(errors="replace").strip() or f"Exit code {proc.returncode}",
        )
    except TimeoutError:
        proc.kill()  # type: ignore[possibly-undefined]
        return PluginIsolationResult(
            success=False, error="Plugin validation timed out"
        )
    except Exception as exc:
        return PluginIsolationResult(success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[PluginResponse])
async def list_plugins(
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> list[dict[str, Any]]:
    """Return all installed plugins for the current user."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM plugins WHERE user_id = ? ORDER BY installed_at DESC",
            (user["id"],),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]


@router.post("/install", response_model=PluginResponse, status_code=201)
async def install_plugin(
    body: PluginInstallRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Install a plugin from a manifest package.

    Validates the manifest, optionally runs the entry point in an isolated
    subprocess for validation, and stores the plugin in the database.
    """
    manifest = body.manifest
    manifest_dict = manifest.model_dump()
    _validate_manifest(manifest_dict)

    # If a directory is provided and the entry point exists, run validation
    if body.directory:
        entry_path = Path(body.directory) / manifest.entry_point
        if entry_path.exists():
            result = await _run_plugin_subprocess(
                manifest.entry_point, body.directory
            )
            if not result.success:
                raise HTTPException(
                    status_code=422,
                    detail=f"Plugin validation failed: {result.error}",
                )

    plugin_id = str(uuid.uuid4())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    async with get_db() as db:
        # Check for duplicate name+version for this user
        cursor = await db.execute(
            "SELECT id FROM plugins WHERE name = ? AND version = ? AND user_id = ?",
            (manifest.name, manifest.version, user["id"]),
        )
        if await cursor.fetchone() is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Plugin '{manifest.name}' v{manifest.version} is already installed",
            )

        await db.execute(
            """
            INSERT INTO plugins (
                id, name, version, type, manifest_json, status,
                entry_point, directory, permissions_json,
                description, author, user_id, installed_at
            ) VALUES (?, ?, ?, ?, ?, 'installed', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plugin_id,
                manifest.name,
                manifest.version,
                manifest.type,
                json.dumps(manifest_dict),
                manifest.entry_point,
                body.directory,
                json.dumps(manifest.permissions),
                manifest.description,
                manifest.author,
                user["id"],
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM plugins WHERE id = ?", (plugin_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)


@router.post("/load-directory", response_model=PluginResponse, status_code=201)
async def load_from_directory(
    body: PluginLoadDirRequest,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Load a plugin from a local directory for development.

    Reads plugin.json manifest from the directory, validates it,
    and installs the plugin.
    """
    dir_path = Path(body.directory)
    if not dir_path.is_dir():
        raise HTTPException(status_code=404, detail="Directory not found")

    manifest_path = dir_path / "plugin.json"
    if not manifest_path.exists():
        raise HTTPException(
            status_code=422,
            detail="No plugin.json found in directory",
        )

    try:
        manifest_raw = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Failed to parse plugin.json: {exc}"
        ) from exc

    _validate_manifest(manifest_raw)

    manifest = PluginManifest(**manifest_raw)
    install_req = PluginInstallRequest(
        manifest=manifest, directory=str(dir_path.resolve())
    )
    return await install_plugin(install_req, user)


@router.get("/{plugin_id}", response_model=PluginResponse)
async def get_plugin(
    plugin_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Return a single plugin by ID."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM plugins WHERE id = ? AND user_id = ?",
            (plugin_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return _row_to_dict(row)


@router.delete("/{plugin_id}", status_code=204)
async def uninstall_plugin(
    plugin_id: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> None:
    """Uninstall (delete) a plugin."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT id FROM plugins WHERE id = ? AND user_id = ?",
            (plugin_id, user["id"]),
        )
        if await cursor.fetchone() is None:
            raise HTTPException(status_code=404, detail="Plugin not found")
        await db.execute("DELETE FROM plugins WHERE id = ?", (plugin_id,))
        await db.commit()


@router.put("/{plugin_id}/status")
async def update_plugin_status(
    plugin_id: str,
    status: str,
    user: dict[str, Any] = Depends(get_current_user),  # noqa: B008
) -> dict[str, Any]:
    """Update plugin status (enable/disable)."""
    if status not in VALID_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Must be one of: {', '.join(sorted(VALID_STATUSES))}",
        )
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM plugins WHERE id = ? AND user_id = ?",
            (plugin_id, user["id"]),
        )
        row = await cursor.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Plugin not found")

        await db.execute(
            "UPDATE plugins SET status = ? WHERE id = ?",
            (status, plugin_id),
        )
        await db.commit()

        cursor = await db.execute("SELECT * FROM plugins WHERE id = ?", (plugin_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row)
