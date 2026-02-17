"""Background cleanup — removes expired sessions, stale tokens, and orphaned uploads."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from orbiter_web.config import settings
from orbiter_web.database import get_db

_log = logging.getLogger(__name__)

_cleanup_task: asyncio.Task[Any] | None = None


async def _run_cleanup() -> None:
    """Execute one cleanup pass and log results."""
    expired_sessions = 0
    stale_tokens = 0
    orphaned_files = 0

    async with get_db() as db:
        # 1. Delete expired sessions (expires_at < now)
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        cursor = await db.execute(
            "DELETE FROM sessions WHERE expires_at < ?", (now,)
        )
        expired_sessions = cursor.rowcount

        # 2. Delete used or expired password reset tokens
        cursor = await db.execute(
            "DELETE FROM password_resets WHERE used = 1 OR expires_at < ?", (now,)
        )
        stale_tokens = cursor.rowcount

        await db.commit()

    # 3. Delete orphaned upload temp files older than 24 hours
    upload_dir = Path(settings.upload_dir)
    if upload_dir.is_dir():
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        # Check which files are still referenced in documents table
        referenced: set[str] = set()
        async with get_db() as db:
            cursor = await db.execute("SELECT file_path FROM documents")
            for row in await cursor.fetchall():
                referenced.add(row["file_path"])

        for file_path in upload_dir.iterdir():
            if not file_path.is_file():
                continue
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
            if mtime < cutoff and str(file_path) not in referenced:
                try:
                    file_path.unlink()
                    orphaned_files += 1
                except OSError:
                    _log.warning("Failed to remove orphaned file: %s", file_path)

    _log.info(
        "Cleanup complete: %d expired sessions, %d stale tokens, %d orphaned files removed",
        expired_sessions,
        stale_tokens,
        orphaned_files,
    )


async def _cleanup_loop() -> None:
    """Run cleanup on startup, then repeat at the configured interval."""
    interval_seconds = settings.cleanup_interval_hours * 3600
    while True:
        try:
            await _run_cleanup()
        except Exception:
            _log.exception("Error during cleanup")
        await asyncio.sleep(interval_seconds)


async def start_cleanup() -> None:
    """Start the background cleanup task."""
    global _cleanup_task
    if _cleanup_task is not None:
        return
    _log.info(
        "Starting cleanup task (every %dh)", settings.cleanup_interval_hours
    )
    _cleanup_task = asyncio.create_task(_cleanup_loop())


async def stop_cleanup() -> None:
    """Stop the background cleanup task."""
    global _cleanup_task
    if _cleanup_task is None:
        return
    _log.info("Stopping cleanup task")
    _cleanup_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await _cleanup_task
    _cleanup_task = None
