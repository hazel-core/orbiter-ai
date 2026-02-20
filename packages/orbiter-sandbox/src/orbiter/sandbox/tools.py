"""Built-in sandbox tools: filesystem access and terminal execution."""

from __future__ import annotations

import asyncio
import logging
import platform
import sys
from pathlib import Path
from typing import Any

from orbiter.tool import Tool, ToolError  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FilesystemTool
# ---------------------------------------------------------------------------

_FS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["read", "write", "list"],
            "description": "Filesystem action to perform.",
        },
        "path": {
            "type": "string",
            "description": "Absolute or relative file/directory path.",
        },
        "content": {
            "type": "string",
            "description": "Content to write (required for 'write' action).",
        },
    },
    "required": ["action", "path"],
}


class FilesystemTool(Tool):
    """Sandboxed filesystem tool with allowed-directory restrictions.

    Only paths that resolve within one of ``allowed_directories`` are
    permitted.  This prevents agents from reading or writing files outside
    the designated workspace.
    """

    name = "filesystem"
    description = "Read, write, or list files within the sandbox workspace."
    parameters = _FS_SCHEMA

    def __init__(self, allowed_directories: list[str] | None = None) -> None:
        self._allowed: list[Path] = [Path(d).resolve() for d in (allowed_directories or [])]

    # -- path validation ----------------------------------------------------

    def _validate_path(self, raw: str) -> Path:
        """Resolve *raw* and ensure it falls inside an allowed directory."""
        resolved = Path(raw).resolve()
        if not self._allowed:
            return resolved
        for allowed in self._allowed:
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        msg = f"Path {str(resolved)!r} is outside allowed directories"
        raise ToolError(msg)

    # -- execute ------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        action: str = kwargs.get("action", "")
        raw_path: str = kwargs.get("path", "")
        content: str | None = kwargs.get("content")

        if action not in {"read", "write", "list"}:
            raise ToolError(f"Unknown filesystem action: {action!r}")

        path = self._validate_path(raw_path)
        logger.debug("FilesystemTool: %s %s", action, path)

        if action == "read":
            return await self._read(path)
        if action == "write":
            if content is None:
                raise ToolError("'content' is required for write action")
            return await self._write(path, content)
        # action == "list"
        return await self._list(path)

    async def _read(self, path: Path) -> str:
        try:
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError as exc:
            raise ToolError(f"File not found: {path}") from exc
        except Exception as exc:
            raise ToolError(f"Read failed: {exc}") from exc

    async def _write(self, path: Path, content: str) -> str:
        try:
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {path}"
        except Exception as exc:
            raise ToolError(f"Write failed: {exc}") from exc

    async def _list(self, path: Path) -> dict[str, Any]:
        try:
            entries = await asyncio.to_thread(lambda: list(path.iterdir()))
            items = [
                {"name": e.name, "type": "dir" if e.is_dir() else "file"}
                for e in sorted(entries, key=lambda e: e.name)
            ]
            return {"directory": str(path), "entries": items}
        except FileNotFoundError as exc:
            raise ToolError(f"Directory not found: {path}") from exc
        except NotADirectoryError as exc:
            raise ToolError(f"Not a directory: {path}") from exc
        except Exception as exc:
            raise ToolError(f"List failed: {exc}") from exc


# ---------------------------------------------------------------------------
# TerminalTool
# ---------------------------------------------------------------------------

_DANGEROUS_COMMANDS: frozenset[str] = frozenset(
    {
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "kill",
        "killall",
        "pkill",
        "format",
        # Windows equivalents
        "del",
        "erase",
        "rd",
    }
)

_TERMINAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute.",
        },
    },
    "required": ["command"],
}


class TerminalTool(Tool):
    """Sandboxed terminal tool with command filtering and timeout.

    Dangerous commands (``rm``, ``shutdown``, etc.) are blocked by default.
    Custom blacklists can be provided.  All commands run with a configurable
    timeout to prevent runaway processes.
    """

    name = "terminal"
    description = "Execute a shell command in the sandbox."
    parameters = _TERMINAL_SCHEMA

    def __init__(
        self,
        *,
        blacklist: frozenset[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._blacklist: frozenset[str] = (
            blacklist if blacklist is not None else _DANGEROUS_COMMANDS
        )
        self._timeout = timeout

    @property
    def platform(self) -> str:
        """Return the current platform identifier."""
        return sys.platform

    def _check_command(self, command: str) -> None:
        """Raise ``ToolError`` if the command's base executable is blacklisted."""
        parts = command.strip().split()
        if not parts:
            raise ToolError("Empty command")
        base = parts[0].rsplit("/", maxsplit=1)[-1]  # strip path prefix
        base = base.rsplit("\\", maxsplit=1)[-1]
        if base.lower() in {b.lower() for b in self._blacklist}:
            logger.warning("TerminalTool: blocked command %r (sandbox policy)", base)
            raise ToolError(f"Command {base!r} is blocked by sandbox policy")

    async def execute(self, **kwargs: Any) -> str | dict[str, Any]:
        command: str = kwargs.get("command", "")
        if not command.strip():
            raise ToolError("Empty command")

        self._check_command(command)
        logger.debug("TerminalTool: executing %r (timeout=%.1fs)", command, self._timeout)

        if platform.system() == "Windows":
            prog: str | None = None  # use default shell
        else:
            prog = "/bin/sh"

        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                executable=prog,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
        except TimeoutError as exc:
            if proc is not None:
                proc.kill()
                await proc.wait()
            logger.warning("TerminalTool: command timed out after %.1fs: %r", self._timeout, command)
            raise ToolError(f"Command timed out after {self._timeout}s") from exc
        except asyncio.CancelledError:
            if proc is not None:
                proc.kill()
                await proc.wait()
            logger.debug("TerminalTool: cancelled, killed subprocess for %r", command)
            raise
        except Exception as exc:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            logger.error("TerminalTool: execution failed for %r: %s", command, exc)
            raise ToolError(f"Command execution failed: {exc}") from exc

        logger.debug("TerminalTool: %r exited with code %s", command, proc.returncode)
        return {
            "exit_code": proc.returncode,
            "stdout": stdout.decode(errors="replace") if stdout else "",
            "stderr": stderr.decode(errors="replace") if stderr else "",
            "platform": self.platform,
        }
