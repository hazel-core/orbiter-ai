"""Backward-compatible shim — re-exports from orbiter.observability.logging.

All new code should import directly from ``orbiter.observability.logging``.
This module exists solely so that ``from orbiter.log import get_logger``
continues to work.
"""

from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    TextFormatter as _Formatter,
)
from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    configure_logging as configure,
)
from orbiter.observability.logging import (  # pyright: ignore[reportMissingImports]
    get_logger,
)

_PREFIX = "orbiter"

__all__ = ["_PREFIX", "_Formatter", "configure", "get_logger"]
