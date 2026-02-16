"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings, loaded from environment variables."""

    database_url: str = os.getenv("ORBITER_DATABASE_URL", "sqlite+aiosqlite:///orbiter.db")
    secret_key: str = os.getenv("ORBITER_SECRET_KEY", "change-me-in-production")
    debug: bool = os.getenv("ORBITER_DEBUG", "false").lower() in ("true", "1", "yes")
    session_expiry_hours: int = int(os.getenv("ORBITER_SESSION_EXPIRY_HOURS", "72"))


settings = Settings()
