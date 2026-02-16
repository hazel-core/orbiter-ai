"""CLI commands for orbiter-web administration."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

import bcrypt

from orbiter_web.database import get_db, run_migrations


def _hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


async def _create_user(email: str, password: str, *, admin: bool = False) -> None:
    """Create a user record in the database."""
    # Ensure migrations are up to date.
    await run_migrations()

    user_id = str(uuid.uuid4())
    password_hash = _hash_password(password)

    async with get_db() as db:
        # Check if email already exists.
        cursor = await db.execute("SELECT id FROM users WHERE email = ?", (email,))
        if await cursor.fetchone():
            print(f"Error: a user with email '{email}' already exists.", file=sys.stderr)
            sys.exit(1)

        await db.execute(
            "INSERT INTO users (id, email, password_hash, is_admin) VALUES (?, ?, ?, ?)",
            (user_id, email, password_hash, int(admin)),
        )
        await db.commit()

    print(f"User created: {email} (id: {user_id})")


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(prog="orbiter-web", description="Orbiter Web CLI")
    subparsers = parser.add_subparsers(dest="command")

    create_user_parser = subparsers.add_parser("create-user", help="Create a new user account")
    create_user_parser.add_argument("--email", required=True, help="User email address")
    create_user_parser.add_argument("--password", required=True, help="User password")
    create_user_parser.add_argument(
        "--admin", action="store_true", default=False, help="Grant admin privileges"
    )

    args = parser.parse_args()

    if args.command == "create-user":
        asyncio.run(_create_user(args.email, args.password, admin=args.admin))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
