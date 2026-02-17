# Dockerfile for Orbiter distributed workers.
#
# Build context should be the repository root:
#   docker build -f examples/distributed/Dockerfile -t orbiter-worker .

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy the full monorepo so workspace packages resolve correctly
COPY . .

# Install all workspace packages including orbiter-cli (provides the
# ``orbiter`` command) and orbiter-distributed (worker runtime).
RUN uv sync --frozen

# Default command — override via docker-compose command or docker run args.
CMD ["uv", "run", "orbiter", "start", "worker"]
