# Multi-stage build: dependencies are resolved in a layer that only changes
# when the lockfile changes, so editing source code does not trigger a full
# reinstall. This is the difference between a 4-second and a 4-minute rebuild.

FROM python:3.12-slim AS builder

# Copy uv from its official image instead of pip-installing it — smaller and
# pinned to an exact version.
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Dependency layer: only the manifest and lockfile, nothing else.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Application layer.
COPY src/ ./src/
RUN uv sync --frozen --no-dev


FROM python:3.12-slim AS runtime

# Run as a non-root user. A container process that does not need root should
# not have it.
RUN useradd --create-home --uid 1000 app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY --from=builder --chown=app:app /app/src ./src

USER app

# The image ships the code and its dependencies. It deliberately does **not**
# ship the data or the trained model:
#
#   * `data/` is gigabytes of parquet that changes daily. Baking it in would
#     produce an image that is stale the moment it is built, and would rebuild
#     from scratch every time the ingestion job runs.
#   * `models/` changes on its own schedule, and an image that contains one
#     model can only ever serve that model.
#
# Both are mounted at run time instead, which is also what makes "roll back to
# yesterday's model" a flag rather than a rebuild:
#
#   docker run --rm -p 8000:8000 \
#     -v "$(pwd)/data:/app/data:ro" \
#     -v "$(pwd)/models:/app/models:ro" \
#     tr-power-forecast
#
# Read-only mounts, because a forecasting service has no business writing to
# either one.
EXPOSE 8000

# 0.0.0.0, not 127.0.0.1. Inside a container, localhost means the container
# itself, so a server bound there is unreachable from the host no matter what
# `-p` says — one of the two or three things that trips up every first
# containerised web service.
CMD ["uvicorn", "powerforecast.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
