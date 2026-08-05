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

CMD ["python", "-c", "import powerforecast; print(powerforecast.__version__)"]
