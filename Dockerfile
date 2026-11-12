# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 (builder): resolve api/poetry.lock into an isolated venv.
#
# Poetry lives in ITS OWN venv (/opt/poetry) and the app's dependencies in
# another (/opt/venv). Sharing one site-packages lets the app's lock overwrite
# libraries Poetry itself needs (setuptools, packaging), which breaks the build.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Same version that generated api/poetry.lock (see its header), so Docker
# resolves exactly the environment local runs and CI use.
ENV POETRY_VERSION=2.4.1 \
    # Poetry's 15 s default times out on large wheels (torch, mlflow) on slow
    # networks.
    POETRY_REQUESTS_TIMEOUT=600 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install "poetry==${POETRY_VERSION}" \
    && python -m venv /opt/venv

# With VIRTUAL_ENV set, Poetry installs into that venv instead of creating one.
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY api/pyproject.toml api/poetry.lock ./
RUN /opt/poetry/bin/poetry install --only main --no-interaction --no-ansi

# ---------------------------------------------------------------------------
# Stage 2 (runtime): only the resolved venv and the code. No Poetry, compilers
# or curl: a smaller image and a smaller CVE surface.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    # api/main.py imports its sibling modules as top-level (`from config_loader
    # import load_config`), as it does locally and in the tests.
    PYTHONPATH=/app/api

# DS-0002: never run as root. UID 10001 matches the securityContext in
# kubernetes/base/api.yaml.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser api/ ./api/
# Required at runtime: api/config_loader.py reads <parent-of-api>/config/config.yaml.
COPY --chown=appuser:appuser config/ ./config/

USER appuser
EXPOSE 8000

# DS-0026: image-level healthcheck, through Python because the runtime image
# has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
