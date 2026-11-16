# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# OPTIONAL container image (the normal way to run the project is uv: `make demo`).
#
#   docker build -t contract-transparency .
#   docker run --rm -p 8501:8501 -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" \
#              -v "$PWD/.secrets:/app/.secrets" contract-transparency            # dashboard
#   docker run --rm -v ... contract-transparency run -c configs/demo.yaml       # any `ct` command
#
# Installs exactly the versions in uv.lock. Data, models and secrets are
# VOLUMES: they are never baked into the image.
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    CT_GIT_COMMIT=${GIT_COMMIT} \
    CT_CONFIG=configs/secop_ii.yaml

RUN pip install "uv>=0.5"

WORKDIR /app

# 1) Dependencies from uv.lock (cached layer; changes only with the lock).
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) The package itself (non-editable; source removed afterwards).
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable && rm -rf src build

COPY configs ./configs

# Non-root user; writable volume mount points.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin ct \
    && mkdir -p data models .secrets \
    && chown -R ct:ct /app
USER ct

VOLUME ["/app/data", "/app/models", "/app/.secrets"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)" || exit 1

ENTRYPOINT ["ct"]
CMD ["app", "--address", "0.0.0.0", "--port", "8501"]
