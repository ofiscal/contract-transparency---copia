#!/usr/bin/env bash
# One-time setup for Linux/macOS without `make`: installs uv if needed, then
# creates .venv with the tested versions from uv.lock.
#
# uv never touches the system Python (no "externally-managed-environment"
# errors) and downloads a suitable Python if none is installed.
#
#   scripts/setup.sh            # core + dashboard + dev tools
#   scripts/setup.sh mlflow     # + MLflow tracking     (also: nlp)
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing it for the current user (https://docs.astral.sh/uv/)"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v pipx >/dev/null 2>&1; then
    pipx install uv
  else
    echo "Install uv manually: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
  fi
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

EXTRA_ARGS=()
for extra in "$@"; do EXTRA_ARGS+=(--extra "$extra"); done
uv sync ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}  # (safe with an empty array on macOS bash 3.2)
mkdir -p data models .secrets

cat <<'MSG'

Done. Run commands through uv (no activation needed):
    uv run ct demo-data --with-records
    uv run ct run -c configs/demo.yaml
    uv run ct app -c configs/demo.yaml        # http://localhost:8501
or activate the environment:  source .venv/bin/activate
MSG
