# Everything runs through uv (https://docs.astral.sh/uv/): it creates .venv,
# downloads a suitable Python if needed and never touches the system Python
# (no "externally-managed-environment" errors). Docker is optional.
#
# Windows without make: run the same commands shown in each recipe, e.g.
#   uv sync ; uv run ct demo-data --with-records ; uv run ct run -c configs/demo.yaml
UV ?= uv
RUN := $(UV) run
CONFIG ?= configs/secop_ii.yaml
DEMO := configs/demo.yaml
# Python used by `make test-latest` (newest releases of everything); uv downloads it if missing.
LATEST_PYTHON ?= 3.14

.DEFAULT_GOAL := help
.PHONY: help check-uv install install-mlflow install-nlp demo app app-real download train score \
        update live-check doctor test test-fast lint format upgrade upgrade-package rollback-deps \
        test-lowest test-latest portability-check lock-check clean-demo docker-build docker-demo docker-app

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
	  echo "uv is not installed. Install it (no admin rights needed):"; \
	  echo "  Linux/macOS:  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	  echo "  Windows:      powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""; \
	  echo "  or:           pipx install uv   |   brew install uv   |   winget install astral-sh.uv"; \
	  exit 1; }

# --- setup ---------------------------------------------------------------------
install: check-uv  ## Create .venv with the tested versions from uv.lock (+ dev tools)
	$(UV) sync
	@mkdir -p data models .secrets

install-mlflow: check-uv  ## Also install MLflow tracking
	$(UV) sync --extra mlflow

install-nlp: check-uv  ## Also install transformer text models (large download)
	$(UV) sync --extra nlp

# --- use -----------------------------------------------------------------------
demo: install  ## Synthetic data -> train -> promote -> score (no internet needed)
	$(RUN) ct demo-data --with-records
	$(RUN) ct run -c $(DEMO)

app: check-uv  ## Dashboard with the demo model: http://localhost:8501
	$(RUN) ct app -c $(DEMO)

app-real: check-uv  ## Dashboard with the real-data model (CONFIG)
	$(RUN) ct app -c $(CONFIG)

download: check-uv  ## Download SECOP II: make download SINCE=2024-01-01 [UNTIL=2025-01-01] [LIMIT=10000]
	$(RUN) ct download -c $(CONFIG) $(if $(SINCE),--since $(SINCE)) $(if $(UNTIL),--until $(UNTIL)) $(if $(LIMIT),--limit $(LIMIT))

train: check-uv  ## Prepare + train on the real data configured in CONFIG
	$(RUN) ct prepare -c $(CONFIG)
	$(RUN) ct train -c $(CONFIG)

score: check-uv  ## Score the processed snapshot with the champion model
	$(RUN) ct score -c $(CONFIG)

update: check-uv  ## Incremental update: make update NEW=data/raw/new_extract.csv
	$(RUN) ct update $(NEW) -c $(CONFIG)

live-check: check-uv  ## Verify everything against the live datos.gov.co API (needs internet)
	$(RUN) ct live-check -c $(CONFIG)

doctor: check-uv  ## Check environment, versions and the champion model
	$(RUN) ct doctor -c $(CONFIG)

# --- quality -------------------------------------------------------------------
test: check-uv  ## Full test-suite with coverage (~2 min)
	$(RUN) pytest --cov --cov-report=term-missing

test-fast: check-uv  ## Unit tests only (skip end-to-end)
	$(RUN) pytest -m "not slow"

lint: check-uv  ## Static checks
	$(RUN) ruff check . && $(RUN) ruff format --check src tests

format: check-uv  ## Auto-format
	$(RUN) ruff format src tests && $(RUN) ruff check --fix .

# --- dependency upgrades (see docs/DEPENDENCIES.md) -----------------------------
upgrade: check-uv  ## Upgrade ALL packages to their newest versions, then test + doctor
	$(UV) lock --upgrade
	$(UV) sync
	$(RUN) pytest -m "not slow"
	$(RUN) ct doctor -c $(DEMO)
	@echo "If everything passed: commit uv.lock. Otherwise: make rollback-deps"

upgrade-package: check-uv  ## Upgrade ONE package: make upgrade-package PKG=pandas
	@test -n "$(PKG)" || { echo "usage: make upgrade-package PKG=<name>"; exit 2; }
	$(UV) lock --upgrade-package $(PKG)
	$(UV) sync
	$(RUN) pytest -m "not slow"
	$(RUN) ct doctor -c $(DEMO)

rollback-deps: check-uv  ## Undo an upgrade: restore uv.lock from git and re-sync
	git checkout -- uv.lock
	$(UV) sync

test-lowest: check-uv  ## Test with the OLDEST versions pyproject.toml allows (Python 3.10)
	$(UV) venv --clear -p 3.10 .venv-lowest
	$(UV) pip install -p .venv-lowest --resolution lowest-direct -e . --group dev
	$(UV) run --no-project -p .venv-lowest pytest -m "not slow"

test-latest: check-uv  ## Test with the NEWEST releases of everything (early warning)
	$(UV) venv --clear -p $(LATEST_PYTHON) .venv-latest
	$(UV) pip install -p .venv-latest --resolution highest -e . --group dev
	$(UV) run --no-project -p .venv-latest pytest -m "not slow" -W error::FutureWarning

portability-check: test-lowest test-latest  ## Train with oldest libs, score with newest: must be identical
	scripts/check_model_portability.sh .venv-lowest .venv-latest

lock-check: check-uv  ## Fail if uv.lock is out of date with pyproject.toml
	$(UV) lock --check

clean-demo:  ## Remove demo data, registry and outputs
	rm -rf data/raw/demo_contracts.csv data/raw/demo_boletin_responsables_fiscales.xlsx \
	       data/processed/demo data/outputs/demo models/registry_demo

# --- optional: Docker ----------------------------------------------------------
define need_docker
	@command -v docker >/dev/null 2>&1 || { echo "Docker is not installed (it is optional): use 'make demo' and 'make app' instead."; exit 1; }
endef

docker-build:  ## (optional) Build the Docker image
	$(need_docker)
	mkdir -p data models .secrets
	test -f .env || cp .env.example .env
	GIT_COMMIT=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) docker compose build

docker-demo: docker-build  ## (optional) Demo inside the container
	docker compose run --rm ct demo-data --with-records
	docker compose run --rm ct run -c configs/demo.yaml

docker-app:  ## (optional) Dashboard in the container: http://localhost:8501
	$(need_docker)
	CT_CONFIG=configs/demo.yaml docker compose up app
