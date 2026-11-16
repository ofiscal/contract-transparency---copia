"""Shared fixtures. All tests run on synthetic data inside tmp directories:
no real SECOP data, no network, no files written to the repository."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from contract_transparency.config import Settings, load_config
from contract_transparency.data.cleaning import clean_contracts
from contract_transparency.data.synthetic import generate_contracts

REPO = Path(__file__).resolve().parents[1]
TEST_SALT = b"test-salt-not-secret"


def make_config(tmp: Path, **overrides) -> Settings:
    base = {
        "paths.raw_data": str(tmp / "raw" / "contracts.csv"),
        "paths.processed_dir": str(tmp / "processed"),
        "paths.outputs_dir": str(tmp / "outputs"),
        "paths.registry_dir": str(tmp / "registry"),
        "paths.secrets_dir": str(tmp / ".secrets"),
        # small & fast model for tests
        "features.text_svd_components": 16,
        "features.text_max_features": 2000,
        "features.min_category_frequency": 5,
        "model.num_boost_round": 150,
        "model.early_stopping_rounds": 20,
        # Small trees and 2 threads keep CI fast and stable on shared runners.
        "model.params": {
            "objective": "reg:pseudohubererror",
            "huber_slope": 1.0,
            "eta": 0.1,
            "max_depth": 6,
            "min_child_weight": 5,
            "tree_method": "hist",
            "nthread": 2,
            "seed": 0,
        },
        "drift.min_new_rows_for_retrain": 500,
        "download.backoff_s": 0.01,
    }
    base.update(overrides)
    return load_config(REPO / "configs" / "demo.yaml", overrides=base)


@pytest.fixture(autouse=True)
def _salt_env(monkeypatch):
    monkeypatch.setenv("CT_PII_SALT", TEST_SALT.decode())


@pytest.fixture
def cfg(tmp_path) -> Settings:
    return make_config(tmp_path)


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return generate_contracts(3000, seed=1)


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory) -> Settings:
    return make_config(tmp_path_factory.mktemp("session"))


@pytest.fixture(scope="session")
def clean_df(raw_df, session_cfg) -> pd.DataFrame:
    return clean_contracts(raw_df, session_cfg, TEST_SALT)


@pytest.fixture
def write_raw(cfg):
    """Write a raw DataFrame to the configured raw path and return the path."""

    def _write(df: pd.DataFrame, name: str = "contracts.csv") -> Path:
        path = Path(cfg.paths.raw_data).parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    return _write


def fast_download(cfg: Settings, page_size: int = 700) -> Settings:
    """Small API pages and near-zero retry backoff for tests against the mock API."""
    return cfg.model_copy(
        update={
            "download": cfg.download.model_copy(update={"page_size": page_size, "backoff_s": 0.01})
        }
    )


def write_cli_config(tmp_path: Path) -> Path:
    """A YAML config (extends demo.yaml) with all paths inside tmp_path, for CLI tests."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(
        f"""extends: {REPO / "configs" / "demo.yaml"}
paths:
  raw_data: {tmp_path / "raw.csv"}
  processed_dir: {tmp_path / "processed"}
  outputs_dir: {tmp_path / "outputs"}
  registry_dir: {tmp_path / "registry"}
  secrets_dir: {tmp_path / ".secrets"}
features: {{text_svd_components: 16, min_category_frequency: 5}}
download: {{backoff_s: 0.01}}
model:
  num_boost_round: 100
  early_stopping_rounds: 15
  params: {{objective: "reg:pseudohubererror", eta: 0.1, max_depth: 6, tree_method: hist, nthread: 2}}
"""
    )
    return cfg
