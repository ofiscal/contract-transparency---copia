"""Optional MLflow experiment tracking.

If ``tracking.enabled`` is false or MLflow is not installed, every call is a
no-op, so the pipeline never depends on a tracking server being available.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from contract_transparency.config import TrackingConfig

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, cfg: TrackingConfig):
        self.cfg = cfg
        self._mlflow = None
        if cfg.enabled:
            try:
                import mlflow

                if cfg.tracking_uri:
                    mlflow.set_tracking_uri(cfg.tracking_uri)
                if cfg.artifact_location and mlflow.get_experiment_by_name(cfg.experiment) is None:
                    # Keep artifacts in a writable, git-ignored place (not ./mlruns).
                    location = Path(cfg.artifact_location).resolve()
                    location.mkdir(parents=True, exist_ok=True)
                    mlflow.create_experiment(cfg.experiment, artifact_location=location.as_uri())
                mlflow.set_experiment(cfg.experiment)
                self._mlflow = mlflow
            except ImportError:
                logger.warning(
                    "tracking.enabled=true but MLflow is not installed (pip install -e .[mlflow])"
                )

    @property
    def active(self) -> bool:
        return self._mlflow is not None

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None):
        if not self.active:
            yield self
            return
        with self._mlflow.start_run(run_name=name, tags=tags):
            yield self

    def log_params(self, params: dict[str, Any]) -> None:
        if self.active:
            flat = {k: (str(v)[:500]) for k, v in _flatten(params).items()}
            self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        if self.active:
            clean = {
                f"{prefix}{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))
            }
            self._mlflow.log_metrics(clean)

    def log_artifacts(self, directory: Path) -> None:
        if self.active:
            self._mlflow.log_artifacts(str(directory))

    def set_tags(self, tags: dict[str, Any]) -> None:
        if self.active:
            self._mlflow.set_tags({k: str(v) for k, v in tags.items()})


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out
