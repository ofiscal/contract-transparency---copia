"""A minimal, file-based model registry (no server needed).

Layout::

    models/registry/<model_name>/
        versions/v0001/   model files + metadata.json
        versions/v0002/
        champion.json     -> {"version": "v0002", ...}  (the model used to score)
        history.jsonl     append-only audit log (registered / promoted / rejected)

* Every artifact is hashed (SHA-256) at registration and verified before
  loading: joblib files are pickles, so a tampered file could execute code.
* Versions are immutable; "rollback" = promote an older version.
* The same metadata is mirrored to MLflow when tracking is enabled, so the
  registry can later be swapped for the MLflow Model Registry without
  changing the pipeline API.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from contract_transparency.environment import (
    ModelCompatibilityError,
    compatibility_hint,
    model_compatibility,
)
from contract_transparency.models.price_model import PriceModel
from contract_transparency.models.segmented import load_model

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.json"
CHAMPION_FILE = "champion.json"
HISTORY_FILE = "history.jsonl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover - containers without a passwd entry
        return "unknown"


class IntegrityError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, root: str | Path, model_name: str):
        self.root = Path(root) / model_name
        self.versions_dir = self.root / "versions"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- queries
    def versions(self) -> list[str]:
        return sorted(p.name for p in self.versions_dir.iterdir() if (p / METADATA_FILE).exists())

    def champion(self) -> str | None:
        path = self.root / CHAMPION_FILE
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["version"]

    def metadata(self, version: str) -> dict[str, Any]:
        version = self._resolve(version)
        return json.loads((self.versions_dir / version / METADATA_FILE).read_text(encoding="utf-8"))

    def summary(self) -> pd.DataFrame:
        champ = self.champion()
        rows = []
        for v in self.versions():
            m = self.metadata(v)
            test = m.get("metrics", {}).get("test", {})
            rows.append(
                {
                    "version": v,
                    "champion": v == champ,
                    "created_at": m.get("created_at"),
                    "model_type": m.get("model_type"),
                    "n_train": m.get("data", {}).get("n_train"),
                    "mae_log": test.get("mae_log"),
                    "mdape": test.get("mdape"),
                    "r2_log": test.get("r2_log"),
                    "promotion": m.get("promotion", {}).get("decision"),
                }
            )
        return pd.DataFrame(rows)

    def history(self) -> list[dict]:
        path = self.root / HISTORY_FILE
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    # ------------------------------------------------------------ mutations
    def _next_version(self) -> str:
        existing = [int(v[1:]) for v in self.versions() if v[1:].isdigit()]
        return f"v{(max(existing) + 1 if existing else 1):04d}"

    def register(self, model: PriceModel, metadata: dict[str, Any]) -> str:
        """Save the model as a new immutable version (atomic directory rename)."""
        version = self._next_version()
        tmp = Path(tempfile.mkdtemp(prefix=".tmp_", dir=self.versions_dir))
        try:
            files = model.save(tmp)
            meta = dict(metadata)
            meta.update(
                {
                    "version": version,
                    "created_at": _now(),
                    "created_by": _user(),
                    "artifacts": {f.relative_to(tmp).as_posix(): _sha256(f) for f in files},
                }
            )
            (tmp / METADATA_FILE).write_text(
                json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, self.versions_dir / version)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._log("registered", version)
        return version

    def update_metadata(self, version: str, **fields: Any) -> None:
        version = self._resolve(version)
        path = self.versions_dir / version / METADATA_FILE
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta.update(fields)
        path.write_text(
            json.dumps(meta, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
        )

    def promote(self, version: str, reason: str = "manual") -> None:
        version = self._resolve(version)
        previous = self.champion()
        payload = {
            "version": version,
            "promoted_at": _now(),
            "promoted_by": _user(),
            "reason": reason,
            "previous": previous,
        }
        path = self.root / CHAMPION_FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        self._log("promoted", version, reason=reason, previous=previous)

    def log_rejection(self, version: str, reasons: list[str]) -> None:
        self._log("rejected", version, reasons=reasons)

    # --------------------------------------------------------------- loading
    def load(
        self, version: str = "champion", verify: bool = True
    ) -> tuple[PriceModel, dict[str, Any]]:
        version = self._resolve(version)
        directory = self.versions_dir / version
        meta = self.metadata(version)
        if verify:
            for name, expected in meta.get("artifacts", {}).items():
                actual = _sha256(directory / name)
                if actual != expected:
                    raise IntegrityError(
                        f"Checksum mismatch for {version}/{name}: refusing to load"
                    )
        problems = model_compatibility(meta.get("libraries"), self.is_portable(version))
        if problems:
            logger.warning(compatibility_hint(version, problems))
        try:
            with warnings.catch_warnings():
                # sklearn warns (InconsistentVersionWarning) when unpickling across
                # versions; the mismatch is already reported above.
                warnings.simplefilter("ignore", UserWarning)
                model = load_model(directory)
        except Exception as exc:  # any unpickling/format error -> actionable message
            raise ModelCompatibilityError(
                f"Could not load model {version}: {exc!s}. "
                + compatibility_hint(version, problems or ["library versions changed?"])
            ) from exc
        return model, meta

    def is_portable(self, version: str) -> bool:
        """v1.1+ models store no pickles and survive library upgrades."""
        directory = self.versions_dir / self._resolve(version)
        # single model: preprocessor.json at the top; segmented: one per segment
        return any(directory.rglob("preprocessor.json"))

    # -------------------------------------------------------------- helpers
    def _resolve(self, version: str) -> str:
        if version in {"champion", "production"}:
            champ = self.champion()
            if champ is None:
                raise LookupError(
                    "No champion model yet: run `ct train` (or `ct promote <version>`)"
                )
            return champ
        if version == "latest":
            versions = self.versions()
            if not versions:
                raise LookupError("Registry is empty: run `ct train` first")
            return versions[-1]
        if not (self.versions_dir / version / METADATA_FILE).exists():
            raise LookupError(f"Unknown model version '{version}'. Available: {self.versions()}")
        return version

    def _log(self, event: str, version: str, **extra: Any) -> None:
        entry = {"ts": _now(), "event": event, "version": version, "user": _user(), **extra}
        with open(self.root / HISTORY_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
