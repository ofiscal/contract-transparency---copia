"""Environment health and version compatibility (``ct doctor``).

Dependencies are declared with minimum versions only (see
docs/DEPENDENCIES.md), so packages can be upgraded freely. The one thing an
upgrade can break silently is a **saved model**: the sklearn preprocessor is
stored with joblib (pickle), which is only guaranteed to load with the
scikit-learn version that wrote it. This module

* records the library versions used to train each model (stored in the
  registry metadata),
* compares them with the running environment when a model is loaded, and
* runs ``ct doctor``: a PASS/WARN/FAIL report to run after every upgrade.

The fix for an incompatible model is always the same: ``ct train`` (re-trains
on the current snapshot with the current libraries; the old version stays in
the registry for reference).
"""

from __future__ import annotations

import importlib.metadata as md
import logging
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DIST_NAME = "contract-transparency"
# Libraries whose version matters for loading a saved model.
MODEL_LIBRARIES = ("scikit-learn", "xgboost", "numpy", "pandas", "joblib")
OPTIONAL_LIBRARIES = ("mlflow", "sentence-transformers", "transformers", "torch")


class ModelCompatibilityError(RuntimeError):
    """A saved model cannot be used with the installed libraries."""


# --------------------------------------------------------------------------- versions
def version_tuple(version: str | None) -> tuple[int, ...]:
    """'1.9.1' -> (1, 9, 1); '2.0.0rc1' -> (2, 0, 0). Unknown -> ()."""
    if not version:
        return ()
    nums = []
    for part in version.split(".")[:3]:
        m = re.match(r"\d+", part)
        if not m:
            break
        nums.append(int(m.group()))
    return tuple(nums)


def installed_version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def library_versions() -> dict[str, str | None]:
    """Versions recorded with every trained model."""
    out = {"python": platform.python_version()}
    out.update({lib: installed_version(lib) for lib in (DIST_NAME, *MODEL_LIBRARIES)})
    return out


def declared_minimums() -> dict[str, str]:
    """Core dependency minimums, read from the installed package metadata
    (so they never drift from pyproject.toml)."""
    mins: dict[str, str] = {}
    for req in md.requires(DIST_NAME) or []:
        if "extra ==" in req:
            continue
        m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*(?:\[[^\]]*\])?\s*>=\s*([^,;\s]+)", req)
        if m:
            mins[m.group(1).lower()] = m.group(2)
    return mins


def model_compatibility(trained_with: dict[str, Any] | None, portable: bool = True) -> list[str]:
    """Human-readable problems between a model's training libraries and now.

    Portable models (v1.1+, no pickles) only need xgboost to be at least as
    new as the one that wrote the booster. Legacy v1.0 models (pickled sklearn
    preprocessor) additionally need the same scikit-learn minor version and
    the same numpy major version.
    """
    if not trained_with:
        return []
    trained_with = dict(trained_with)
    if "sklearn" in trained_with:  # key used by registries created before v1.1
        trained_with.setdefault("scikit-learn", trained_with["sklearn"])
    current = library_versions()
    problems = []

    def pair(lib: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return version_tuple(trained_with.get(lib)), version_tuple(current.get(lib))

    if not portable:
        old, new = pair("scikit-learn")
        if old and new and old[:2] != new[:2]:
            problems.append(
                f"scikit-learn {trained_with['scikit-learn']} -> {current['scikit-learn']}"
            )
        old, new = pair("numpy")
        if old and new and old[:1] != new[:1]:
            problems.append(f"numpy {trained_with['numpy']} -> {current['numpy']}")
    old, new = pair("xgboost")
    if old and new and new < old:
        problems.append(f"xgboost {trained_with['xgboost']} -> {current['xgboost']} (older)")
    return problems


def compatibility_hint(version: str, problems: list[str]) -> str:
    return (
        f"Model {version} was trained with different library versions ({'; '.join(problems)}). "
        "Re-train it with the current environment: `ct train` (or `uv run ct train`). "
        "Alternatively roll the environment back: `git checkout uv.lock && uv sync`."
    )


# --------------------------------------------------------------------------- doctor
@dataclass
class Check:
    name: str
    status: str  # PASS | WARN | FAIL | INFO
    detail: str = ""
    hint: str = ""


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def ok(self) -> bool:
        return all(c.status != "FAIL" for c in self.checks)


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            return True
    except OSError:
        return False


def doctor(cfg, network: bool = False) -> DoctorReport:
    from contract_transparency.mlops import pipeline

    rep = DoctorReport()

    # Python & packages ------------------------------------------------------
    py = sys.version_info
    rep.add("python", "PASS" if py >= (3, 10) else "FAIL", platform.python_version())
    for lib, minimum in sorted(declared_minimums().items()):
        have = installed_version(lib)
        if have is None:
            rep.add(f"package {lib}", "FAIL", "not installed", "Run `uv sync`.")
        elif version_tuple(have) < version_tuple(minimum):
            rep.add(f"package {lib}", "FAIL", f"{have} < required {minimum}", "Run `uv sync`.")
        else:
            rep.add(f"package {lib}", "PASS", f"{have} (>= {minimum})")
    for lib in OPTIONAL_LIBRARIES:
        have = installed_version(lib)
        rep.add(f"optional {lib}", "INFO", have or "not installed")

    # Secrets & folders -------------------------------------------------------
    salt_env = cfg.privacy.salt_env_var
    salt_file = Path(cfg.paths.secrets_dir) / "pii_salt"
    if os.environ.get(salt_env):
        rep.add("pseudonymisation key", "PASS", f"from ${salt_env}")
    elif salt_file.exists():
        rep.add("pseudonymisation key", "PASS", f"from {salt_file}")
    else:
        rep.add(
            "pseudonymisation key",
            "WARN",
            "not set yet",
            f"Set {salt_env} (see .env.example); otherwise one is generated on first use.",
        )
    for name in ("processed_dir", "outputs_dir", "registry_dir"):
        p = Path(getattr(cfg.paths, name))
        rep.add(f"writable {name}", "PASS" if _writable(p) else "FAIL", str(p))

    # Data & model --------------------------------------------------------------
    snap = pipeline.processed_path(cfg)
    rep.add(
        "processed snapshot",
        "PASS" if snap.exists() else "INFO",
        str(snap) if snap.exists() else "none yet (run `ct prepare`)",
    )
    registry = pipeline.get_registry(cfg)
    champion = registry.champion()
    if not champion:
        rep.add("champion model", "INFO", "none yet (run `ct train`)")
    else:
        meta = registry.metadata(champion)
        trained_with = meta.get("libraries", {})
        portable = registry.is_portable(champion)
        problems = model_compatibility(trained_with, portable)
        libs = ", ".join(f"{k} {v}" for k, v in trained_with.items() if v)
        fmt = (
            "portable format: safe across library upgrades" if portable else "legacy pickled format"
        )
        warn = bool(problems) or not portable
        rep.add(
            "champion trained with",
            "WARN" if warn else "PASS",
            f"{libs} ({fmt})",
            compatibility_hint(champion, problems or ["legacy v1.0 format"]) if warn else "",
        )
        try:
            model, _ = registry.load("champion")
            if snap.exists():
                import pandas as pd

                sample = pd.read_parquet(snap).head(5)
                pred = model.predict(sample)
                ok = len(pred) == len(sample)
                rep.add("champion loads & predicts", "PASS" if ok else "FAIL", f"{champion}")
            else:
                rep.add(
                    "champion loads & predicts",
                    "PASS",
                    f"{champion} (loaded; no snapshot to predict)",
                )
        except Exception as exc:  # report any failure as a FAIL check
            rep.add(
                "champion loads & predicts",
                "FAIL",
                f"{champion}: {exc!s}"[:300],
                compatibility_hint(champion, problems or ["unknown"]),
            )

    # Network ---------------------------------------------------------------------
    if network:
        from contract_transparency.data.download import DownloadError, client_from_config

        client = client_from_config(cfg)
        client.max_retries, client.timeout = 0, 10
        try:
            meta = client.metadata()
            rep.add(
                "datos.gov.co API",
                "PASS",
                f"{meta.get('name')} ({len(meta.get('columns', []))} columns)",
            )
        except (DownloadError, OSError) as exc:
            rep.add(
                "datos.gov.co API",
                "WARN",
                str(exc)[:200],
                "No internet access? `ct download` needs it.",
            )
    return rep
