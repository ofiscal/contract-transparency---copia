"""Data-drift monitoring with the Population Stability Index (PSI).

At training time a *profile* of the training data is stored in the model
metadata: only aggregated proportions (quantile bins, category shares), never
raw rows, so monitoring does not copy personal data around.

Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 retrain.
New SECOP extracts drift naturally (new year, new UNSPSC codes, inflation,
changes of government), so drift is checked on every update.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from contract_transparency.config import Settings
from contract_transparency.data import columns as C

_EPS = 1e-4
_OTHER = "__other__"
_MISSING = "__missing__"


def _numeric_series(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "log_value": np.log1p(df[C.VALUE].clip(lower=0)),
        "log_duration_days": np.log1p(df[C.DURATION_DAYS].clip(lower=0)),
        "text_length": df[C.TEXT].fillna("").str.len().astype(float),
    }


def monitored_categorical(cfg: Settings, df: pd.DataFrame) -> list[str]:
    cols = cfg.drift.monitored_features or [*cfg.schema_.categorical, C.UNSPSC_SEGMENT]
    return [c for c in cols if c in df.columns]


def build_profile(
    df: pd.DataFrame, cfg: Settings, bins: int = 10, top_k: int = 50
) -> dict[str, Any]:
    prof: dict[str, Any] = {"n_rows": len(df), "numeric": {}, "categorical": {}}
    for name, s in _numeric_series(df).items():
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            continue
        counts, _ = np.histogram(np.clip(s, edges[0], edges[-1]), bins=edges)
        prof["numeric"][name] = {"edges": edges.tolist(), "props": (counts / counts.sum()).tolist()}
    for col in monitored_categorical(cfg, df):
        vc = df[col].fillna(_MISSING).astype(str).value_counts(normalize=True)
        top = vc.head(top_k)
        props = top.to_dict()
        if len(vc) > top_k:
            props[_OTHER] = float(vc.iloc[top_k:].sum())
        prof["categorical"][col] = props
    return prof


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def compare_to_profile(df: pd.DataFrame, profile: dict[str, Any], cfg: Settings) -> pd.DataFrame:
    rows = []
    numeric = _numeric_series(df)
    for name, ref in profile.get("numeric", {}).items():
        s = numeric[name].replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        edges = np.asarray(ref["edges"])
        counts, _ = np.histogram(np.clip(s, edges[0], edges[-1]), bins=edges)
        rows.append(
            {
                "feature": name,
                "kind": "numeric",
                "psi": psi(ref["props"], counts / max(counts.sum(), 1)),
            }
        )
    for col, ref in profile.get("categorical", {}).items():
        if col not in df.columns:
            rows.append({"feature": col, "kind": "categorical", "psi": np.nan})
            continue
        vc = df[col].fillna(_MISSING).astype(str).value_counts(normalize=True)
        known = [k for k in ref if k != _OTHER]
        cur = [float(vc.get(k, 0.0)) for k in known]
        exp = [ref[k] for k in known]
        if _OTHER in ref or vc.index.difference(known).size:
            exp.append(ref.get(_OTHER, 0.0))
            cur.append(float(vc[vc.index.difference(known)].sum()))
        rows.append({"feature": col, "kind": "categorical", "psi": psi(exp, cur)})
    out = pd.DataFrame(rows, columns=["feature", "kind", "psi"])
    out["status"] = np.select(
        [out["psi"] >= cfg.drift.psi_retrain, out["psi"] >= cfg.drift.psi_warning],
        ["retrain", "warning"],
        default="ok",
    )
    return out.sort_values("psi", ascending=False).reset_index(drop=True)
