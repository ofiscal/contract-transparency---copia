"""Naive, interpretable baseline: hierarchical median price.

Every candidate model must beat this baseline by a margin (see
``promotion.min_improvement_over_baseline``) before it can be promoted. If a
complex model cannot beat "the median value of similar contracts", the extra
complexity is not justified -- a check the legacy experiments never had.

It also replaces the OLS / MLP baselines of ``models/baseline.py`` (which
needed hand-made dummy matrices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MedianBaseline:
    """Median of log(value) by the most specific group with enough support."""

    def __init__(self, group_levels: list[list[str]], min_support: int = 10):
        self.group_levels = group_levels
        self.min_support = min_support
        self.tables_: list[tuple[list[str], pd.Series]] = []
        self.global_: float = 0.0

    def fit(self, df: pd.DataFrame, log_y: np.ndarray) -> MedianBaseline:
        data = df.copy()
        data["_y"] = np.asarray(log_y, dtype=float)
        self.global_ = float(np.median(data["_y"]))
        self.tables_ = []
        for keys in self.group_levels:
            keys = [k for k in keys if k in data.columns]
            if not keys:
                continue
            g = data.groupby(keys, dropna=False)["_y"].agg(["median", "size"])
            g = g[g["size"] >= self.min_support]["median"]
            self.tables_.append((keys, g))
        return self

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        pred = pd.Series(np.nan, index=df.index)
        for keys, table in self.tables_:  # most specific first
            if table.empty or not all(k in df.columns for k in keys):
                continue
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
            vals = pd.Series(table.reindex(idx).to_numpy(), index=df.index)
            pred = pred.fillna(vals)
        return pred.fillna(self.global_).to_numpy()
