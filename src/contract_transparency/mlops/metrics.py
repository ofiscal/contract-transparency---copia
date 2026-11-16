"""Evaluation metrics.

The legacy scripts computed ``r2_score`` on raw COP values (dominated by a
handful of huge contracts) and discarded the result. The primary metrics are
now on the log scale, plus MdAPE, which is easy to explain: "half of the
contracts are predicted within X % of their real value".
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, r2_score, recall_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    lt, lp = np.log1p(np.clip(y_true, 0, None)), np.log1p(np.clip(y_pred, 0, None))
    err = lp - lt
    ape = np.abs(y_pred - y_true) / np.where(y_true > 0, y_true, np.nan)
    return {
        "n": int(len(y_true)),
        "mae_log": float(np.mean(np.abs(err))),
        "rmse_log": float(np.sqrt(np.mean(err**2))),
        "r2_log": float(r2_score(lt, lp)) if len(lt) > 1 else float("nan"),
        "mdape": float(np.nanmedian(ape)),
        "share_within_50pct": float(np.nanmean(ape <= 0.5)),
        "r2_raw": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def detection_metrics(truth: np.ndarray, alerts: np.ndarray) -> dict[str, float]:
    """Precision / recall of alerts against known anomalies (synthetic data or
    audited cases)."""
    truth = np.asarray(truth, dtype=bool)
    alerts = np.asarray(alerts, dtype=bool)
    return {
        "alert_rate": float(alerts.mean()),
        "alert_precision": float(precision_score(truth, alerts, zero_division=0)),
        "alert_recall": float(recall_score(truth, alerts, zero_division=0)),
    }
