"""Compare model architectures on accuracy AND cost (``ct benchmark``).

Every variant is trained on the same temporal split of the same data and
reports:

* accuracy   -- MAE / RMSE / R² in log space, MdAPE (on the most recent contracts)
* detection  -- alert precision / recall / false-alert rate when known anomalies
                exist (``evaluation.label_column``), and how uneven the
                false-alert rate is across contract types
* cost       -- training time, scoring time per 1 000 contracts, model size on
                disk, number of features

Run it on the REAL extract before changing defaults: conclusions drawn on
synthetic data do not automatically transfer.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract_transparency.config import Settings, with_overrides
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import training_mask
from contract_transparency.data.splits import split_data
from contract_transparency.mlops.metrics import detection_metrics, regression_metrics
from contract_transparency.mlops.pipeline import calibration_column, parse_labels
from contract_transparency.models.baseline import MedianBaseline
from contract_transparency.models.segmented import build_model
from contract_transparency.scoring.anomaly import ResidualDistribution, group_keys, score_prices

logger = logging.getLogger(__name__)

# name -> (description, dotted config overrides)
NO_BLEND = {"model.blend_linear_weight": 0.0}
# v1.2 and earlier: one model, no person type
SINGLE = {"model.segment_by": None, "features.add_person_type": False}
VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "xgb_no_text": (
        "XGBoost on categorical + numeric features only",
        {
            "features.text_embedder": "none",
            "features.text_price_model": False,
            "features.add_text_quantity": False,
            **NO_BLEND,
            **SINGLE,
        },
    ),
    "xgb_tfidf_svd": (
        "v1.1: XGBoost + TF-IDF compressed by SVD",
        {
            "features.text_embedder": "tfidf_svd",
            "features.text_price_model": False,
            "features.add_text_quantity": False,
            **NO_BLEND,
            **SINGLE,
        },
    ),
    "xgb_text_model": (
        "XGBoost + stacked text price model (word+char TF-IDF -> Ridge, out-of-fold)",
        {"features.add_text_quantity": False, **NO_BLEND, **SINGLE},
    ),
    "xgb_text_model_qty": (
        "+ quantity extracted from the description",
        {**NO_BLEND, **SINGLE},
    ),
    "ridge_sparse": (
        "Linear model on sparse one-hot + TF-IDF (cheap reference)",
        {
            "model.type": "ridge",
            "features.text_embedder": "tfidf",
            "features.text_price_model": False,
            **NO_BLEND,
            **SINGLE,
        },
    ),
    "v1_2": (
        "v1.2 default: text model + quantity + blend, one model",
        {**SINGLE},
    ),
    "v1_2_person_type": (
        "v1.2 + person type (natural / juridica) as a feature, one model",
        {"model.segment_by": None},
    ),
    "segmented": (
        "v1.3 default: one model per person type (+ global fallback), calibrated by person type x contract type",
        {},
    ),
    "xgb_sentence_embeddings": (
        "XGBoost + multilingual sentence embeddings (needs `uv sync --extra nlp` and a model download)",
        {
            "features.text_embedder": "sentence_transformer",
            "features.text_price_model": False,
            **NO_BLEND,
            **SINGLE,
        },
    ),
}
DEFAULT_VARIANTS = [
    "xgb_no_text",
    "xgb_tfidf_svd",
    "xgb_text_model",
    "xgb_text_model_qty",
    "ridge_sparse",
    "v1_2",
    "v1_2_person_type",
    "segmented",
]


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _false_alert_spread(
    test: pd.DataFrame, alerts: np.ndarray, truth: np.ndarray, col: str
) -> dict:
    """False-alert rate per contract type (clean contracts only)."""
    if col not in test.columns:
        return {}
    clean = ~truth
    rates = pd.Series(alerts[clean]).groupby(test[col].to_numpy()[clean]).mean()
    rates = rates[test[col][clean].value_counts().reindex(rates.index) >= 30]
    if rates.empty:
        return {}
    return {
        "false_alert_rate_max_type": float(rates.max()),
        "false_alert_rate_min_type": float(rates.min()),
    }


def _sentence_available(cfg: Settings) -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def run_benchmark(
    cfg: Settings, data: pd.DataFrame, variants: list[str] | None = None
) -> pd.DataFrame:
    variants = variants or DEFAULT_VARIANTS
    usable = data[training_mask(data)].reset_index(drop=True)
    split = split_data(usable, cfg.evaluation)
    label = cfg.evaluation.label_column
    truth = parse_labels(split.test, label)  # same parsing as `ct train`
    y_test = split.test[C.VALUE].to_numpy()
    y_valid = split.valid[C.VALUE].to_numpy()
    group_col = cfg.scoring.calibration_column or cfg.schema_.contract_type
    rows: list[dict[str, Any]] = []

    # Naive baseline (no training cost to speak of)
    s = cfg.schema_
    base = MedianBaseline(
        [
            [s.contract_type, s.modality, C.UNSPSC_CLASS],
            [s.contract_type, C.UNSPSC_FAMILY],
            [s.contract_type],
        ]
    ).fit(split.train, np.log1p(split.train[C.VALUE]))
    rows.append(
        {
            "variant": "baseline_median",
            "description": "Median price of similar contracts",
            **regression_metrics(y_test, np.expm1(base.predict_log(split.test))),
        }
    )

    for name in variants:
        if name not in VARIANTS:
            raise KeyError(f"Unknown variant '{name}'. Available: {list(VARIANTS)}")
        description, overrides = VARIANTS[name]
        if name == "xgb_sentence_embeddings" and not _sentence_available(cfg):
            logger.warning("Skipping %s: sentence-transformers not installed", name)
            continue
        vcfg = with_overrides(cfg, overrides)
        logger.info("Benchmark: %s", name)
        t0 = time.perf_counter()
        model = build_model(vcfg).fit(split.train, split.valid)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred_test = model.predict(split.test)
        predict_ms = (time.perf_counter() - t0) / max(len(split.test), 1) * 1e6  # per 1k rows
        pred_valid = model.predict(split.valid)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            size_mb = _dir_size_mb(Path(tmp))
        row = {
            "variant": name,
            "description": description,
            **regression_metrics(y_test, pred_test),
            "fit_seconds": round(fit_s, 2),
            "predict_ms_per_1k": round(predict_ms, 1),
            "model_mb": round(size_mb, 2),
            "n_features": len(model.feature_names_),
        }
        if truth is not None:
            for calibration in ("global", "by_group"):
                ccfg = with_overrides(vcfg, {"scoring.calibration": calibration})
                spec = calibration_column(ccfg)
                groups = group_keys(split.valid, spec)
                dist = ResidualDistribution.fit(
                    np.log1p(y_valid) - np.log1p(pred_valid),
                    groups,
                    group_column=spec if groups is not None else None,
                )
                alerts = score_prices(split.test, pred_test, dist, cfg.scoring)[
                    "price_deviation"
                ].to_numpy()
                det = detection_metrics(truth, alerts)
                det["false_alert_rate"] = float(alerts[~truth].mean())
                det.update(_false_alert_spread(split.test, alerts, truth, group_col))
                suffix = "" if calibration == cfg.scoring.calibration else f"__{calibration}"
                if suffix:  # report the non-default calibration as its own row
                    rows.append({**row, "variant": f"{name}{suffix}", **det})
                else:
                    row.update(det)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(1, "n_train", len(split.train))
    return out


def to_markdown(df: pd.DataFrame) -> str:
    cols = [
        c
        for c in [
            "variant",
            "mae_log",
            "mdape",
            "r2_log",
            "alert_precision",
            "alert_recall",
            "false_alert_rate",
            "false_alert_rate_max_type",
            "false_alert_rate_min_type",
            "fit_seconds",
            "predict_ms_per_1k",
            "model_mb",
            "n_features",
        ]
        if c in df.columns
    ]
    table = df[cols].copy()
    for c in cols[1:]:
        table[c] = table[c].map(
            lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
        )
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in table.to_numpy()]
    return "\n".join(lines)
