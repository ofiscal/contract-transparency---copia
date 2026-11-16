"""End-to-end pipeline stages.

    prepare  -> read raw extract, apply privacy rules, clean, validate, snapshot
    train    -> temporal split, fit model + baseline, evaluate, fit residual
                distribution, register a new version, promote if it passes gates
    score    -> expected prices, price deviations, rule signals, alert index for review
    drift    -> compare new data with the champion's training profile
    update   -> incremental loop for new SECOP extracts:
                new/modified contracts -> drift -> retrain? -> promote? -> score

Replaces ``main.py`` / ``maintry.py`` whose plan was exactly:
"load the dataset using cleaning -> train the models or update them -> run
the models with the dataset -> send the updated excel file to tableau".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract_transparency.config import PromotionConfig, Settings
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import clean_contracts, training_mask
from contract_transparency.data.incremental import find_modified_contracts, find_new_contracts
from contract_transparency.data.io import read_contracts, write_table
from contract_transparency.data.privacy import get_salt
from contract_transparency.data.records import load_records, match_records
from contract_transparency.data.splits import split_data
from contract_transparency.data.validation import validate_clean, validate_raw
from contract_transparency.environment import (
    ModelCompatibilityError,
    compatibility_hint,
    library_versions,
)
from contract_transparency.mlops.drift import build_profile, compare_to_profile
from contract_transparency.mlops.metrics import detection_metrics, regression_metrics
from contract_transparency.mlops.registry import ModelRegistry
from contract_transparency.mlops.tracking import Tracker
from contract_transparency.models.baseline import MedianBaseline
from contract_transparency.models.segmented import build_model
from contract_transparency.provenance import (
    audit_links,
    build_manifest,
    code_version,
    contracts_source,
    write_manifest,
)
from contract_transparency.scoring.anomaly import ResidualDistribution, group_keys, score_prices
from contract_transparency.scoring.rules import SIGNAL_PREFIX, alert_index, evaluate_rules

logger = logging.getLogger(__name__)

PROCESSED_FILE = "contracts.parquet"
PREPARE_REPORT = "prepare_report.json"


def get_registry(cfg: Settings) -> ModelRegistry:
    return ModelRegistry(cfg.paths.registry_dir, cfg.model.name)


def processed_path(cfg: Settings) -> Path:
    return Path(cfg.paths.processed_dir) / PROCESSED_FILE


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def prepare(
    cfg: Settings, input_path: str | Path | None = None, nrows: int | None = None, save: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(input_path or cfg.paths.raw_data)
    raw = read_contracts(path, cfg, nrows)
    raw_report = validate_raw(raw, cfg)
    for w in raw_report.warnings:
        logger.warning(w)
    raw_report.raise_if_failed()

    clean = clean_contracts(raw, cfg, get_salt(cfg))
    clean_report = validate_clean(clean, cfg, for_training=False)
    for w in clean_report.warnings:
        logger.warning(w)
    report = {
        "source": contracts_source(cfg, path, len(raw)),
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
        "config_fingerprint": cfg.fingerprint(),
        "raw": raw_report.to_dict(),
        "clean": clean_report.to_dict(),
    }
    if save:
        out = processed_path(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(out, index=False)
        (out.parent / PREPARE_REPORT).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Processed snapshot saved to %s", out)
    return clean, report


def snapshot_source(cfg: Settings) -> dict[str, Any] | None:
    """Provenance of the processed snapshot (from its prepare report)."""
    path = processed_path(cfg).parent / PREPARE_REPORT
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("source")


def load_processed(cfg: Settings) -> pd.DataFrame:
    path = processed_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `ct prepare` first")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# train / evaluate / promote
# --------------------------------------------------------------------------- #
@dataclass
class TrainResult:
    version: str
    promoted: bool
    reasons: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)


def decide_promotion(
    candidate: dict[str, float],
    baseline: dict[str, float],
    champion: dict[str, float] | None,
    pc: PromotionConfig,
) -> tuple[bool, list[str]]:
    """Quality gates (all metrics are 'lower is better' except r2)."""
    m = pc.metric
    ok, reasons = True, []
    if not candidate.get("r2_log", -np.inf) >= pc.min_r2_log:
        ok = False
        reasons.append(f"r2_log {candidate.get('r2_log'):.3f} < minimum {pc.min_r2_log}")
    base_gain = (baseline[m] - candidate[m]) / baseline[m] if baseline[m] else 0.0
    if base_gain < pc.min_improvement_over_baseline:
        ok = False
        reasons.append(
            f"{m} improves baseline by {base_gain:.1%} < required {pc.min_improvement_over_baseline:.1%}"
        )
    else:
        reasons.append(f"{m} improves baseline by {base_gain:.1%}")
    if champion is None:
        reasons.append("no current champion")
    else:
        champ_gain = (champion[m] - candidate[m]) / champion[m] if champion[m] else 0.0
        if champ_gain < pc.min_relative_improvement:
            ok = False
            reasons.append(
                f"{m} vs champion {champ_gain:+.1%} (needs >= {pc.min_relative_improvement:+.1%})"
            )
        else:
            reasons.append(f"{m} vs champion {champ_gain:+.1%}")
    return ok, reasons


def parse_labels(df: pd.DataFrame, col: str | None) -> np.ndarray | None:
    """Known-anomaly labels as booleans ("true", "1", "si", "sí", "yes"); None if absent."""
    if not col or col not in df.columns:
        return None
    return (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "si", "sí", "yes"})
        .to_numpy()
    )


def _date_range(df: pd.DataFrame) -> list[str | None]:
    d = df[C.SIGNED_AT].dropna()
    return [str(d.min().date()), str(d.max().date())] if len(d) else [None, None]


def calibration_column(cfg: Settings) -> str | None:
    """Calibration groups: contract type, and also person type when the model is
    segmented by it (a professional's fee and a firm's invoice vary differently)."""
    if cfg.scoring.calibration != "by_group":
        return None
    base = cfg.scoring.calibration_column or cfg.schema_.contract_type
    if cfg.model.segment_by == "person_type" and base:
        return f"{C.PERSON_TYPE}|{base}"
    return base


def fit_residual_distribution(
    cfg: Settings, valid: pd.DataFrame, y_valid: np.ndarray, pred_valid: np.ndarray
) -> ResidualDistribution:
    """Residual distribution on out-of-sample (validation) contracts."""
    spec = calibration_column(cfg)
    groups = group_keys(valid, spec)
    residuals = np.log1p(y_valid) - np.log1p(pred_valid)
    return ResidualDistribution.fit(
        residuals, groups, group_column=spec if groups is not None else None
    )


def weak_label_metrics(test: pd.DataFrame, alerts: np.ndarray, cfg: Settings) -> dict[str, Any]:
    """Alert rate among contracts whose supplier appears in an official record
    (e.g. Contraloría bulletin) vs. the rest. A *weak* label: being listed is
    not the same as an irregular contract, but a useful model should alert
    more often on these suppliers (lift > 1)."""
    records = {k: v for k, v in load_records(cfg).items() if k in cfg.records.weak_label_sources}
    out: dict[str, Any] = {}
    if not records:
        return out
    matched = match_records(test, cfg, records)
    for source in records:
        m = matched["sig_" + cfg.records.sources[source].flag].to_numpy()
        rate_in = float(alerts[m].mean()) if m.any() else float("nan")
        rate_out = float(alerts[~m].mean()) if (~m).any() else float("nan")
        out[source] = {
            "n_matched": int(m.sum()),
            "alert_rate_matched": rate_in,
            "alert_rate_other": rate_out,
            "lift": rate_in / rate_out if rate_out and not np.isnan(rate_in) else float("nan"),
        }
    return out


def train(
    cfg: Settings, data: pd.DataFrame | None = None, tracker: Tracker | None = None
) -> TrainResult:
    df = data if data is not None else load_processed(cfg)
    report = validate_clean(df, cfg, for_training=True)
    report.raise_if_failed()
    usable = df[training_mask(df)].reset_index(drop=True)
    split = split_data(usable, cfg.evaluation)
    logger.info("Split sizes: %s", split.sizes())
    tracker = tracker or Tracker(cfg.tracking)
    registry = get_registry(cfg)

    with tracker.run(f"train-{cfg.model.name}", tags={"config": cfg.fingerprint()}):
        model = build_model(cfg).fit(split.train, split.valid)

        y_valid, y_test = split.valid[C.VALUE].to_numpy(), split.test[C.VALUE].to_numpy()
        pred_valid, pred_test = model.predict(split.valid), model.predict(split.test)

        s = cfg.schema_
        baseline = MedianBaseline(
            [
                [s.contract_type, s.modality, C.UNSPSC_CLASS],
                [s.contract_type, C.UNSPSC_FAMILY],
                [s.contract_type],
            ]
        ).fit(split.train, np.log1p(split.train[C.VALUE]))
        pred_base = np.expm1(baseline.predict_log(split.test))

        metrics: dict[str, Any] = {
            "valid": regression_metrics(y_valid, pred_valid),
            "test": regression_metrics(y_test, pred_test),
            "baseline_test": regression_metrics(y_test, pred_base),
        }
        dist = fit_residual_distribution(cfg, split.valid, y_valid, pred_valid)

        alerts = score_prices(split.test, pred_test, dist, cfg.scoring)[
            "price_deviation"
        ].to_numpy()
        truth = parse_labels(split.test, cfg.evaluation.label_column)
        if truth is not None:
            metrics["detection_test"] = detection_metrics(truth, alerts)
        weak = weak_label_metrics(split.test, alerts, cfg)
        if weak:
            metrics["weak_labels_test"] = weak

        champion_version = registry.champion()
        champion_metrics = None
        if champion_version:
            try:
                champ, _ = registry.load("champion")
                champion_metrics = regression_metrics(y_test, champ.predict(split.test))
                metrics["champion_test"] = champion_metrics
            except Exception as exc:  # schema change etc. -> candidate judged on gates only
                logger.warning("Champion %s could not be evaluated: %s", champion_version, exc)

        promote_ok, reasons = decide_promotion(
            metrics["test"], metrics["baseline_test"], champion_metrics, cfg.promotion
        )

        metadata = {
            "model_name": cfg.model.name,
            "model_type": cfg.model.type,
            "config_fingerprint": cfg.fingerprint(),
            "git_commit": code_version()["git_commit"],  # same source as output manifests
            "libraries": library_versions(),
            "data": {
                "n_train": len(split.train),
                "n_valid": len(split.valid),
                "n_test": len(split.test),
                "train_dates": _date_range(split.train),
                "valid_dates": _date_range(split.valid),
                "test_dates": _date_range(split.test),
                "split": cfg.evaluation.split,
                "validation": report.to_dict(),
            },
            "metrics": metrics,
            "residual_distribution": dist.to_dict(),
            "best_iteration": model.best_iteration_,
            "top_features": model.top_features(25).to_dict(orient="records"),
            "drift_profile": build_profile(split.train, cfg),
            "promotion": {
                "eligible": promote_ok,
                "reasons": reasons,
                "compared_with": champion_version,
            },
        }
        version = registry.register(model, metadata)

        promoted = False
        if promote_ok and cfg.promotion.auto_promote:
            registry.promote(version, reason="auto: " + "; ".join(reasons))
            promoted = True
        elif not promote_ok:
            registry.log_rejection(version, reasons)
        registry.update_metadata(
            version,
            promotion={
                **metadata["promotion"],
                "decision": "promoted" if promoted else ("eligible" if promote_ok else "rejected"),
            },
        )

        tracker.log_params(
            {
                "model": cfg.model.model_dump(),
                "features": cfg.features.model_dump(),
                "split": split.sizes(),
            }
        )
        for name, values in metrics.items():
            tracker.log_metrics(values, prefix=f"{name}.")
        tracker.set_tags(
            {
                "registry_version": version,
                "promoted": promoted,
                "git_commit": metadata["git_commit"],
            }
        )
        tracker.log_artifacts(registry.versions_dir / version)

    logger.info("Registered %s (promoted=%s): %s", version, promoted, "; ".join(reasons))
    return TrainResult(version=version, promoted=promoted, reasons=reasons, metrics=metrics)


def promote(cfg: Settings, version: str, reason: str = "manual") -> None:
    get_registry(cfg).promote(version, reason=reason)


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def _assemble_output(
    df: pd.DataFrame,
    price: pd.DataFrame,
    flags: pd.DataFrame,
    index: pd.DataFrame,
    cfg: Settings,
    version: str,
    segment: pd.Series | None = None,
) -> pd.DataFrame:
    # The contract id is always published: it is the key to audit every row.
    wanted = [cfg.schema_.id, *cfg.output.passthrough_columns]
    passthrough = [c for c in dict.fromkeys(wanted) if c in df.columns]
    base = df[passthrough].copy()
    links = audit_links(df, df[C.ID], cfg)
    base.insert(min(1, len(base.columns)), "secop_url", links["secop_url"])
    base.insert(min(2, len(base.columns)), "open_data_url", links["open_data_url"])
    base["supplier"] = df[C.SUPPLIER_DISPLAY]
    base["supplier_pseudonym"] = df[C.SUPPLIER]
    base["natural_person"] = df[C.NATURAL_PERSON]
    if C.PERSON_TYPE in df.columns:
        base["person_type"] = df[C.PERSON_TYPE]
    if segment is not None:  # which model scored the contract
        base["model_segment"] = segment
    base["contract_value"] = df[C.VALUE]
    base["duration_days"] = df[C.DURATION_DAYS].round(1)
    out = pd.concat([base, price, index, flags], axis=1)
    out["model_version"] = version
    if cfg.public_data.add_row_notice:
        out["aviso"] = cfg.public_data.row_notice
    out = out.sort_values(["alert_index", "weighted_excess"], ascending=False, kind="stable")
    out.insert(0, "review_priority", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def score(
    cfg: Settings,
    input_path: str | Path | None = None,
    data: pd.DataFrame | None = None,
    version: str = "champion",
    output_path: str | Path | None = None,
    write: bool = True,
    source: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    """Score contracts and write the result + its provenance manifest.

    The returned frame carries the manifest in ``scored.attrs["manifest"]``.
    """
    if data is None:
        data, report = prepare(cfg, input_path, save=False)
        source = report["source"]
    model, meta = get_registry(cfg).load(version)
    dist = ResidualDistribution.from_dict(meta["residual_distribution"])

    try:
        predicted = model.predict(data)
    except Exception as exc:  # re-raised below, with advice for legacy models
        registry = get_registry(cfg)
        if registry.is_portable(meta["version"]):
            raise
        # Legacy pickled models can load but fail later under a newer sklearn.
        raise ModelCompatibilityError(
            f"Model {meta['version']} failed to predict ({exc!s}). "
            + compatibility_hint(meta["version"], ["legacy v1.0 format"])
        ) from exc
    price = score_prices(data, predicted, dist, cfg.scoring)
    flags = evaluate_rules(data, cfg)
    flags[SIGNAL_PREFIX + "price_deviation"] = price.pop("price_deviation")
    records = load_records(cfg)
    if records:
        flags = flags.join(match_records(data, cfg, records))
    index = alert_index(flags, cfg.rules.weights)
    segment = model.segment_of(data) if hasattr(model, "segment_of") else None
    scored = _assemble_output(data, price, flags, index, cfg, meta["version"], segment)

    # Public-data policy: publish only rows a third party can audit at the source.
    n_total = len(scored)
    if cfg.public_data.require_secop_url:
        auditable = scored["secop_url"].notna()
        if (~auditable).any():
            logger.warning(
                "%d of %d contracts have no valid official SECOP link and are left out of the "
                "output (public_data.require_secop_url)",
                int((~auditable).sum()),
                n_total,
            )
        scored = scored[auditable].reset_index(drop=True)
        scored["review_priority"] = np.arange(1, len(scored) + 1)

    sources = [source] if source else []
    sources += [man for _, man in records.values()]
    manifest = build_manifest(
        cfg,
        list(scored.columns),
        sources,
        meta,
        rows={
            "input": n_total,
            "published": len(scored),
            "excluded_not_auditable": n_total - len(scored),
            "price_deviations": int(scored["sig_price_deviation"].sum()),
        },
    )
    scored.attrs["manifest"] = manifest

    path = None
    if write:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (
            Path(output_path)
            if output_path
            else Path(cfg.paths.outputs_dir) / f"scored_{meta['version']}_{stamp}"
        )
        fmt = cfg.output.format if not Path(path).suffix else None
        path = write_table(scored, path, fmt, cfg.output.csv_sep, manifest=manifest)
        write_manifest(path, manifest)
    return scored, path


# --------------------------------------------------------------------------- #
# drift & incremental update
# --------------------------------------------------------------------------- #
def check_drift(
    cfg: Settings, data: pd.DataFrame, version: str = "champion"
) -> tuple[pd.DataFrame, bool]:
    meta = get_registry(cfg).metadata(version)
    report = compare_to_profile(data, meta["drift_profile"], cfg)
    return report, bool((report["status"] == "retrain").any())


def update(
    cfg: Settings, new_input_path: str | Path, force_retrain: bool = False
) -> dict[str, Any]:
    """Incremental MLOps loop for a fresh SECOP extract."""
    new, report = prepare(cfg, new_input_path, save=False)
    snap = processed_path(cfg)
    previous = pd.read_parquet(snap) if snap.exists() else new.iloc[0:0]

    new_rows = find_new_contracts(previous, new, C.ID)
    compare_cols = [C.VALUE, C.END_AT, C.DURATION_DAYS]
    if cfg.schema_.days_added:
        compare_cols.append(cfg.schema_.days_added)
    modified = find_modified_contracts(previous, new, C.ID, compare_cols)
    summary: dict[str, Any] = {"n_new": len(new_rows), "n_modified": len(modified), "n_total": 0}

    registry = get_registry(cfg)
    drift_retrain = False
    if registry.champion() and len(new_rows):
        drift_report, drift_retrain = check_drift(cfg, new_rows)
        summary["drift"] = drift_report.to_dict(orient="records")

    combined = pd.concat([previous[~previous[C.ID].isin(new[C.ID])], new], ignore_index=True)
    summary["n_total"] = len(combined)
    snap.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(snap, index=False)

    reasons = []
    if force_retrain:
        reasons.append("forced")
    if registry.champion() is None:
        reasons.append("no champion")
    if len(new_rows) >= cfg.drift.min_new_rows_for_retrain:
        reasons.append(f"{len(new_rows)} new contracts >= {cfg.drift.min_new_rows_for_retrain}")
    if drift_retrain:
        reasons.append("data drift above psi_retrain")
    summary["retrain_reasons"] = reasons
    if reasons:
        result = train(cfg, combined)
        summary["trained_version"] = result.version
        summary["promoted"] = result.promoted
        summary["promotion_reasons"] = result.reasons

    to_score = combined[combined[C.ID].isin(pd.concat([new_rows[C.ID], modified[C.ID]]))]
    if len(to_score) and registry.champion():
        _, path = score(cfg, data=to_score.reset_index(drop=True), source=report["source"])
        summary["scored_file"] = str(path)
    return summary
