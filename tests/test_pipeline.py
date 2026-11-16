"""End-to-end tests: prepare -> train -> promote -> score -> update."""

import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from contract_transparency.config import PromotionConfig
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.mlops import pipeline
from contract_transparency.mlops.pipeline import decide_promotion
from tests.conftest import make_config


# ----------------------------------------------------------------- fast unit tests
def _m(mae, r2=0.8):
    return {"mae_log": mae, "r2_log": r2, "rmse_log": mae, "mdape": mae}


def test_promotion_gates():
    pc = PromotionConfig(
        min_improvement_over_baseline=0.05, min_relative_improvement=0.0, min_r2_log=0.2
    )
    ok, reasons = decide_promotion(_m(0.30), _m(0.50), None, pc)
    assert ok and "no current champion" in reasons
    ok, _ = decide_promotion(_m(0.49), _m(0.50), None, pc)  # only 2% better than baseline
    assert not ok
    ok, _ = decide_promotion(_m(0.30), _m(0.50), _m(0.29), pc)  # worse than champion
    assert not ok
    ok, _ = decide_promotion(_m(0.30, r2=0.1), _m(0.50), None, pc)  # r2 too low
    assert not ok
    ok, _ = decide_promotion(_m(0.28), _m(0.50), _m(0.29), pc)
    assert ok


# ----------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def _trained_root(tmp_path_factory):
    """Train ONCE per module; tests get a private copy of the whole tree."""
    root = tmp_path_factory.mktemp("trained")
    cfg = make_config(root)
    Path(cfg.paths.raw_data).parent.mkdir(parents=True)
    # clear-cut x3-x10 anomalies: this test checks the wiring, not the statistical limits
    generate_contracts(3000, seed=11, end="2023-06-30", anomaly_factor=(3.0, 10.0)).to_csv(
        cfg.paths.raw_data, index=False
    )
    data, report = pipeline.prepare(cfg)
    result = pipeline.train(cfg, data)
    return root, report, result


@pytest.fixture
def trained(_trained_root, tmp_path):
    root, report, result = _trained_root
    shutil.copytree(root, tmp_path / "w")
    cfg = make_config(tmp_path / "w")
    return cfg, pipeline.load_processed(cfg), report, result


@pytest.mark.slow
def test_full_pipeline(trained):
    cfg, data, report, result = trained
    assert report["raw"]["ok"] and pipeline.processed_path(cfg).exists()
    assert result.version == "v0001" and result.promoted
    m = result.metrics
    assert m["test"]["mae_log"] < m["baseline_test"]["mae_log"]
    assert m["test"]["r2_log"] > 0.6
    # planted x6-x30 anomalies must be found with few false alerts
    assert m["detection_test"]["alert_recall"] > 0.8
    assert m["detection_test"]["alert_precision"] > 0.6

    meta = pipeline.get_registry(cfg).metadata("champion")
    for key in (
        "residual_distribution",
        "drift_profile",
        "top_features",
        "git_commit",
        "config_fingerprint",
        "artifacts",
    ):
        assert key in meta
    assert meta["data"]["train_dates"][1] <= meta["data"]["test_dates"][0]

    scored, path = pipeline.score(cfg, data=data)
    assert path.exists() and len(scored) == len(data)
    assert scored["review_priority"].tolist() == list(range(1, len(scored) + 1))
    assert scored["alert_index"].is_monotonic_decreasing
    top = scored.head(60)
    assert top["sig_price_deviation"].mean() > 0.5


@pytest.mark.slow
def test_scored_output_contains_no_personal_data(trained):
    cfg, data, _, _ = trained
    _, path = pipeline.score(cfg, data=data)
    text = path.read_text(encoding="utf-8-sig")
    raw = pd.read_csv(cfg.paths.raw_data, dtype=str)
    people = raw.loc[
        raw["TipoDocProveedor"] == "Cédula de Ciudadanía", "Documento Proveedor"
    ].unique()
    found = set(re.findall(r"(?<![\d.])\d{10}(?![\d.])", text)) & set(people)
    assert not found, sorted(found)[:5]
    for leaked in (
        "Contratista Ficticio",
        "Representante Ficticio",
        "Supervisor Ficticio",
        "000-DEMO",
        "Banco Demo",
        "C.C. 1",
    ):
        assert leaked not in text


@pytest.mark.slow
def test_retrain_compares_with_champion_and_can_reject(trained):
    cfg, data, _, first = trained
    strict = cfg.model_copy(
        update={"promotion": cfg.promotion.model_copy(update={"min_relative_improvement": 0.5})}
    )
    second = pipeline.train(strict, data)
    assert not second.promoted and "champion_test" in second.metrics
    reg = pipeline.get_registry(cfg)
    assert reg.champion() == first.version
    assert reg.metadata(second.version)["promotion"]["decision"] == "rejected"
    assert reg.history()[-1]["event"] == "rejected"


@pytest.mark.slow
def test_update_detects_new_modified_and_retrains(trained, write_raw):
    cfg, _, _, _ = trained
    raw = pd.read_csv(cfg.paths.raw_data, dtype=str)
    new = generate_contracts(800, seed=99, start="2023-07-01", end="2023-12-31")
    new["ID Contrato"] = new["ID Contrato"].str.replace("DEMO", "NEW", regex=False)
    changed = raw.head(5).copy()
    changed["Valor del Contrato"] = "999999000"  # value modified after signature (otrosí)
    path = write_raw(pd.concat([changed, new], ignore_index=True), "extract_2023h2.csv")

    summary = pipeline.update(cfg, path)
    assert summary["n_new"] == 800 and summary["n_modified"] == 5
    assert summary["n_total"] == 3800
    assert summary["trained_version"] == "v0002" and any(
        "new contracts" in r for r in summary["retrain_reasons"]
    )
    scored = pd.read_csv(summary["scored_file"], sep=";")
    assert len(scored) == 805
    drift = pd.DataFrame(summary["drift"])
    assert {"feature", "psi", "status"} <= set(drift.columns)


@pytest.mark.slow
def test_check_drift(trained):
    cfg, data, _, _ = trained
    report, retrain = pipeline.check_drift(cfg, data)
    assert not retrain and (report["psi"] < 0.1).all()
