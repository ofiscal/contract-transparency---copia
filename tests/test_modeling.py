"""Modelling components added in v1.2: text price model, quantity feature,
group calibration, blend, benchmark and neutral terminology."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from contract_transparency.config import NOTICE_EN, NOTICE_ES, ROW_NOTICE, with_overrides
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import training_mask
from contract_transparency.data.privacy import redact_text
from contract_transparency.data.splits import split_data
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.features.builder import text_quantity
from contract_transparency.features.text import SPANISH_STOP_WORDS
from contract_transparency.features.text_model import TextPriceModel
from contract_transparency.mlops.registry import IntegrityError, ModelRegistry
from contract_transparency.models.price_model import BLEND_GRID, PriceModel
from contract_transparency.scoring.anomaly import ResidualDistribution


# --------------------------------------------------------------------------- text features
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Compra de 92 equipos de cómputo", 92),
        ("Construcción de 3,5 km de placa huella", 3.5),
        ("Suministro de 2.500 galones de ACPM", 2500),
        ("Arrendamiento de inmueble de 250 m2", 250),
        ("Contrato No. 045 de 2023 para apoyo", None),  # contract number + year
        ("Prestación de servicios por 6 meses", None),  # duration has its own feature
        ("vigencia 2023 compra de 15 sillas", 15),
        ("", None),
        (None, None),
    ],
)
def test_text_quantity(text, expected):
    got = text_quantity(text)
    assert np.isnan(got) if expected is None else got == pytest.approx(expected)


def test_amounts_written_in_words_are_masked():
    out = redact_text(
        pd.Series(["por valor de 12 millones de pesos", "$ 1,5 mil millones", "40 sillas"])
    )
    assert out.tolist() == ["por valor de <NUM> de pesos", "<NUM>", "40 sillas"]


@pytest.fixture(scope="module")
def texts_and_prices():
    rng = np.random.default_rng(0)
    levels = {
        "auxiliar administrativo": 0.0,
        "técnico administrativo": 0.35,
        "abogado especializado": 0.8,
    }
    rows = []
    for _ in range(900):
        level = rng.choice(list(levels))
        rows.append(
            (
                f"Prestación de servicios como {level} en la secretaría",
                15 + levels[level] + rng.normal(0, 0.1),
            )
        )
    return pd.Series([r[0] for r in rows]), np.array([r[1] for r in rows])


def test_text_model_equals_sklearn_pipeline(texts_and_prices):
    texts, y = texts_and_prices
    model = TextPriceModel(word_max_features=500, char_max_features=500, min_df=2, n_folds=3)
    model.fit_oof(texts, y)
    common = {"lowercase": True, "strip_accents": "unicode", "sublinear_tf": True, "min_df": 2}
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        stop_words=SPANISH_STOP_WORDS,
        max_features=500,
        **common,
    )
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=500, **common)
    X = sp.hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    expected = Ridge(alpha=1.0).fit(X, y).predict(X)
    np.testing.assert_allclose(model.predict(texts), expected, atol=2e-6)


def test_text_model_learns_levels_and_oof_is_honest(texts_and_prices, tmp_path):
    texts, y = texts_and_prices
    model = TextPriceModel(min_df=2, n_folds=5)
    oof = model.fit_oof(texts, y)
    in_sample = model.predict(texts)
    assert np.mean(np.abs(oof - y)) >= np.mean(
        np.abs(in_sample - y)
    )  # out-of-fold is not optimistic
    pred = model.predict(
        ["servicios como abogado especializado", "servicios como auxiliar administrativo"]
    )
    assert pred[0] - pred[1] > 0.5  # learned the price of the professional level (true: 0.8)
    model.save(tmp_path)
    again = TextPriceModel.load(tmp_path)
    np.testing.assert_array_equal(again.predict(texts), model.predict(texts))
    assert np.isfinite(model.predict(["", None, "texto totalmente nuevo"])).all()


# --------------------------------------------------------------------------- calibration
def test_group_calibration_adapts_to_each_contract_type():
    rng = np.random.default_rng(1)
    groups = np.array(["Obra"] * 1000 + ["Prestación de servicios"] * 1000)
    r = np.concatenate([rng.normal(0, 0.5, 1000), rng.normal(0, 0.1, 1000)])
    dist = ResidualDistribution.fit(r, groups, group_column="Tipo de Contrato")
    assert dist.group_scales["Obra"] > 2 * dist.group_scales["Prestación de servicios"]
    # the same +0.6 log deviation is unusual for services, ordinary for public works
    tail = dist.tail_probability(
        np.array([0.6, 0.6]), np.array(["Obra", "Prestación de servicios"])
    )
    assert tail[1] < 0.01 < tail[0]
    # unknown groups fall back to the global scale
    assert dist.relative_scale(np.array(["Nuevo tipo"]), 1)[0] == 1.0
    # small groups are shrunk towards the global scale
    small = ResidualDistribution.fit(
        np.concatenate([r, [3.0, -3.0]]), np.concatenate([groups, ["X", "X"]])
    )
    assert 0.5 < small.group_scales["X"] < 10


def test_old_residual_distributions_still_load():
    old = {"df": 4.0, "loc": 0.0, "scale": 0.3, "n": 1000}  # v1.0/v1.1 metadata
    dist = ResidualDistribution.from_dict(old)
    assert dist.group_scales == {} and dist.tail_probability(np.array([0.0]))[0] == pytest.approx(
        0.5
    )


# --------------------------------------------------------------------------- blend
@pytest.fixture(scope="module")
def split(session_cfg):
    from contract_transparency.data.cleaning import clean_contracts

    df = clean_contracts(generate_contracts(4000, seed=5), session_cfg, b"k")
    return split_data(df[training_mask(df)].reset_index(drop=True), session_cfg.evaluation)


def test_blend_weight_is_chosen_on_validation_and_persisted(session_cfg, split, tmp_path):
    model = PriceModel(session_cfg).fit(split.train, split.valid)
    assert model.blend_weight_ in BLEND_GRID
    reg = ModelRegistry(tmp_path, "m")
    version = reg.register(model, {})
    meta = reg.metadata(version)
    if model.blend_weight_ > 0:
        assert any(k.startswith("linear/") for k in meta["artifacts"])
    loaded, _ = reg.load(version)
    np.testing.assert_allclose(loaded.predict(split.test), model.predict(split.test))
    fixed = PriceModel(with_overrides(session_cfg, {"model.blend_linear_weight": 0.5})).fit(
        split.train, split.valid
    )
    assert fixed.blend_weight_ == 0.5 and fixed.linear is not None
    off = PriceModel(with_overrides(session_cfg, {"model.blend_linear_weight": 0.0})).fit(
        split.train, split.valid
    )
    assert off.linear is None


def test_tampering_inside_the_linear_submodel_is_detected(session_cfg, split, tmp_path):
    model = PriceModel(with_overrides(session_cfg, {"model.blend_linear_weight": 0.5})).fit(
        split.train, split.valid
    )
    reg = ModelRegistry(tmp_path, "m")
    version = reg.register(model, {})
    with open(reg.versions_dir / version / "linear" / "ridge.npz", "ab") as fh:
        fh.write(b"x")
    with pytest.raises(IntegrityError):
        reg.load(version)


def test_new_model_is_more_accurate_than_v1_1(session_cfg, split):
    v11 = with_overrides(
        session_cfg,
        {
            "features.text_embedder": "tfidf_svd",
            "features.text_price_model": False,
            "features.add_text_quantity": False,
            "model.blend_linear_weight": 0.0,
        },
    )
    y = np.log1p(split.test[C.VALUE].to_numpy())
    mae_new = np.mean(
        np.abs(PriceModel(session_cfg).fit(split.train, split.valid).predict_log(split.test) - y)
    )
    mae_old = np.mean(
        np.abs(PriceModel(v11).fit(split.train, split.valid).predict_log(split.test) - y)
    )
    assert mae_new < 0.85 * mae_old


# --------------------------------------------------------------------------- benchmark
def test_benchmark_reports_accuracy_detection_and_cost(session_cfg, split):
    from contract_transparency.mlops.benchmark import run_benchmark, to_markdown

    data = pd.concat([split.train, split.valid, split.test], ignore_index=True)
    out = run_benchmark(session_cfg, data, ["xgb_no_text", "ridge_sparse"])
    assert set(out["variant"]) >= {
        "baseline_median",
        "xgb_no_text",
        "ridge_sparse",
        "xgb_no_text__global",
    }
    for col in (
        "mae_log",
        "alert_precision",
        "alert_recall",
        "fit_seconds",
        "predict_ms_per_1k",
        "model_mb",
    ):
        assert col in out.columns
    assert "| variant |" in to_markdown(out)
    with pytest.raises(KeyError, match="Unknown variant"):
        run_benchmark(session_cfg, data, ["nope"])


# --------------------------------------------------------------------------- terminology
FORBIDDEN = re.compile(
    r"riesgo|\brisk|sospech|suspicious|sobrecost|overprice|red.?flag|bandera", re.I
)


def test_notices_use_neutral_language():
    for text in (NOTICE_ES, NOTICE_EN, ROW_NOTICE):
        assert not FORBIDDEN.search(text), text
    assert "no asumen responsabilidad" not in NOTICE_ES  # wording: "ni asumen responsabilidad"
    assert "ni asumen responsabilidad" in NOTICE_ES and "accept no responsibility" in NOTICE_EN
    assert "revis" in ROW_NOTICE.lower()


def test_output_columns_use_neutral_language(session_cfg, split):
    from contract_transparency.provenance import DERIVED_COLUMNS
    from contract_transparency.scoring.rules import SIGNAL_PREFIX, alert_index, evaluate_rules

    flags = evaluate_rules(split.test, session_cfg)
    cols = list(flags.columns) + list(alert_index(flags, session_cfg.rules.weights).columns)
    assert all(
        c.startswith(SIGNAL_PREFIX)
        or c in {"alert_index", "n_signals", "signals", "days_to_next_restriction"}
        for c in cols
    )
    for text in [*cols, *DERIVED_COLUMNS, *DERIVED_COLUMNS.values(), *session_cfg.rules.weights]:
        assert not FORBIDDEN.search(text), text


def test_dashboard_text_uses_neutral_language():
    from pathlib import Path

    import contract_transparency

    source = (Path(contract_transparency.__file__).parent / "app" / "streamlit_app.py").read_text()
    labels = re.findall(r'"([^"]*[a-záéíóú][^"]*)"', source)
    offending = [label for label in labels if FORBIDDEN.search(label)]
    assert not offending, offending


def test_synthetic_anomaly_factor_is_recorded():
    df = generate_contracts(2000, seed=3, anomaly_factor=(2.0, 4.0))
    f = df.loc[df["synthetic_is_anomaly"], "synthetic_anomaly_factor"]
    assert (
        f.between(2.0, 4.0).all()
        and (df.loc[~df["synthetic_is_anomaly"], "synthetic_anomaly_factor"] == 1).all()
    )


def test_every_signal_has_a_neutral_spanish_label(session_cfg):
    from contract_transparency.scoring.rules import SIGNAL_LABELS, signal_label

    for code in session_cfg.rules.weights:
        assert code in SIGNAL_LABELS, code
    for label in SIGNAL_LABELS.values():
        assert not FORBIDDEN.search(label), label
    assert signal_label("sig_price_deviation") == SIGNAL_LABELS["price_deviation"]
    assert signal_label("unknown_code") == "unknown code"
