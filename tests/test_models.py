import numpy as np
import pandas as pd
import pytest

from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import training_mask
from contract_transparency.data.splits import split_data
from contract_transparency.features.builder import build_preprocessor
from contract_transparency.models.baseline import MedianBaseline
from contract_transparency.models.price_model import PriceModel
from contract_transparency.models.target import TargetTransform


@pytest.fixture(scope="module")
def split(clean_df, session_cfg):
    return split_data(
        clean_df[training_mask(clean_df)].reset_index(drop=True), session_cfg.evaluation
    )


@pytest.fixture(scope="module")
def fitted(split, session_cfg):
    return PriceModel(session_cfg).fit(split.train, split.valid)


def test_temporal_split_has_no_overlap_and_is_ordered(split):
    """Regression for legacy main.py, which predicted on rows used for training."""
    ids = [set(split.train[C.ID]), set(split.valid[C.ID]), set(split.test[C.ID])]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert split.train[C.SIGNED_AT].max() <= split.valid[C.SIGNED_AT].min()
    assert split.valid[C.SIGNED_AT].max() <= split.test[C.SIGNED_AT].min()


def test_random_split(clean_df, session_cfg):
    ev = session_cfg.evaluation.model_copy(update={"split": "random"})
    s = split_data(clean_df, ev)
    assert sum(s.sizes().values()) == len(clean_df)


@pytest.mark.parametrize("kind", ["log1p", "zscore", "none"])
def test_target_transform_roundtrip(kind):
    y = np.array([1e3, 5e6, 2e9, 7e11])
    t = TargetTransform(kind).fit(y)
    assert np.allclose(t.inverse(t.transform(y)), y)


def test_zscore_inverse_sign_legacy_bug():
    """legacy main.py de-normalised with `x*ssd - mean` (wrong sign)."""
    t = TargetTransform("zscore", mean=163740447, std=1716771980)
    assert t.inverse(np.array([0.0]))[0] == pytest.approx(163740447)


def test_preprocessor_excludes_personal_data(clean_df, session_cfg):
    pre = build_preprocessor(session_cfg, clean_df)
    pre.fit(clean_df.head(500))
    names = " ".join(pre.get_feature_names_out())
    for forbidden in (
        "Documento Proveedor",
        "Proveedor Adjudicado",
        C.SUPPLIER,
        C.LEGAL_REP,
        "Representante",
    ):
        assert forbidden not in names


def test_model_learns_better_than_baseline(fitted, split):
    y = np.log1p(split.test[C.VALUE].to_numpy())
    model_mae = np.mean(np.abs(fitted.predict_log(split.test) - y))
    base = MedianBaseline([["Tipo de Contrato"]]).fit(split.train, np.log1p(split.train[C.VALUE]))
    base_mae = np.mean(np.abs(base.predict_log(split.test) - y))
    assert model_mae < base_mae * 0.9
    assert (fitted.predict(split.test) >= 0).all()


def test_predict_handles_unseen_categories_and_missing_columns(fitted, split):
    df = split.test.head(20).copy()
    df["Departamento"] = "Departamento Nuevo"
    df = df.drop(columns=["Ciudad", C.DURATION_DAYS])
    pred = fitted.predict(df)
    assert np.isfinite(pred).all() and len(pred) == 20


def test_save_load_roundtrip(fitted, split, tmp_path):
    fitted.save(tmp_path)
    loaded = PriceModel.load(tmp_path)
    assert np.allclose(loaded.predict(split.test), fitted.predict(split.test))
    assert (tmp_path / "booster.ubj").exists()  # native format, not a pickle


def test_top_features_are_named(fitted):
    top = fitted.top_features(5)
    assert len(top) == 5
    assert top["importance"].sum() == pytest.approx(1.0)  # normalised shares
    assert not top["feature"].str.fullmatch(r"f\d+").any()


def test_ridge_model(split, session_cfg, tmp_path):
    cfg = session_cfg.model_copy(
        update={"model": session_cfg.model.model_copy(update={"type": "ridge"})}
    )
    m = PriceModel(cfg).fit(split.train)
    m.save(tmp_path)
    assert np.allclose(PriceModel.load(tmp_path).predict(split.test), m.predict(split.test))


def test_median_baseline_fallback():
    df = pd.DataFrame({"t": ["a"] * 20 + ["b"] * 2})
    y = np.array([1.0] * 20 + [9.0] * 2)
    b = MedianBaseline([["t"]], min_support=10).fit(df, y)
    pred = b.predict_log(pd.DataFrame({"t": ["a", "b", "zzz"]}))
    assert pred[0] == 1.0
    assert pred[1] == b.global_ and pred[2] == b.global_  # group too small / unknown
