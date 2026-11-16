"""The pickle-free preprocessor must reproduce scikit-learn exactly."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from contract_transparency.config import with_overrides
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import training_mask
from contract_transparency.features.builder import build_preprocessor, ensure_columns
from contract_transparency.features.portable import NPZ_FILE, PortablePreprocessor
from contract_transparency.models.price_model import PriceModel


def _cfg(session_cfg, **features):
    return session_cfg.model_copy(
        update={"features": session_cfg.features.model_copy(update=features)}
    )


@pytest.fixture(scope="module")
def data(clean_df):
    df = clean_df[training_mask(clean_df)].reset_index(drop=True)
    return df.iloc[:2000], df.iloc[2000:]


def _hard_cases(df):
    """Unseen categories, missing values and empty text."""
    x = df.head(60).copy()
    x.loc[x.index[:10], "Departamento"] = "Departamento Nunca Visto"
    x.loc[x.index[10:20], "Tipo de Contrato"] = None
    x.loc[x.index[20:30], C.DURATION_DAYS] = np.nan
    x.loc[x.index[30:40], C.SIGNED_AT] = None
    x.loc[x.index[40:50], C.TEXT] = ""
    x.loc[x.index[50:55], C.UNSPSC_CLASS] = None
    return x


@pytest.mark.parametrize(
    "features",
    [
        {},  # default: one-hot with infrequent grouping + TF-IDF/SVD + numeric
        {"text_embedder": "tfidf"},  # sparse TF-IDF without SVD
        {"min_category_frequency": 1, "max_categories": None},  # no infrequent grouping
        {"text_embedder": "none"},
    ],
)
def test_portable_equals_sklearn(session_cfg, data, features):
    train, test = data
    cfg = _cfg(session_cfg, **features)
    fitted = build_preprocessor(cfg, train).fit(train)
    portable = PortablePreprocessor.from_sklearn(fitted)
    for df in (train, test, _hard_cases(test)):
        expected = fitted.transform(df)
        got = portable.transform(df)
        expected = expected.toarray() if hasattr(expected, "toarray") else np.asarray(expected)
        assert got.shape == expected.shape
        # Same transformation; differences only from sklearn's float32 arithmetic
        # and the 6-decimal rounding of text features (see portable.TEXT_DECIMALS).
        np.testing.assert_allclose(got.toarray(), expected, rtol=0, atol=2e-6)


def test_portable_files_contain_no_pickle(session_cfg, data, tmp_path):
    train, test = data
    fitted = build_preprocessor(session_cfg, train).fit(train)
    portable = PortablePreprocessor.from_sklearn(fitted)
    portable.save(tmp_path)
    with np.load(tmp_path / NPZ_FILE, allow_pickle=False) as npz:  # raises if pickled objects
        assert all(npz[k].dtype != object for k in npz.files)
    json.loads((tmp_path / "preprocessor.json").read_text())
    again = PortablePreprocessor.load(tmp_path)
    np.testing.assert_array_equal(
        again.transform(test).toarray(), portable.transform(test).toarray()
    )


def test_newer_format_is_refused(tmp_path, session_cfg, data):
    train, _ = data
    PortablePreprocessor.from_sklearn(build_preprocessor(session_cfg, train).fit(train)).save(
        tmp_path
    )
    spec = json.loads((tmp_path / "preprocessor.json").read_text())
    spec["format_version"] = 99
    (tmp_path / "preprocessor.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="newer than this code"):
        PortablePreprocessor.load(tmp_path)


def test_model_artifacts_have_no_pickles(session_cfg, data, tmp_path):
    train, test = data
    model = PriceModel(session_cfg).fit(train.iloc[:1500], train.iloc[1500:])
    files = {p.name for p in model.save(tmp_path)}
    assert {
        "preprocessor.json",
        "preprocessor.npz",
        "booster.ubj",
        "model.json",
        "text_model.json",
        "text_model.npz",
    } <= files
    assert not {f for f in files if f.endswith((".joblib", ".pkl", ".pickle"))}
    assert np.allclose(PriceModel.load(tmp_path).predict(test), model.predict(test))


def test_v1_models_with_pickled_preprocessor_still_load(session_cfg, data, tmp_path, monkeypatch):
    """Registries written by v1.0 stored a pickled sklearn preprocessor and a
    booster trained on its output. They must keep loading (same sklearn version)."""
    train, test = data
    # Build a genuine v1.0 model: v1.0 settings (no text model, no blend) and a
    # booster trained on sklearn's own (pickled) preprocessor output.
    v1_cfg = with_overrides(
        session_cfg,
        {
            "features.text_price_model": False,
            "features.text_embedder": "tfidf_svd",
            "features.add_text_quantity": False,
            "model.blend_linear_weight": 0.0,
        },
    )
    monkeypatch.setattr(PortablePreprocessor, "from_sklearn", staticmethod(lambda ct: ct))
    v1 = PriceModel(v1_cfg).fit(train.iloc[:1500], train.iloc[1500:])
    monkeypatch.undo()
    joblib.dump(v1.preprocessor, tmp_path / "preprocessor.joblib")
    v1.estimator.save_model(tmp_path / "booster.ubj")
    (tmp_path / "model.json").write_text(
        json.dumps(
            {
                "model_type": "xgboost",
                "target": v1.target.to_dict(),
                "input_columns": v1.input_columns_,
                "feature_names": v1.feature_names_,
                "best_iteration": v1.best_iteration_,
                "config": v1_cfg.model_dump(mode="json", by_alias=True),
            }
        )
    )
    loaded = PriceModel.load(tmp_path)
    assert not isinstance(loaded.preprocessor, PortablePreprocessor)
    np.testing.assert_allclose(loaded.predict(test), v1.predict(test))


def test_model_config_is_loaded_tolerantly(session_cfg, data, tmp_path):
    train, test = data
    model = PriceModel(session_cfg).fit(train.iloc[:1500], train.iloc[1500:])
    model.save(tmp_path)
    meta = json.loads((tmp_path / "model.json").read_text())
    meta["config"]["a_setting_removed_in_the_future"] = 1
    meta["config"]["features"]["renamed_option"] = True
    (tmp_path / "model.json").write_text(json.dumps(meta))
    assert np.allclose(PriceModel.load(tmp_path).predict(test), model.predict(test))


def test_ensure_columns_still_applies(session_cfg, data):
    train, test = data
    fitted = build_preprocessor(session_cfg, train).fit(train)
    portable = PortablePreprocessor.from_sklearn(fitted)
    df = ensure_columns(test.drop(columns=["Ciudad"]).head(5), ["Ciudad"])
    assert portable.transform(df).shape[0] == 5
