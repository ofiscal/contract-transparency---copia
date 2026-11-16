"""Optional integrations: MLflow tracking and sentence-transformer embeddings."""

import sys
import types

import numpy as np
import pytest

from contract_transparency.config import TrackingConfig
from contract_transparency.data.cleaning import training_mask
from contract_transparency.features.text import SentenceEmbedder
from contract_transparency.mlops.tracking import Tracker
from contract_transparency.models.price_model import PriceModel


def test_tracker_disabled_is_noop():
    t = Tracker(TrackingConfig(enabled=False))
    with t.run("x"):
        t.log_params({"a": {"b": 1}})
        t.log_metrics({"m": 1.0})
    assert not t.active


def test_tracker_logs_to_mlflow(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)  # MLflow writes artifacts to ./mlruns: keep the repo clean
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    t = Tracker(TrackingConfig(enabled=True, tracking_uri=uri, experiment="test-exp"))
    (tmp_path / "art").mkdir()
    (tmp_path / "art" / "a.txt").write_text("x")
    with t.run("unit", tags={"k": "v"}):
        t.log_params({"model": {"eta": 0.1}})
        t.log_metrics({"mae_log": 0.3, "not_a_number": "skip"}, prefix="test.")
        t.set_tags({"promoted": True})
        t.log_artifacts(tmp_path / "art")
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    assert runs.loc[0, "metrics.test.mae_log"] == pytest.approx(0.3)
    assert runs.loc[0, "params.model.eta"] == "0.1"
    assert runs.loc[0, "tags.promoted"] == "True"


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    class FakeST:
        loads = 0

        def __init__(self, name, device=None):
            FakeST.loads += 1

        def encode(self, texts, batch_size, show_progress_bar):
            return np.array([[len(t), t.count("a"), 1.0] for t in texts], dtype=np.float32)

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return FakeST


def test_sentence_embedder_is_lazy_and_not_pickled(fake_sentence_transformers):
    import pickle

    emb = SentenceEmbedder("fake-model")
    assert fake_sentence_transformers.loads == 0
    out = emb.fit_transform(np.array(["abc", "aaa"], dtype=object))
    assert out.shape == (2, 3) and fake_sentence_transformers.loads == 1
    assert emb.get_feature_names_out().tolist() == ["st_0", "st_1", "st_2"]
    restored = pickle.loads(pickle.dumps(emb))
    assert getattr(restored, "_st_model", None) is None  # network weights not serialised


def test_price_model_with_sentence_embeddings(
    fake_sentence_transformers, clean_df, session_cfg, tmp_path
):
    cfg = session_cfg.model_copy(
        update={
            "features": session_cfg.features.model_copy(
                update={"text_embedder": "sentence_transformer"}
            )
        }
    )
    df = clean_df[training_mask(clean_df)].head(600)
    m = PriceModel(cfg).fit(df.iloc[:500], df.iloc[500:])
    assert "text__st_0" in m.feature_names_  # real names, not the f0..fN fallback
    m.save(tmp_path)
    assert np.allclose(PriceModel.load(tmp_path).predict(df.iloc[500:]), m.predict(df.iloc[500:]))
