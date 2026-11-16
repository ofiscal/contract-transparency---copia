import pytest

from contract_transparency.data.cleaning import training_mask
from contract_transparency.mlops.registry import IntegrityError, ModelRegistry
from contract_transparency.models.price_model import PriceModel


@pytest.fixture(scope="module")
def small_model(clean_df, session_cfg):
    cfg = session_cfg.model_copy(
        update={"model": session_cfg.model.model_copy(update={"type": "ridge"})}
    )
    return PriceModel(cfg).fit(clean_df[training_mask(clean_df)].head(800))


def test_register_promote_load(tmp_path, small_model, clean_df):
    reg = ModelRegistry(tmp_path, "m")
    assert reg.versions() == [] and reg.champion() is None
    with pytest.raises(LookupError):
        reg.load("champion")
    v1 = reg.register(small_model, {"metrics": {"test": {"mae_log": 0.5}}})
    v2 = reg.register(small_model, {"metrics": {"test": {"mae_log": 0.4}}})
    assert (v1, v2) == ("v0001", "v0002")
    reg.promote(v2, reason="better")
    assert reg.champion() == "v0002"
    model, meta = reg.load("champion")
    assert meta["version"] == "v0002" and set(meta["artifacts"]) >= {
        "model.json",
        "preprocessor.json",
    }
    assert len(model.predict(clean_df.head(5))) == 5
    reg.promote(v1, reason="rollback")
    assert reg.champion() == "v0001"
    assert [e["event"] for e in reg.history()] == [
        "registered",
        "registered",
        "promoted",
        "promoted",
    ]
    assert reg.summary()["champion"].tolist() == [True, False]
    assert reg.metadata("latest")["version"] == "v0002"


def test_tampered_artifact_is_refused(tmp_path, small_model):
    reg = ModelRegistry(tmp_path, "m")
    v = reg.register(small_model, {})
    with open(reg.versions_dir / v / "preprocessor.npz", "ab") as fh:
        fh.write(b"malicious")
    with pytest.raises(IntegrityError):
        reg.load(v)


def test_unknown_version(tmp_path):
    with pytest.raises(LookupError, match="Unknown model version"):
        ModelRegistry(tmp_path, "m").metadata("v0099")
