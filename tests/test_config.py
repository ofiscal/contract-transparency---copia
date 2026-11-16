import pytest
import yaml
from pydantic import ValidationError

from contract_transparency.config import Settings, load_config
from tests.conftest import REPO


def test_default_configs_are_valid():
    for name in ("secop_ii.yaml", "demo.yaml"):
        cfg = load_config(REPO / "configs" / name)
        assert cfg.schema_.target == "Valor del Contrato"
        assert cfg.model.target_transform == "log1p"


def test_extends_merges_nested_keys():
    demo = load_config(REPO / "configs" / "demo.yaml")
    base = load_config(REPO / "configs" / "secop_ii.yaml")
    assert demo.evaluation.label_column == "synthetic_is_anomaly"
    assert demo.model.num_boost_round == 600 and demo.public_data.synthetic
    # untouched nested keys come from the parent
    assert demo.features.text_max_features == base.features.text_max_features
    assert demo.privacy.drop_columns == base.privacy.drop_columns


def test_unknown_keys_fail(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"model": {"typo_param": 1}}))
    with pytest.raises(ValidationError):
        load_config(p)


def test_pii_cannot_be_used_as_feature():
    with pytest.raises(ValidationError, match="PII columns configured as model features"):
        Settings.model_validate(
            {
                "schema": {"categorical": ["Documento Proveedor"]},
                "privacy": {"pseudonymize_columns": ["Documento Proveedor"]},
            }
        )


def test_dropped_pii_cannot_be_output():
    with pytest.raises(ValidationError, match="output columns"):
        Settings.model_validate(
            {
                "privacy": {"drop_columns": ["Número de cuenta"]},
                "output": {"passthrough_columns": ["Número de cuenta"]},
            }
        )


def test_dotted_overrides_and_fingerprint():
    a = load_config(REPO / "configs" / "secop_ii.yaml")
    b = load_config(REPO / "configs" / "secop_ii.yaml", overrides={"model.num_boost_round": 10})
    assert b.model.num_boost_round == 10
    assert a.fingerprint() == load_config(REPO / "configs" / "secop_ii.yaml").fingerprint()
    assert a.fingerprint() != b.fingerprint()


def test_split_fraction_validation():
    with pytest.raises(ValidationError):
        Settings.model_validate({"evaluation": {"test_size": 0.7}})


def test_elections_typed():
    cfg = load_config(REPO / "configs" / "secop_ii.yaml")
    kinds = {e.type for e in cfg.rules.elections}
    assert kinds == {"presidential", "congress", "territorial"}
    assert all(e.second_round for e in cfg.rules.elections if e.type == "presidential")
