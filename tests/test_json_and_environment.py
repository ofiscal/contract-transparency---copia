"""JSON saved from the public API URL, `ct doctor` and version-compatibility handling."""

from __future__ import annotations

import json
import urllib.request

import pandas as pd
import pytest
from typer.testing import CliRunner

from contract_transparency.cli import app, main
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import clean_contracts
from contract_transparency.data.io import read_contracts
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.environment import (
    ModelCompatibilityError,
    declared_minimums,
    doctor,
    model_compatibility,
    version_tuple,
)
from contract_transparency.mlops import pipeline
from tests.conftest import TEST_SALT, make_config, write_cli_config
from tests.mock_socrata import MockSocrata

runner = CliRunner()


# --------------------------------------------------------------------------- JSON from the URL
@pytest.fixture(scope="module")
def api_json(tmp_path_factory):
    """Exactly what a browser saves from .../resource/jbjy-vk9h.json?$limit=10000
    (no $select -> ALL columns, including personal data)."""
    contracts = generate_contracts(1500, seed=4)
    with MockSocrata(contracts) as api:
        url = f"{api.base_url}/resource/jbjy-vk9h.json?$limit=10000"
        raw = json.loads(urllib.request.urlopen(url).read())
    path = tmp_path_factory.mktemp("json") / "jbjy-vk9h.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, contracts, raw


def test_raw_api_json_contains_personal_data(api_json):
    _, _, raw = api_json
    assert (
        "n_mero_de_cuenta" in raw[0] and "nombre_supervisor" in raw[0]
    )  # why the allowlist matters


def test_json_is_read_with_an_allowlist(cfg, api_json):
    path, contracts, _ = api_json
    df = read_contracts(path, cfg)
    assert len(df) == 1500
    assert not set(df.columns) & set(cfg.privacy.drop_columns)  # no bank / supervisor data
    assert {
        "ID Contrato",
        "Valor del Contrato",
        "Justificacion Modalidad de Contratacion",
        "URLProceso",
    } <= set(df.columns)
    assert df["URLProceso"].str.startswith("https://").all()  # {"url": ...} flattened
    text = df.to_csv()
    assert "000-DEMO" not in text and "Supervisor Ficticio" not in text


def test_json_route_equals_csv_route(cfg, api_json):
    path, contracts, _ = api_json
    via_json = clean_contracts(read_contracts(path, cfg), cfg, TEST_SALT).set_index(C.ID)
    via_csv = (
        clean_contracts(contracts.astype(str), cfg, TEST_SALT).set_index(C.ID).loc[via_json.index]
    )
    for col in (
        C.VALUE,
        C.SIGNED_AT,
        C.DURATION_DAYS,
        C.SUPPLIER,
        C.LEGAL_REP,
        C.UNSPSC_CLASS,
        C.TEXT,
        "Modalidad de Contratacion",
    ):
        pd.testing.assert_series_equal(
            via_json[col], via_csv[col], check_dtype=False, check_names=False
        )


def test_json_missing_fields_are_reported(cfg, tmp_path, caplog):
    p = tmp_path / "x.json"
    p.write_text(
        json.dumps([{"id_contrato": "A", "valor_del_contrato": "100", "campo_nuevo": "z"}])
    )
    df = read_contracts(p, cfg)
    assert list(df.columns) == ["ID Contrato", "Valor del Contrato"]
    assert "api_field_overrides" in caplog.text
    p.write_text(json.dumps({"not": "a list"}))
    assert len(read_contracts(p, cfg)) == 1  # single record object tolerated
    p.write_text(json.dumps("nonsense"))
    with pytest.raises(ValueError, match="JSON array"):
        read_contracts(p, cfg)


# --------------------------------------------------------------------------- versions
def test_version_tuple():
    assert version_tuple("1.9.1") == (1, 9, 1)
    assert version_tuple("2.0.0rc1") == (2, 0, 0)
    assert version_tuple(None) == ()


def test_declared_minimums_match_pyproject():
    mins = declared_minimums()
    assert mins["pandas"] == "2.1" and mins["xgboost"] == "2.0.3" and mins["typer"] == "0.15.4"


def test_model_compatibility_rules():
    import sklearn
    import xgboost

    same = {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
    assert model_compatibility(same) == []
    old_sklearn = {"scikit-learn": "0.1.0", "xgboost": xgboost.__version__}
    assert model_compatibility(old_sklearn, portable=True) == []  # portable: sklearn irrelevant
    assert "scikit-learn" in model_compatibility(old_sklearn, portable=False)[0]
    assert model_compatibility({"sklearn": "0.1.0"}, portable=False)  # v1.0 metadata key
    newer_xgb = {"xgboost": "999.0.0"}
    assert "older" in model_compatibility(newer_xgb, portable=True)[0]


# --------------------------------------------------------------------------- doctor & errors
@pytest.fixture
def trained(tmp_path, write_raw):
    cfg = make_config(tmp_path)
    write_raw(generate_contracts(2500, seed=30))
    data, _ = pipeline.prepare(cfg)
    pipeline.train(cfg, data)
    return cfg


@pytest.mark.slow
def test_doctor_reports_a_healthy_environment(trained):
    rep = doctor(trained)
    status = {c.name: c.status for c in rep.checks}
    assert rep.ok and status["champion loads & predicts"] == "PASS"
    assert status["champion trained with"] == "PASS" and status["package pandas"] == "PASS"


@pytest.mark.slow
def test_doctor_and_score_on_incompatible_legacy_model(trained, monkeypatch):
    reg = pipeline.get_registry(trained)
    version = reg.champion()
    vdir = reg.versions_dir / version
    # Turn the champion into a "v1.0" model trained with an ancient scikit-learn.
    for f in vdir.rglob("preprocessor.json"):  # single or segmented model
        f.rename(f.with_suffix(".json.bak"))
    meta = json.loads((vdir / "metadata.json").read_text())
    meta["libraries"]["scikit-learn"] = "0.24.0"
    meta["artifacts"] = {}
    (vdir / "metadata.json").write_text(json.dumps(meta))
    rep = doctor(trained)
    checks = {c.name: c for c in rep.checks}
    assert not rep.ok
    assert (
        checks["champion trained with"].status == "WARN"
        and "ct train" in checks["champion trained with"].hint
    )
    assert checks["champion loads & predicts"].status == "FAIL"
    with pytest.raises(ModelCompatibilityError, match="ct train"):
        reg.load("champion")


def test_cli_doctor_command(tmp_path):
    res = runner.invoke(app, ["doctor", "-c", str(write_cli_config(tmp_path))])
    assert res.exit_code == 0 and "Overall: OK" in res.output and "package xgboost" in res.output


def test_console_entry_point_prints_friendly_errors(tmp_path, monkeypatch, capsys):
    cfg = write_cli_config(tmp_path)
    monkeypatch.setattr("sys.argv", ["ct", "score", "-c", str(cfg)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "CT_DEBUG" in err and "Traceback" not in err
    monkeypatch.setenv("CT_DEBUG", "1")
    with pytest.raises(FileNotFoundError):
        main()
