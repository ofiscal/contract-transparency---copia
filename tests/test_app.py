"""Smoke test of the Streamlit dashboard (headless)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import contract_transparency
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.mlops import pipeline
from tests.conftest import write_cli_config

# Located by path: importing the module would execute the Streamlit script.
APP_FILE = Path(contract_transparency.__file__).parent / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_dashboard_renders(tmp_path, monkeypatch):
    cfg_path = write_cli_config(tmp_path)
    generate_contracts(2500, seed=5).to_csv(tmp_path / "raw.csv", index=False)
    from contract_transparency.config import load_config

    cfg = load_config(cfg_path)
    data, _ = pipeline.prepare(cfg)
    pipeline.train(cfg, data)

    monkeypatch.setenv("CT_CONFIG", str(cfg_path))
    at = AppTest.from_file(str(APP_FILE), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = {m.label for m in at.metric}
    assert {"Contratos", "Desviaciones de precio"} <= labels
    assert [t.label for t in at.tabs] == [
        "Alertas",
        "Detalle de contrato",
        "Distribuciones",
        "Proveedores",
        "Modelos",
        "Fuentes",
    ]

    at.checkbox(key="only_alerts").check().run()
    assert not at.exception
    n_contracts = next(m.value for m in at.metric if m.label == "Contratos")
    n_alerts = next(m.value for m in at.metric if m.label == "Desviaciones de precio")
    assert n_contracts == n_alerts  # filter applied
    notice = " ".join(i.value for i in at.info).lower()
    assert "no constituye prueba" in notice and "enlaza a su fuente oficial" in notice
    assert "ni asumen responsabilidad" in notice and "riesgo" not in notice
    assert any("DATOS SINTÉTICOS" in e.value for e in at.error)  # demo data is watermarked
    table = at.dataframe[0].value
    assert table["secop_url"].str.startswith("https://community.secop.gov.co").all()
