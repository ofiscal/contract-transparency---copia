"""Public-data policy: audit links, notices and provenance manifests."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from contract_transparency.cli import app
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.mlops import pipeline
from contract_transparency.provenance import (
    file_sha256,
    open_data_url,
    valid_official_url,
)
from tests.conftest import make_config, write_cli_config
from tests.mock_socrata import MockSocrata

ALLOWED = ["secop.gov.co", "colombiacompra.gov.co"]


@pytest.mark.parametrize(
    ("url", "ok"),
    [
        (
            "https://community.secop.gov.co/Public/Tendering/OpportunityDetail/Index?noticeUID=CO1.NTC.1",
            True,
        ),
        ("https://www.colombiacompra.gov.co/tienda-virtual", True),
        ("http://community.secop.gov.co/x", True),
        ("https://secop.gov.co.evil.com/x", False),  # look-alike domain
        ("https://evilsecop.gov.co/x", False),
        ("ftp://community.secop.gov.co/x", False),
        ("javascript:alert(1)", False),
        ("", False),
        (None, False),
    ],
)
def test_valid_official_url(url, ok):
    assert (valid_official_url(url, ALLOWED) is not None) == ok


def test_open_data_url_escapes_id(cfg):
    url = open_data_url(cfg, "CO1.PCCNTR.1 2&3")
    assert (
        url == "https://www.datos.gov.co/resource/jbjy-vk9h.json?id_contrato=CO1.PCCNTR.1%202%263"
    )


@pytest.fixture
def trained_cfg(tmp_path, write_raw):
    cfg = make_config(tmp_path)
    raw = generate_contracts(2500, seed=21)
    raw.loc[:19, "URLProceso"] = None  # no link
    raw.loc[20:29, "URLProceso"] = "https://example.com/fake"  # not an official domain
    write_raw(raw)
    data, report = pipeline.prepare(cfg)
    pipeline.train(cfg, data)
    return cfg, data, report


@pytest.mark.slow
def test_outputs_are_auditable_and_carry_the_notice(trained_cfg):
    cfg, data, report = trained_cfg
    scored, path = pipeline.score(cfg, data=data, source=report["source"])
    # 30 rows cannot be verified at an official source -> not published, but counted
    assert len(scored) == len(data) - 30
    assert scored["secop_url"].str.startswith("https://community.secop.gov.co").all()
    assert scored["open_data_url"].str.contains("datos.gov.co/resource/jbjy-vk9h.json").all()
    assert scored["review_priority"].tolist() == list(range(1, len(scored) + 1))
    assert (scored["aviso"] == cfg.public_data.row_notice).all()
    assert scored[cfg.schema_.id].notna().all()

    manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text())
    assert manifest["rows"] == {
        **manifest["rows"],
        "input": len(data),
        "published": len(data) - 30,
        "excluded_not_auditable": 30,
    }
    assert (
        "No constituye prueba" in manifest["notice_es"] and "not evidence" in manifest["notice_en"]
    )
    assert manifest["synthetic_data"] is True
    src = manifest["sources"][0]
    assert src["file_sha256"] == file_sha256(cfg.paths.raw_data) and src["publisher"].startswith(
        "Agencia Nacional"
    )
    cols = manifest["columns"]
    assert (
        cols["contract_value"]["origin"] == "derived"
        and cols["Nombre Entidad"]["origin"] == "source"
    )
    assert cols["Descripcion del Proceso"]["origin"] == "source (redacted)"
    assert cols["secop_url"]["origin"] == "source (validated)"
    assert set(cols) == set(scored.columns)


@pytest.mark.slow
def test_excel_output_starts_with_notice_sheet(trained_cfg, tmp_path):
    cfg, data, report = trained_cfg
    _, path = pipeline.score(
        cfg, data=data, source=report["source"], output_path=tmp_path / "r.xlsx"
    )
    book = pd.ExcelFile(path)
    assert book.sheet_names == ["AVISO", "resultados", "diccionario"]
    notice = pd.read_excel(path, sheet_name="AVISO")["AVISO"].dropna().tolist()
    assert notice[0].startswith("DATOS SINTÉTICOS") and any(
        "No constituye prueba" in n for n in notice
    )


@pytest.mark.slow
def test_policy_can_be_relaxed_explicitly(trained_cfg):
    cfg, data, report = trained_cfg
    relaxed = cfg.model_copy(
        update={
            "public_data": cfg.public_data.model_copy(
                update={"require_secop_url": False, "add_row_notice": False}
            )
        }
    )
    scored, _ = pipeline.score(relaxed, data=data, write=False)
    assert len(scored) == len(data) and "aviso" not in scored.columns
    assert scored["secop_url"].isna().sum() == 30


# --------------------------------------------------------------------------- CLI
runner = CliRunner()


@pytest.mark.slow
def test_cli_download_records_and_live_check(tmp_path):
    cfg_path = write_cli_config(tmp_path)
    contracts = generate_contracts(2500, seed=8, end="2023-06-30")
    with MockSocrata(contracts) as api:
        res = runner.invoke(
            app,
            ["download", "-c", str(cfg_path), "--api-base", api.base_url, "--since", "2021-01-01"],
        )
        assert res.exit_code == 0, res.output
        assert (tmp_path / "raw.csv").exists() and (tmp_path / "raw.csv.source.json").exists()

        bulletin = tmp_path / "boletin.xlsx"
        from contract_transparency.data.synthetic import generate_fiscal_bulletin

        generate_fiscal_bulletin(contracts, n=20).to_excel(bulletin, index=False, header=False)
        res = runner.invoke(
            app,
            [
                "records",
                "import",
                "contraloria_responsables_fiscales",
                str(bulletin),
                "-c",
                str(cfg_path),
            ],
        )
        assert res.exit_code == 0 and '"rows_with_valid_id": 20' in res.output
        res = runner.invoke(app, ["records", "list", "-c", str(cfg_path)])
        assert (
            "imported" in res.output and "not imported" in res.output
        )  # second source not imported

        res = runner.invoke(app, ["run", "-c", str(cfg_path)])
        assert res.exit_code == 0 and "AVISO" in res.output and "DATOS SINTÉTICOS" in res.output
        manifest = json.loads(next((tmp_path / "outputs").glob("*.manifest.json")).read_text())
        assert manifest["sources"][0]["retrieval"]["method"] == "datos.gov.co Socrata API"
        assert manifest["sources"][1]["publisher"] == "Contraloría General de la República"

        res = runner.invoke(
            app,
            [
                "live-check",
                "-c",
                str(cfg_path),
                "--api-base",
                api.base_url,
                "--since",
                "2020-01-01",
                "--workdir",
                str(tmp_path / "lc"),
                "--verify-links",
                "3",
            ],
        )
        assert res.exit_code == 0, res.output
        assert "Overall: PASS" in res.output
    res = runner.invoke(
        app,
        [
            "live-check",
            "-c",
            str(cfg_path),
            "--api-base",
            "http://127.0.0.1:9",
            "--workdir",
            str(tmp_path / "lc2"),
        ],
    )
    assert res.exit_code == 1 and "Overall: FAIL" in res.output
