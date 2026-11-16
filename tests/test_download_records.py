"""datos.gov.co downloader, official records and the live check, against a mock API."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import clean_contracts
from contract_transparency.data.download import (
    DownloadError,
    SocrataClient,
    client_from_config,
    download_contracts,
    needed_columns,
    socrata_field_name,
)
from contract_transparency.data.records import (
    id_variants,
    import_records,
    load_records,
    match_records,
    read_records_file,
)
from contract_transparency.data.synthetic import generate_contracts, generate_fiscal_bulletin
from tests.conftest import TEST_SALT, fast_download
from tests.mock_socrata import MockSocrata


@pytest.fixture(scope="module")
def contracts():
    return generate_contracts(3000, seed=3)


# --------------------------------------------------------------------------- download
def test_socrata_field_name():
    assert socrata_field_name("Duración del contrato") == "duraci_n_del_contrato"
    assert socrata_field_name("ID Contrato") == "id_contrato"
    assert socrata_field_name("Nit Entidad") == "nit_entidad"


def test_needed_columns_exclude_personal_data(cfg):
    cols = needed_columns(cfg)
    assert cfg.schema_.id in cols and cfg.schema_.target in cols and "URLProceso" in cols
    assert not set(cols) & set(cfg.privacy.drop_columns)


def test_download_pages_selects_and_flattens(cfg, contracts, tmp_path):
    cfg = fast_download(cfg)
    out = tmp_path / "dl.csv"
    with MockSocrata(contracts) as api:
        info = download_contracts(cfg, out, client=client_from_config(cfg, api.base_url))
        data_requests = [r for r in api.requests if r["path"].startswith("/resource")]
    assert info["rows"] == 3000 and len(data_requests) == 5  # 700-row pages
    assert info["field_names_from"] == "live metadata"
    # personal data is never requested from the API
    requested = set(data_requests[0]["params"]["$select"].split(","))
    pii_fields = {socrata_field_name(c) for c in cfg.privacy.drop_columns}
    assert not requested & pii_fields
    assert "justificacion_modalidad_de" in requested  # irregular name taken from metadata
    df = pd.read_csv(out, dtype=str)
    assert list(df.columns) == [c for c in needed_columns(cfg) if c in contracts.columns]
    assert (
        df["URLProceso"].str.startswith("https://community.secop.gov.co").all()
    )  # {"url": ...} flattened
    side = json.loads((tmp_path / "dl.csv.source.json").read_text())
    assert side["rows"] == 3000 and len(side["file_sha256"]) == 64 and side["order"] == ":id"


def test_downloaded_data_cleans_like_a_portal_export(cfg, contracts, tmp_path):
    with MockSocrata(contracts) as api:
        download_contracts(
            fast_download(cfg), tmp_path / "dl.csv", client=client_from_config(cfg, api.base_url)
        )
    via_api = clean_contracts(
        pd.read_csv(tmp_path / "dl.csv", dtype=str), cfg, TEST_SALT
    ).set_index(C.ID)
    via_csv = (
        clean_contracts(contracts.astype(str), cfg, TEST_SALT).set_index(C.ID).loc[via_api.index]
    )
    for col in (C.VALUE, C.SIGNED_AT, C.DURATION_DAYS, C.SUPPLIER, C.UNSPSC_CLASS, C.TEXT):
        pd.testing.assert_series_equal(
            via_api[col], via_csv[col], check_dtype=False, check_names=False
        )


def test_download_date_window_and_limit(cfg, contracts, tmp_path):
    with MockSocrata(contracts) as api:
        client = client_from_config(cfg, api.base_url)
        info = download_contracts(
            fast_download(cfg),
            tmp_path / "a.csv",
            since="2022-01-01",
            until="2022-07-01",
            client=client,
        )
        limited = download_contracts(
            fast_download(cfg), tmp_path / "b.csv", limit=1000, client=client
        )
    signed = pd.to_datetime(pd.read_csv(tmp_path / "a.csv")["Fecha de Firma"])
    assert signed.min() >= pd.Timestamp("2022-01-01") and signed.max() < pd.Timestamp("2022-07-01")
    expected = pd.to_datetime(contracts["Fecha de Firma"]).between("2022-01-01", "2022-06-30").sum()
    assert info["rows"] == expected and "fecha_de_firma >=" in info["where"]
    assert limited["rows"] == 1000


@pytest.mark.parametrize("code", [503, 429])
def test_retries_transient_errors(cfg, contracts, tmp_path, code):
    with MockSocrata(contracts.head(50), fail_first=2, fail_code=code) as api:
        info = download_contracts(
            fast_download(cfg),
            tmp_path / "r.csv",
            client=client_from_config(fast_download(cfg), api.base_url),
        )
    assert info["rows"] == 50


def test_permanent_errors_and_unreachable_api(contracts):
    with MockSocrata(contracts.head(5)) as api:
        client = SocrataClient(api.base_url, "jbjy-vk9h", backoff_s=0.01)
        with pytest.raises(DownloadError, match="HTTP 400"):
            next(client.iter_pages(["no_such_field"]))
    dead = SocrataClient(
        "http://127.0.0.1:9", "jbjy-vk9h", max_retries=1, backoff_s=0.01, timeout=2
    )
    with pytest.raises(DownloadError, match="Giving up"):
        dead.metadata()


def test_app_token_header(cfg, contracts, tmp_path, monkeypatch):
    monkeypatch.setenv("CT_SOCRATA_APP_TOKEN", "tok123")
    with MockSocrata(contracts.head(10)) as api:
        download_contracts(
            fast_download(cfg), tmp_path / "t.csv", client=client_from_config(cfg, api.base_url)
        )
        assert {r["token"] for r in api.requests} == {"tok123"}


# --------------------------------------------------------------------------- records
@pytest.fixture
def bulletin_file(contracts, tmp_path):
    path = tmp_path / "boletin.xlsx"
    generate_fiscal_bulletin(contracts, n=30, seed=1).to_excel(path, index=False, header=False)
    return path


def test_id_variants():
    assert id_variants("900.123.456-7") == ["9001234567", "900123456"]
    assert id_variants("52.123.456") == ["52123456"]
    assert id_variants(1234567.0) == ["1234567"]
    assert id_variants("N/A") == [] and id_variants(None) == []


def test_read_records_detects_header_below_titles(bulletin_file):
    df = read_records_file(bulletin_file)
    assert "NÚMERO DE DOCUMENTO" in df.columns and len(df) == 30


def test_import_records_keeps_only_pseudonyms(cfg, bulletin_file):
    rec, man = import_records(cfg, "contraloria_responsables_fiscales", bulletin_file, TEST_SALT)
    assert man["id_column"] == "NÚMERO DE DOCUMENTO" and man["date_column"] == "FECHA DE INCLUSIÓN"
    assert man["rows_with_valid_id"] == 30 and len(man["file_sha256"]) == 64
    assert man["publisher"] == "Contraloría General de la República"
    stored = pd.read_parquet(
        cfg.paths.processed_dir / "records" / "contraloria_responsables_fiscales.parquet"
    )
    assert list(stored.columns) == ["record_key", "record_date"]
    assert stored["record_key"].str.startswith("p_").all()
    text = stored.astype(str).to_csv()
    assert "Ficticia" not in text  # names discarded
    assert not re.search(r"(?<![\w-])\d{7,}(?![\w-])", text)  # no raw ID numbers


def test_records_match_suppliers_and_respect_dates(cfg, contracts, bulletin_file):
    import_records(cfg, "contraloria_responsables_fiscales", bulletin_file, TEST_SALT)
    clean = clean_contracts(contracts, cfg, TEST_SALT)
    flags = match_records(clean, cfg, load_records(cfg))["sig_supplier_in_fiscal_bulletin"]
    listed = read_records_file(bulletin_file)
    listed_ids = {v for x in listed["NÚMERO DE DOCUMENTO"] for v in id_variants(x)}
    raw = contracts.set_index("ID Contrato")
    raw_ids = raw["Documento Proveedor"]
    # a contract matches through its supplier OR its legal representative (match_on)
    supplier_listed = clean[C.ID].map(raw_ids).isin(listed_ids) | clean[C.ID].map(
        raw["Identificación Representante Legal"]
    ).isin(listed_ids)
    assert flags.any() and not flags[~supplier_listed.to_numpy()].any()  # no false matches
    # contracts signed before the listing date are not flagged
    first = {
        v: pd.to_datetime(d, dayfirst=True)
        for x, d in zip(listed["NÚMERO DE DOCUMENTO"], listed["FECHA DE INCLUSIÓN"], strict=True)
        for v in id_variants(x)
    }
    since = clean[C.ID].map(raw_ids).map(first)
    before = supplier_listed & (clean[C.SIGNED_AT] < since)
    assert before.any() and not flags[before.to_numpy()].any()


def test_records_csv_override_and_errors(cfg, tmp_path):
    p = tmp_path / "sanciones.csv"
    pd.DataFrame(
        {"Contratista": ["A", "B"], "Identificacion del contratista": ["999800001", "999800002"]}
    ).to_csv(p, sep=";", index=False)
    with pytest.raises(ValueError, match="Could not find the ID column"):
        import_records(cfg, "secop_multas_sanciones", p, TEST_SALT)
    src = cfg.records.sources["secop_multas_sanciones"].model_copy(
        update={"id_column": "Identificacion del contratista"}
    )
    cfg2 = cfg.model_copy(
        update={
            "records": cfg.records.model_copy(
                update={"sources": {**cfg.records.sources, "secop_multas_sanciones": src}}
            )
        }
    )
    rec, man = import_records(cfg2, "secop_multas_sanciones", p, TEST_SALT)
    assert len(rec) == 2 and man["date_column"] is None
    with pytest.raises(KeyError, match="Unknown record source"):
        import_records(cfg, "nope", p, TEST_SALT)


def test_legal_rep_matching(cfg):
    rec = pd.DataFrame({"record_key": ["p_rep"], "record_date": [pd.NaT]})
    df = pd.DataFrame(
        {
            C.SUPPLIER: ["p_a", "p_b"],
            C.LEGAL_REP: ["p_rep", "p_x"],
            C.SIGNED_AT: pd.to_datetime(["2022-01-01"] * 2),
        }
    )
    out = match_records(df, cfg, {"contraloria_responsables_fiscales": (rec, {})})
    assert out["sig_supplier_in_fiscal_bulletin"].tolist() == [True, False]


# --------------------------------------------------------------------------- live check
@pytest.mark.slow
def test_live_check_against_mock_api(cfg, tmp_path, bulletin_file):
    from contract_transparency.mlops.live_check import live_check

    data = generate_contracts(3000, seed=3, end="2023-06-30")
    with MockSocrata(data) as api:
        out = live_check(
            fast_download(cfg),
            tmp_path / "lc",
            since="2020-01-01",
            limit=3000,
            fiscal_bulletin=bulletin_file,
            verify_links=4,
            api_base=api.base_url,
        )
    status = {s["step"]: s["status"] for s in out["steps"]}
    assert out["ok"], out["steps"]
    assert status["download"] == status["prepare"] == status["audit_links"] == "PASS"
    assert status["records_import"] == "PASS" and status["score"] == "PASS"
    assert (tmp_path / "lc" / "report.md").exists() and (tmp_path / "lc" / "registry").exists()
    assert not (cfg.paths.registry_dir).exists()  # production registry untouched


def test_live_check_reports_unreachable_api(cfg, tmp_path):
    from contract_transparency.mlops.live_check import live_check

    c = cfg.model_copy(
        update={"download": cfg.download.model_copy(update={"max_retries": 0, "timeout_s": 2})}
    )
    out = live_check(c, tmp_path / "lc", api_base="http://127.0.0.1:9")
    metadata = next(s for s in out["steps"] if s["step"] == "api_metadata")
    assert not out["ok"] and metadata["status"] == "FAIL" and "HTTPS_PROXY" in metadata["hint"]


def test_live_check_detects_wrong_id_field(cfg, tmp_path):
    from contract_transparency.mlops.live_check import live_check

    bad = cfg.model_copy(
        update={"public_data": cfg.public_data.model_copy(update={"id_api_field": "wrong"})}
    )
    with MockSocrata(generate_contracts(300, seed=2)) as api:
        out = live_check(
            fast_download(bad), tmp_path / "lc", since="2020-01-01", api_base=api.base_url
        )
    step = next(s for s in out["steps"] if s["step"] == "id_api_field")
    assert step["status"] == "WARN" and step["live"] == "id_contrato"
    names = next(s for s in out["steps"] if s["step"] == "json_field_names")
    assert names["status"] == "PASS"  # default overrides match the (mock) live metadata


def test_live_check_reports_json_field_name_mismatches(cfg, tmp_path):
    from contract_transparency.mlops.live_check import live_check

    no_overrides = cfg.model_copy(
        update={"public_data": cfg.public_data.model_copy(update={"api_field_overrides": {}})}
    )
    with MockSocrata(generate_contracts(300, seed=2)) as api:
        out = live_check(
            fast_download(no_overrides), tmp_path / "lc", since="2020-01-01", api_base=api.base_url
        )
    step = next(s for s in out["steps"] if s["step"] == "json_field_names")
    assert step["status"] == "WARN"
    assert step["mismatches"] == {
        "Justificacion Modalidad de Contratacion": "justificacion_modalidad_de"
    }
    assert '"Justificacion Modalidad de Contratacion": "justificacion_modalidad_de"' in step["hint"]
    download = next(s for s in out["steps"] if s["step"] == "download")
    assert download["status"] == "WARN" and "earlier --since" in download["hint"]  # 300 rows
