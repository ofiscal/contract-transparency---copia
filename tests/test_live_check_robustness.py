"""Real-world failure modes of the datos.gov.co API and of client machines."""

from __future__ import annotations

import ssl
import urllib.error

import pytest
from typer.testing import CliRunner

from contract_transparency.cli import app
from contract_transparency.data import download as dl
from contract_transparency.data.synthetic import generate_contracts
from contract_transparency.mlops.live_check import live_check
from tests.conftest import fast_download, write_cli_config
from tests.mock_socrata import MockSocrata

runner = CliRunner()


def _status(out, step):
    return next(s for s in out["steps"] if s["step"] == step)


@pytest.fixture(scope="module")
def contracts():
    return generate_contracts(2500, seed=12, end="2023-06-30")


def test_dates_stored_as_text_fall_back_to_updated_at(cfg, contracts, tmp_path):
    with MockSocrata(contracts, date_type="text") as api:
        info = dl.download_contracts(
            fast_download(cfg, 900),
            tmp_path / "a.csv",
            since="2022-01-01",
            until="2022-07-01",
            client=dl.client_from_config(fast_download(cfg, 900), api.base_url),
        )
    assert info["date_filter"] == "updated_at" and ":updated_at >=" in info["where"]
    assert info["rows"] > 300


def test_unknown_column_types_retry_with_updated_at_when_empty(
    cfg, contracts, tmp_path, monkeypatch
):
    # metadata without types + text dates: the signing-date filter returns nothing -> fallback
    monkeypatch.setattr(
        dl.SocrataClient,
        "columns_info",
        lambda self: {k: {"field": v, "type": ""} for k, v in self.field_map().items()},
    )
    with MockSocrata(contracts, date_type="text") as api:
        info = dl.download_contracts(
            fast_download(cfg, 900),
            tmp_path / "b.csv",
            since="2022-01-01",
            until="2022-07-01",
            client=dl.client_from_config(fast_download(cfg, 900), api.base_url),
        )
    assert info["rows"] > 300 and info["date_filter"] == "updated_at"


@pytest.mark.slow
def test_live_check_survives_blocked_metadata_text_dates_and_duplicates(cfg, contracts, tmp_path):
    with MockSocrata(
        contracts, metadata_status=403, date_type="text", duplicate_ids=len(contracts)
    ) as api:
        out = live_check(
            fast_download(cfg, 900),
            tmp_path / "lc",
            since="2020-01-01",
            limit=6000,
            verify_links=4,
            api_base=api.base_url,
        )
    assert out["ok"], [s for s in out["steps"] if s["status"] == "FAIL"]
    assert _status(out, "api_metadata")["status"] == "WARN"
    assert _status(out, "download")["date_filter"] == "updated_at"
    links = _status(out, "audit_links")
    assert links["status"] == "WARN" and links["duplicated_ids"] and not links["broken"]
    assert _status(out, "environment")["python"]


def test_ssl_errors_fail_fast_with_a_hint(monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError(
            ssl.SSLCertVerificationError(
                "certificate verify failed: unable to get local issuer certificate"
            )
        )

    monkeypatch.setattr(dl.urllib.request, "urlopen", boom)
    client = dl.SocrataClient(
        "https://www.datos.gov.co", "jbjy-vk9h", max_retries=5, backoff_s=0.01
    )
    with pytest.raises(dl.DownloadError, match="SSL_CERT_FILE"):
        client.metadata()
    assert len(calls) == 1  # no pointless retries


def test_ssl_context_prefers_explicit_bundle_then_os_store(monkeypatch, tmp_path):
    import certifi

    for var in ("CT_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        monkeypatch.delenv(var, raising=False)
    ctx = dl.ssl_context()
    assert type(ctx).__module__.startswith("truststore")  # OS certificate store
    monkeypatch.setenv("CT_CA_BUNDLE", certifi.where())
    assert isinstance(dl.ssl_context(), ssl.SSLContext) and not type(
        dl.ssl_context()
    ).__module__.startswith("truststore")


def test_cli_prints_failures_and_hints(tmp_path):
    res = runner.invoke(
        app,
        [
            "live-check",
            "-c",
            str(write_cli_config(tmp_path)),
            "--api-base",
            "http://127.0.0.1:9",
            "--workdir",
            str(tmp_path / "lc"),
        ],
    )
    assert res.exit_code == 1
    assert "[FAIL] api_metadata" in res.output and "error:" in res.output and "->" in res.output
    assert "HTTPS_PROXY" in res.output and "report.md" in res.output
