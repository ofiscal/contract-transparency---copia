"""Download SECOP II contracts from the datos.gov.co (Socrata) open-data API.

* Column names are resolved from the dataset's **live metadata**
  (``/api/views/<id>.json``: display name -> API ``fieldName``), so the CSV
  written here has exactly the same columns as a manual "Export -> CSV" and the
  rest of the pipeline does not change.
* Only the columns the pipeline needs are requested (``$select``); personal
  data listed in ``privacy.drop_columns`` (bank accounts, supervisors, ...)
  is **never downloaded**.
* Stable paging (``$order=:id``), retries with exponential backoff on 429/5xx,
  optional app token (``CT_SOCRATA_APP_TOKEN``) for higher rate limits.
* A ``<file>.source.json`` sidecar records the exact query, time and row count,
  so every result can be traced back to the public source.

Uses only the standard library (``urllib``).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import ssl
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from contract_transparency.config import Settings

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    pass


def socrata_field_name(display_name: str) -> str:
    """Approximation of Socrata's display-name -> fieldName rule (fallback only;
    the live metadata is always preferred): 'Duración del contrato' ->
    'duraci_n_del_contrato'."""
    text = display_name.strip().lower()
    text = "".join(ch if ord(ch) < 128 else "_" for ch in unicodedata.normalize("NFC", text))
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", text)).strip("_")


def needed_columns(cfg: Settings) -> list[str]:
    """Display names the pipeline uses (schema roles, features, outputs,
    pseudonymised ids) minus the personal data that must never be downloaded."""
    s = cfg.schema_
    cols = [
        s.id,
        s.target,
        *s.text,
        *s.categorical,
        s.unspsc,
        s.duration,
        s.signed_date,
        s.start_date,
        s.end_date,
        s.entity_name,
        s.entity_id,
        s.supplier_id,
        s.supplier_name,
        s.supplier_doc_type,
        s.legal_rep_id,
        s.modality,
        s.modality_justification,
        s.entity_order,
        s.contract_type,
        s.days_added,
        s.n_bidders,
        s.url,
        *cfg.output.passthrough_columns,
    ]
    drop = set(cfg.privacy.drop_columns)
    seen: list[str] = []
    for c in cols:
        if c and c not in drop and c not in seen:
            seen.append(c)
    return seen


def api_field_map(cfg: Settings) -> dict[str, str]:
    """Display name -> expected API field name for the columns the pipeline uses
    (config overrides first, then the standard Socrata transformation)."""
    overrides = cfg.public_data.api_field_overrides
    return {c: overrides.get(c) or socrata_field_name(c) for c in needed_columns(cfg)}


def frame_from_api_records(records: list[dict[str, Any]], cfg: Settings) -> pd.DataFrame:
    """Convert raw API records (e.g. JSON saved from
    ``https://www.datos.gov.co/resource/jbjy-vk9h.json?$limit=10000``) into the
    column layout of a portal CSV export.

    Privacy by allowlist: only the fields the pipeline needs are kept. A raw
    API URL without ``$select`` returns every column (bank accounts,
    supervisors' IDs, ...); all of them are dropped here, known or not.
    """
    by_field = {field: display for display, field in api_field_map(cfg).items()}
    # Values as strings, missing -> NaN: identical to reading a portal CSV with dtype=str.
    rows = [
        {by_field[k]: str(_flatten(v)) for k, v in rec.items() if k in by_field and v is not None}
        for rec in records
    ]
    df = pd.DataFrame(rows, columns=list(by_field.values()), dtype=object)
    present = set().union(*(rec.keys() for rec in records)) if records else set()
    dropped = sorted(present - set(by_field))
    if dropped:
        logger.info("Ignored %d API fields not used by the pipeline", len(dropped))
    missing = [d for f, d in by_field.items() if f not in present]
    if missing:
        logger.warning(
            "Fields not found in the API records (set public_data.api_field_overrides "
            "or check `ct live-check`): %s",
            missing,
        )
    return df.dropna(axis=1, how="all")


def read_api_json(path: str | Path, cfg: Settings) -> pd.DataFrame:
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    if isinstance(records, dict):  # {"data": [...]} or a single record
        records = records.get("data", [records])
    if not isinstance(records, list):
        raise ValueError(f"{path}: expected a JSON array of records (Socrata /resource/ output)")
    return frame_from_api_records(records, cfg)


TIMESTAMP_TYPES = {"floating_timestamp", "calendar_date", "date", "fixed_timestamp"}
SSL_HINT = (
    "TLS certificate verification failed. Typical causes: a corporate proxy that "
    "inspects HTTPS, or a Python without access to the system certificates. Fixes: "
    "set SSL_CERT_FILE=/path/to/ca-bundle.pem (your organisation's CA bundle), "
    "or make sure the 'truststore' package is installed (`uv sync`)."
)


def ssl_context() -> ssl.SSLContext:
    """Certificate store for HTTPS requests, in order of preference:

    1. an explicit bundle: SSL_CERT_FILE / REQUESTS_CA_BUNDLE / CT_CA_BUNDLE;
    2. the operating-system store via ``truststore`` (works with corporate proxies
       whose CA is installed in the OS, and with uv-managed Pythons on macOS/Windows);
    3. ``certifi``; 4. Python's default.
    """
    for var in ("CT_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        path = os.environ.get(var)
        if path and os.path.exists(path):
            return ssl.create_default_context(cafile=path)
    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:  # optional dependency / unsupported platform
        pass
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # certifi missing: fall back to Python's default store
        return ssl.create_default_context()


def _is_ssl_error(exc: BaseException) -> bool:
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(exc)


class SocrataClient:
    def __init__(
        self,
        base_url: str,
        dataset_id: str,
        app_token: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 5,
        backoff_s: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.app_token = app_token
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_s = backoff_s

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {"Accept": "application/json", "User-Agent": "contract-transparency"}
        if self.app_token:
            headers["X-App-Token"] = self.app_token
        last: Exception | None = None
        context = ssl_context() if url.startswith("https") else None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout, context=context) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in {429, 500, 502, 503, 504}:
                    detail = exc.read().decode("utf-8", "replace")[:300] if exc.fp else ""
                    raise DownloadError(f"HTTP {exc.code} for {url} {detail}".strip()) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError) as exc:
                if _is_ssl_error(exc):  # retrying will not help
                    raise DownloadError(f"{exc} -- {SSL_HINT}") from exc
                last = exc
            if attempt < self.max_retries:
                wait = self.backoff_s * 2**attempt
                logger.warning("Request failed (%s); retrying in %.1fs", last, wait)
                time.sleep(wait)
        raise DownloadError(f"Giving up on {url}: {last}")

    # ---------------------------------------------------------------- metadata
    def metadata(self) -> dict[str, Any]:
        return self._get(f"/api/views/{self.dataset_id}.json")

    def field_map(self) -> dict[str, str]:
        """Display name -> API fieldName (from live metadata)."""
        cols = self.metadata().get("columns", [])
        return {c["name"].strip(): c["fieldName"] for c in cols if "fieldName" in c}

    def columns_info(self) -> dict[str, dict[str, str]]:
        """Display name -> {"field": fieldName, "type": dataTypeName}."""
        cols = self.metadata().get("columns", [])
        return {
            c["name"].strip(): {"field": c["fieldName"], "type": c.get("dataTypeName", "")}
            for c in cols
            if "fieldName" in c
        }

    def sample_fields(self) -> set[str]:
        """Field names present in one record (fallback when metadata is unavailable)."""
        rows = self._get(f"/resource/{self.dataset_id}.json", {"$limit": 1})
        return set(rows[0]) if rows else set()

    # ------------------------------------------------------------------ rows
    def iter_pages(
        self,
        select: list[str],
        where: str | None = None,
        page_size: int = 50_000,
        limit: int | None = None,
        order: str = ":id",
    ) -> Iterator[list[dict[str, Any]]]:
        offset = 0
        while True:
            size = page_size if limit is None else min(page_size, limit - offset)
            if size <= 0:
                return
            params = {
                "$select": ",".join(select),
                "$order": order,
                "$limit": size,
                "$offset": offset,
            }
            if where:
                params["$where"] = where
            page = self._get(f"/resource/{self.dataset_id}.json", params)
            if not page:
                return
            yield page
            offset += len(page)
            if len(page) < size:
                return

    def record(self, field: str, value: str) -> list[dict[str, Any]]:
        """Simple-filter lookup of one record (used to verify audit links)."""
        return self._get(f"/resource/{self.dataset_id}.json", {field: value})


def _flatten(value: Any) -> Any:
    # URL columns come back as {"url": "...", "description": "..."}
    if isinstance(value, dict):
        return value.get("url") or value.get("description") or json.dumps(value, ensure_ascii=False)
    return value


def _date_where(
    cfg: Settings,
    fields: dict[str, str],
    types: dict[str, str],
    since: str | None,
    until: str | None,
    force_system: bool = False,
) -> tuple[str | None, str | None]:
    """SoQL date filter. Uses the signing date when it is a timestamp column;
    otherwise (text column, or forced) the system field ``:updated_at``."""
    if not since and not until:
        return None, None
    signed = cfg.schema_.signed_date or ""
    col_type = types.get(signed)
    use_signed = (
        not force_system and signed in fields and (col_type is None or col_type in TIMESTAMP_TYPES)
    )
    if not use_signed and col_type and col_type not in TIMESTAMP_TYPES:
        logger.warning(
            "'%s' is stored as %s in the API; filtering by :updated_at", signed, col_type
        )
    field = fields[signed] if use_signed else ":updated_at"
    clauses = []
    if since:
        clauses.append(f"{field} >= '{since}T00:00:00'")
    if until:
        clauses.append(f"{field} < '{until}T00:00:00'")
    return " AND ".join(clauses), ("signed_date" if use_signed else "updated_at")


def client_from_config(cfg: Settings, base_url: str | None = None) -> SocrataClient:
    return SocrataClient(
        base_url or cfg.public_data.api_base,
        cfg.public_data.dataset_id,
        app_token=os.environ.get(cfg.download.app_token_env_var) or None,
        timeout=cfg.download.timeout_s,
        max_retries=cfg.download.max_retries,
        backoff_s=cfg.download.backoff_s,
    )


def download_contracts(
    cfg: Settings,
    output: str | Path,
    since: str | None = None,
    until: str | None = None,
    limit: int | None = None,
    client: SocrataClient | None = None,
) -> dict[str, Any]:
    """Download contracts signed in [since, until) to ``output`` (CSV).

    Returns the provenance info (also written to ``<output>.source.json``).
    """
    from contract_transparency.provenance import file_sha256, write_source_sidecar

    client = client or client_from_config(cfg)
    types: dict[str, str] = {}
    try:
        info_cols = client.columns_info()
        fmap = {k: v["field"] for k, v in info_cols.items()}
        types = {k: v["type"] for k, v in info_cols.items()}
        resolved_from = "live metadata"
    except DownloadError as exc:
        # e.g. the metadata endpoint is blocked: use names present in a sample record
        logger.warning("Metadata unavailable (%s); using field names from a sample record", exc)
        present_fields = client.sample_fields()
        fmap = {d: f for d, f in api_field_map(cfg).items() if f in present_fields}
        resolved_from = "sample record + derived names (metadata unavailable)"

    wanted = needed_columns(cfg)
    present = [c for c in wanted if c in fmap]
    missing = [c for c in wanted if c not in fmap]
    if missing:
        logger.warning("Columns not in the live dataset (skipped): %s", missing)
    fields = {c: fmap[c] for c in present}

    where, date_filter = _date_where(cfg, fields, types, since, until)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    def write(where_clause: str | None) -> int:
        n_rows = 0
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(present)
            pages = client.iter_pages(
                list(fields.values()), where_clause, cfg.download.page_size, limit
            )
            for page in pages:
                for row in page:
                    writer.writerow([_flatten(row.get(fields[c])) for c in present])
                n_rows += len(page)
                logger.info("Downloaded %s rows", f"{n_rows:,}")
        return n_rows

    n = write(where)
    if n == 0 and date_filter == "signed_date":
        logger.warning(
            "No rows with the signing-date filter (date stored as text?); "
            "retrying with the system field :updated_at"
        )
        where, date_filter = _date_where(cfg, fields, {}, since, until, force_system=True)
        n = write(where)
    if n == 0:
        logger.warning("The query returned no rows")

    info = {
        "method": "datos.gov.co Socrata API",
        "api_base": client.base_url,
        "dataset_id": client.dataset_id,
        "endpoint": f"{client.base_url}/resource/{client.dataset_id}.json",
        "select": list(fields.values()),
        "where": where,
        "date_filter": date_filter,
        "order": ":id",
        "limit": limit,
        "field_names_from": resolved_from,
        "columns_missing_in_source": missing,
        "excluded_personal_data_columns": sorted(set(cfg.privacy.drop_columns)),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": n,
        "file_sha256": file_sha256(output),
    }
    write_source_sidecar(output, info)
    return info
