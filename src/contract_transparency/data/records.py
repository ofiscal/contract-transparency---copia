"""Official public records matched against contract suppliers.

Typical sources (configured in ``records.sources``):

* **Contraloría General de la República - Boletín de Responsables Fiscales.**
  Ley 610 de 2000, art. 60: entities must abstain from contracting with
  anyone listed in the bulletin. A contract signed with a listed supplier (or
  a company whose legal representative is listed) is therefore a strong,
  legally grounded alert signal.
* **SECOP - Multas y sanciones** (datos.gov.co): previous fines/sanctions of
  a supplier.

Privacy: these files contain names and ID numbers of natural persons. The
importer keeps **only keyed pseudonyms** of the IDs (same HMAC key as the
contracts, so they match ``ct_supplier`` / ``ct_legal_rep``) plus the record
date; names and raw IDs are discarded immediately. A manifest records the
official source, the publisher and the SHA-256 of the imported file, so every
match can be audited against the official publication.

The exact column layout of each publication may change between editions:
columns are auto-detected (header row, ID column, date column) and can be
pinned in the config (``id_column`` / ``date_column``).
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract_transparency.config import RecordSourceConfig, Settings
from contract_transparency.data import columns as C
from contract_transparency.data.cleaning import parse_dates
from contract_transparency.data.privacy import pseudonymize

logger = logging.getLogger(__name__)

RECORDS_DIR = "records"
KEY = "record_key"
DATE = "record_date"

_ID_PATTERNS = [
    r"n[uú]mero.*(documento|identificaci)",
    r"(documento|identificaci[oó]n).*(n[uú]mero|no\.?$)",
    r"^(nit|c[eé]dula|cc|documento|identificaci[oó]n|nro.?documento|no.?documento)$",
    r"nit.*c[eé]dula|c[eé]dula.*nit",
    r"documento",
]
_DATE_PATTERNS = [r"fecha.*(inclusi|ejecutori|fallo|firmeza|sanci|publicaci|resoluci)", r"^fecha"]


def _norm(text: object) -> str:
    t = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t).strip()


def _find_header_row(raw: pd.DataFrame, max_rows: int = 30) -> int:
    """Official Excel files often have title rows above the header."""
    for i in range(min(max_rows, len(raw))):
        cells = [_norm(v) for v in raw.iloc[i].tolist() if pd.notna(v)]
        if any(re.search(p, c) for c in cells for p in _ID_PATTERNS[:4]):
            return i
    return 0


def read_records_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raw = pd.read_csv(
            path, header=None, dtype=str, sep=None, engine="python", encoding_errors="replace"
        )
    header = _find_header_row(raw)
    df = raw.iloc[header + 1 :].reset_index(drop=True)
    df.columns = [
        str(c).strip() if pd.notna(c) else f"col_{i}" for i, c in enumerate(raw.iloc[header])
    ]
    return df.dropna(how="all")


def detect_column(columns: list[str], patterns: list[str]) -> str | None:
    normed = {c: _norm(c) for c in columns}
    for p in patterns:
        for c, n in normed.items():
            if re.search(p, n):
                return c
    return None


def id_variants(value: object) -> list[str]:
    """Candidate identifiers for matching.

    ``"900.123.456-7"`` -> ``["9001234567", "900123456"]``: NITs may be published
    with or without the verification digit, so both variants are kept.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if len(digits) < 5:
        return []
    out = [digits]
    if re.search(r"\d\s*-\s*\d$", text):  # explicit verification digit
        out.append(digits[:-1])
    return out


def records_dir(cfg: Settings) -> Path:
    return Path(cfg.paths.processed_dir) / RECORDS_DIR


def import_records(
    cfg: Settings, source: str, path: str | Path, salt: bytes
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read an official publication, keep only pseudonymised IDs (+ date), save it."""
    if source not in cfg.records.sources:
        raise KeyError(f"Unknown record source '{source}'. Configured: {list(cfg.records.sources)}")
    src: RecordSourceConfig = cfg.records.sources[source]
    from contract_transparency.provenance import file_sha256

    df = read_records_file(path)
    id_col = src.id_column or detect_column(list(df.columns), _ID_PATTERNS)
    if not id_col or id_col not in df.columns:
        raise ValueError(
            f"Could not find the ID column in {path}. Columns: {list(df.columns)}. "
            f"Set records.sources.{source}.id_column in the config."
        )
    date_col = src.date_column or detect_column(list(df.columns), _DATE_PATTERNS)

    rows = []
    for _, r in df.iterrows():
        variants = id_variants(r[id_col])
        date = r[date_col] if date_col else None
        rows.extend({"raw": v, "date": date} for v in variants)
    tmp = pd.DataFrame(rows, columns=["raw", "date"])
    out = pd.DataFrame(
        {
            KEY: pseudonymize(tmp["raw"], salt),
            DATE: parse_dates(tmp["date"], dayfirst=True) if date_col else pd.NaT,
        }
    )
    # Keep the earliest date per key (first time the person/company was listed).
    out = out.dropna(subset=[KEY]).sort_values(DATE).drop_duplicates(KEY, keep="first")
    out = out.reset_index(drop=True)

    n_ids = int(df[id_col].map(lambda v: bool(id_variants(v))).sum())
    manifest = {
        "source": source,
        "title": src.title,
        "publisher": src.publisher,
        "official_url": src.official_url,
        "legal_basis": src.legal_basis,
        "file": Path(path).name,
        "file_sha256": file_sha256(path),
        "imported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_in_file": int(len(df)),
        "rows_with_valid_id": n_ids,
        "id_column": id_col,
        "date_column": date_col,
        "stored": "HMAC pseudonyms of IDs and record dates only (names and raw IDs discarded)",
    }
    target = records_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target / f"{source}.parquet", index=False)
    (target / f"{source}.manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Imported %d records from %s (%s)", n_ids, src.title, Path(path).name)
    return out, manifest


def load_records(cfg: Settings) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    """All imported record sources: {source: (records, manifest)}."""
    out: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    if not cfg.records.enabled:
        return out
    d = records_dir(cfg)
    for source in cfg.records.sources:
        data, man = d / f"{source}.parquet", d / f"{source}.manifest.json"
        if data.exists() and man.exists():
            out[source] = (pd.read_parquet(data), json.loads(man.read_text(encoding="utf-8")))
    return out


def match_records(
    df: pd.DataFrame, cfg: Settings, records: dict[str, tuple[pd.DataFrame, dict[str, Any]]]
) -> pd.DataFrame:
    """One boolean column ``sig_<flag>`` per imported source."""
    out = pd.DataFrame(index=df.index)
    role_col = {"supplier": C.SUPPLIER, "legal_rep": C.LEGAL_REP}
    for source, (rec, _) in records.items():
        src = cfg.records.sources[source]
        first_date = rec.set_index(KEY)[DATE]
        hit = pd.Series(False, index=df.index)
        for role in src.match_on:
            keys = df[role_col[role]]
            listed = keys.isin(first_date.index)
            if src.only_after_record_date and first_date.notna().any():
                since = keys.map(first_date)
                # listed with a known date -> only contracts signed on/after it
                listed &= since.isna() | (df[C.SIGNED_AT] >= since)
            hit |= listed.fillna(False)
        out["sig_" + src.flag] = hit.astype(bool)
    return out
