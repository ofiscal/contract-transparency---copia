"""Personal-data protection utilities.

SECOP is public information (Ley 1712 de 2014), but the datasets also contain
personal data of natural persons -- contractors hired by *prestación de
servicios*, legal representatives, supervisors, even bank account numbers.
Processing that data is governed by Ley 1581 de 2012 (Habeas Data) and its
regulatory decree 1377 de 2013. This module applies *privacy by default*:

1. ``drop_pii``          -- columns never needed for modelling are removed at load time.
2. ``pseudonymize``      -- identifiers needed for joins / concentration analyses are
                            replaced by a keyed HMAC-SHA256 pseudonym. A plain SHA-256 of a
                            cédula is NOT anonymous: the ID space (~10^10) is trivially
                            brute-forced, therefore a secret key is mandatory.
3. ``redact_text``       -- ID numbers, e-mails, phones and long numbers are masked in
                            free-text descriptions (this also removes target leakage such as
                            "por valor de $12.000.000").
4. ``mask_person_names`` -- names of natural persons are hidden in outputs; names of
                            companies (NIT) and public entities are kept.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
from pathlib import Path

import pandas as pd

from contract_transparency.config import Settings

logger = logging.getLogger(__name__)

PSEUDONYM_PREFIX = "p_"
_SALT_FILE = "pii_salt"

# --- free-text redaction patterns ------------------------------------------------
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# "C.C. 1.234.567", "cédula No. 1234567", "NIT 900.123.456-7", "CE 12345"
_ID_WITH_PREFIX = re.compile(
    r"(?i)\b(?:c\.?\s?c\.?|c[ée]dula(?:\s+de\s+ciudadan[ií]a)?|nit|c\.?\s?e\.?|t\.?\s?i\.?|"
    r"pasaporte|identificad[oa]\s+con)\s*(?:n[o°º]\.?|n[úu]mero|#)?\s*[:.]?\s*[\d][\d.,\s-]{4,}\d"
)
# Colombian mobile / landline phone numbers (3xx xxx xxxx, +57 ...).
_PHONE = re.compile(r"(?:\+?57\s?)?\b3\d{2}[\s-]?\d{3}[\s-]?\d{4}\b")
# Any remaining long number (>= 6 digits, optionally with thousands separators),
# incl. money amounts "$ 12.000.000". Short numbers (years, quantities) are kept.
# Amounts written with words ("12 millones", "$ 1,5 mil millones"): they state the
# contract value inside the text, which would let a text model echo the target.
_MONEY_WORDS = re.compile(
    r"(?i)\$?\s?\b\d+(?:[.,]\d+)*\s*(?:mil\s+millones|millones|mill[oó]n|billones|bill[oó]n)\b"
)
_LONG_NUMBER = re.compile(r"\$?\s?\b\d{1,3}(?:[.,]\d{3}){2,}(?:[.,]\d+)?\b|\$?\s?\b\d{6,}\b")


def get_salt(cfg: Settings) -> bytes:
    """Return the secret key used for pseudonymisation.

    Resolution order:
      1. environment variable ``cfg.privacy.salt_env_var`` (recommended in CI/servers);
      2. file ``<secrets_dir>/pii_salt`` (git-ignored, created with 0600 permissions
         on first use so that pseudonyms are stable between runs on one machine).

    Pseudonyms produced with different keys are NOT comparable: share the key
    (through a secret manager, never through git) between machines that must
    produce joinable outputs.
    """
    env_value = os.environ.get(cfg.privacy.salt_env_var)
    if env_value:
        return env_value.encode("utf-8")

    salt_path = Path(cfg.paths.secrets_dir) / _SALT_FILE
    if salt_path.exists():
        return salt_path.read_bytes().strip()

    salt_path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32).encode("utf-8")
    # Create with restrictive permissions from the start.
    fd = os.open(salt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(value)
    logger.warning(
        "No %s environment variable found: generated a new pseudonymisation key at %s. "
        "Back it up in a secret manager; losing it breaks pseudonym continuity.",
        cfg.privacy.salt_env_var,
        salt_path,
    )
    return value


def normalize_identifier(value: object) -> str | None:
    """Canonical form of an ID so that '1.234.567', '1234567 ' and 1234567.0 match."""
    try:
        if value is None or pd.isna(value):  # None, NaN, pd.NA, NaT
            return None
    except (TypeError, ValueError):  # non-scalar -> treat as a value
        pass
    text = str(value).strip().upper()
    if text in {
        "",
        "NA",
        "N/A",
        "<NA>",
        "NAN",
        "NONE",
        "NULL",
        "NO DEFINIDO",
        "NO APLICA",
        "SIN DATO",
    }:
        return None
    if re.fullmatch(r"\d+\.0", text):  # float-coerced integers
        text = text[:-2]
    text = re.sub(r"[^0-9A-Z]", "", text)
    return text or None


def pseudonymize(values: pd.Series, salt: bytes, length: int = 16) -> pd.Series:
    """Keyed, deterministic pseudonym (HMAC-SHA256) for each identifier."""

    def _one(v: object) -> str | None:
        norm = normalize_identifier(v)
        if norm is None:
            return None
        digest = hmac.new(salt, norm.encode("utf-8"), hashlib.sha256).hexdigest()
        return PSEUDONYM_PREFIX + digest[:length]

    # Hash each distinct value once (IDs repeat a lot in SECOP).
    uniques = values.dropna().unique()
    mapping = {u: _one(u) for u in uniques}
    return values.map(mapping).astype("object")


def drop_pii(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    present = [c for c in cfg.privacy.drop_columns if c in df.columns]
    if present:
        logger.info("Dropping %d personal-data columns", len(present))
    return df.drop(columns=present)


def apply_privacy(df: pd.DataFrame, cfg: Settings, salt: bytes) -> pd.DataFrame:
    """Drop and pseudonymise configured PII columns (returns a copy)."""
    out = drop_pii(df, cfg).copy()
    for col in cfg.privacy.pseudonymize_columns:
        if col in out.columns:
            out[col] = pseudonymize(out[col], salt)
    return out


_WHITESPACE = re.compile(r"\s+")


def redact_one(text: str) -> str:
    text = _EMAIL.sub("<EMAIL>", text)
    text = _ID_WITH_PREFIX.sub("<ID>", text)
    text = _PHONE.sub("<TEL>", text)
    text = _MONEY_WORDS.sub(" <NUM>", text)
    text = _LONG_NUMBER.sub(" <NUM>", text)
    return _WHITESPACE.sub(" ", text).strip()


def redact_text(text: pd.Series) -> pd.Series:
    """Mask personal identifiers and long numbers inside free text.

    Applied once per *distinct* text (SECOP descriptions repeat a lot).
    """
    s = text.fillna("").astype(str)
    mapping = {u: redact_one(u) for u in s.unique()}
    return s.map(mapping)


def mask_person_names(
    names: pd.Series,
    mask: pd.Series,
    pseudonyms: pd.Series | None = None,
    labels: pd.Series | str = "Persona natural",
) -> pd.Series:
    """Replace the names selected by ``mask`` with a neutral label.

    Companies and public entities keep their names (their contracting is public
    and is what auditors need to see). Natural persons are shown as
    ``"Persona natural <pseudonym>"`` and unclassified suppliers as
    ``"Proveedor sin clasificar <pseudonym>"`` so that repeated contracts can
    still be followed without exposing identity.
    """
    masked = names.astype("object").copy()
    label = labels if isinstance(labels, pd.Series) else pd.Series(labels, index=names.index)
    replacement = label.astype(str)
    if pseudonyms is not None:
        replacement = replacement + " " + pseudonyms.fillna("").astype(str)
    selected = mask.fillna(False).astype(bool)
    masked[selected] = replacement[selected]
    return masked
