"""Cleaning of raw SECOP extracts into a model-ready table.

Replaces ``python3_10/modules/cleaning.py``. The legacy module had one reader
per column family (``secop2_general``, ``secop2_valor``, ``secop2_categoric``,
...) each re-reading the CSV; here the file is read once and every
transformation is a small, pure, unit-tested function.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from contract_transparency.config import Settings
from contract_transparency.data import columns as C
from contract_transparency.data.person_type import (
    JURIDICA,
    NATURAL,
    UNKNOWN,
    agreement_with_doc_type,
    classify_person_type,
)
from contract_transparency.data.privacy import (
    apply_privacy,
    mask_person_names,
    pseudonymize,
    redact_text,
)

logger = logging.getLogger(__name__)

_MISSING_TOKENS = {
    "",
    "nan",
    "none",
    "null",
    "no definido",
    "no aplica",
    "n/a",
    "na",
    "sin definir",
}


# --------------------------------------------------------------------------- #
# Scalar parsers
# --------------------------------------------------------------------------- #
def _parse_money_one(raw: str) -> float:
    x = re.sub(r"[^\d,.\-]", "", raw)
    if not x or x in {"-", ".", ","}:
        return np.nan
    if "," in x and "." in x:
        if x.rfind(",") > x.rfind("."):  # 1.234.567,89  (es-CO)
            x = x.replace(".", "").replace(",", ".")
        else:  # 1,234,567.89   (en-US, legacy "VALOR NORM" files)
            x = x.replace(",", "")
    elif "," in x:
        parts = x.split(",")
        x = x.replace(",", "") if len(parts) > 2 or len(parts[-1]) == 3 else x.replace(",", ".")
    elif x.count(".") > 1:  # 1.234.567
        x = x.replace(".", "")
    try:
        return float(x)
    except ValueError:
        return np.nan


def parse_money(values: pd.Series) -> pd.Series:
    """Parse COP amounts written as ``"$1,234,567"``, ``"1.234.567,00"`` or ``1234567``.

    Unparseable values become NaN (the legacy code crashed or silently replaced
    them with 1e10).
    """
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    s = values.astype("string").str.strip()
    uniques = s.dropna().unique()
    mapping = {u: _parse_money_one(u) for u in uniques}
    return s.map(mapping).astype(float)


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


_DURATION_UNITS = {"dia": 1.0, "semana": 7.0, "mes": 30.4375, "ano": 365.25, "hora": 1 / 24}
_DURATION_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-z]+)")


def parse_duration_days(values: pd.Series) -> pd.Series:
    """``"30 Dia(s)"`` -> 30, ``"3 Mes(es)"`` -> 91.3, ``"1 Año(s)"`` -> 365.25."""

    def _one(raw: object) -> float:
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return np.nan
        text = _strip_accents(str(raw)).casefold()
        m = _DURATION_RE.search(text)
        if not m:
            return np.nan
        number = float(m.group(1).replace(",", "."))
        unit = m.group(2)
        for prefix, factor in _DURATION_UNITS.items():
            if unit.startswith(prefix):
                return number * factor
        return np.nan

    uniques = values.dropna().unique()
    mapping = {u: _one(u) for u in uniques}
    return values.map(mapping).astype(float)


def parse_dates(values: pd.Series, dayfirst: bool = False, fmt: str | None = None) -> pd.Series:
    """Robust date parsing; invalid dates become NaT (never raise)."""
    return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, format=fmt or "mixed")


_LINEBREAKS = re.compile(r"(\\r\\n|\r\n|\\n|\n|\r|\t)")
_SPACES = re.compile(r"\s+")


def normalize_text(values: pd.Series) -> pd.Series:
    """Remove line breaks (legacy ``post_processing``) and collapse whitespace."""
    s = values.fillna("").astype(str)
    mapping = {u: _SPACES.sub(" ", _LINEBREAKS.sub(" ", u)).strip() for u in s.unique()}
    return s.map(mapping)


def combine_text(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Join several free-text columns, skipping missing ones and duplicates."""
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.Series("", index=df.index)
    parts = [df[c].fillna("").astype(str) for c in present]

    def _join(row: tuple[str, ...]) -> str:
        seen: list[str] = []
        for p in row:
            if p and p.casefold() not in {s.casefold() for s in seen}:
                seen.append(p)
        return " | ".join(seen)

    return pd.Series([_join(r) for r in zip(*parts, strict=True)], index=df.index)


def unspsc_levels(codes: pd.Series) -> pd.DataFrame:
    """Split an UNSPSC code (``"V1.80111600"``) into segment / family / class.

    The hierarchy lets the model generalise to rare product codes: an unseen
    class still shares its segment with known contracts.
    """
    digits = codes.fillna("").astype(str).str.replace(r"^[A-Za-z]\d\.", "", regex=True)
    digits = digits.str.replace(r"\D", "", regex=True)
    valid = digits.str.len() >= 6
    return pd.DataFrame(
        {
            C.UNSPSC_SEGMENT: digits.str[:2].where(valid),
            C.UNSPSC_FAMILY: digits.str[:4].where(valid),
            C.UNSPSC_CLASS: digits.str[:6].where(valid),
        },
        index=codes.index,
    )


def clean_categorical(values: pd.Series) -> pd.Series:
    s = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.str.casefold().isin(_MISSING_TOKENS)).astype("object")


# --------------------------------------------------------------------------- #
# Table-level cleaning
# --------------------------------------------------------------------------- #
def clean_contracts(raw: pd.DataFrame, cfg: Settings, salt: bytes) -> pd.DataFrame:
    """Apply privacy rules and derive the canonical ``ct_*`` columns.

    The output keeps the (non-PII) raw columns so that scored files remain
    understandable by auditors.
    """
    s = cfg.schema_
    if s.id not in raw.columns:
        raise KeyError(f"ID column '{s.id}' not found. Check `schema.id` in the config.")
    if s.target not in raw.columns:
        raise KeyError(f"Target column '{s.target}' not found. Check `schema.target`.")

    # Keep the raw doc type / supplier id before pseudonymisation changes them.
    raw_supplier_id = raw[s.supplier_id] if s.supplier_id and s.supplier_id in raw else None
    raw_supplier_name = raw[s.supplier_name] if s.supplier_name and s.supplier_name in raw else None
    df = apply_privacy(raw, cfg, salt)

    df[C.ID] = df[s.id].astype("string").str.strip()

    # Target -----------------------------------------------------------------
    df[C.VALUE] = parse_money(df[s.target])
    df[C.VALUE_OUT_OF_RANGE] = ~df[C.VALUE].between(cfg.data.min_value, cfg.data.max_value)

    # Dates --------------------------------------------------------------------
    for role, out_col in (
        ("signed_date", C.SIGNED_AT),
        ("start_date", C.START_AT),
        ("end_date", C.END_AT),
    ):
        col = getattr(s, role)
        df[out_col] = (
            parse_dates(df[col], cfg.data.dayfirst, cfg.data.date_format)
            if col and col in df
            else pd.NaT
        )

    # Duration -----------------------------------------------------------------
    duration = (
        parse_duration_days(df[s.duration])
        if s.duration and s.duration in df
        else pd.Series(np.nan, index=df.index)
    )
    from_dates = (df[C.END_AT] - df[C.START_AT]).dt.days.astype(float)
    from_dates = from_dates.where(from_dates >= 0)
    df[C.DURATION_DAYS] = duration.fillna(from_dates)

    # Text: redact each source column (never keep un-redacted free text), then combine.
    for col in s.text:
        if col in df:
            df[col] = normalize_text(df[col])
            if cfg.privacy.redact_free_text:
                df[col] = redact_text(df[col])
    df[C.TEXT] = combine_text(df, s.text)

    # Categorical ----------------------------------------------------------------
    for col in s.categorical:
        if col in df:
            df[col] = clean_categorical(df[col])
    if s.unspsc and s.unspsc in df:
        df = df.join(unspsc_levels(df[s.unspsc]))
    else:
        for c in (C.UNSPSC_SEGMENT, C.UNSPSC_FAMILY, C.UNSPSC_CLASS):
            df[c] = None

    # Parties --------------------------------------------------------------------
    # Natural person vs legal entity from document type, number and name
    # (classified on the RAW number, before pseudonymisation).
    doc_type = (
        df[s.supplier_doc_type] if s.supplier_doc_type and s.supplier_doc_type in df else None
    )
    group = df[s.supplier_group] if s.supplier_group and s.supplier_group in df else None
    if doc_type is not None or raw_supplier_id is not None or raw_supplier_name is not None:
        kinds = classify_person_type(
            doc_type,
            raw_supplier_id,
            raw_supplier_name,
            group,
            cfg.privacy.natural_person_doc_types,
        )
        df[C.PERSON_TYPE] = kinds["person_type"].to_numpy()
        df[C.PERSON_TYPE_RULE] = kinds["rule"].to_numpy()
        if doc_type is not None and raw_supplier_id is not None:
            # Quality check of the number rule where the document type is known.
            df.attrs["person_type_agreement"] = agreement_with_doc_type(doc_type, raw_supplier_id)
    else:
        df[C.PERSON_TYPE], df[C.PERSON_TYPE_RULE] = UNKNOWN, "no_data"
    df[C.NATURAL_PERSON] = df[C.PERSON_TYPE] == NATURAL
    # Privacy: names are masked unless the supplier is positively a legal entity.
    mask_name = df[C.PERSON_TYPE] != JURIDICA

    if raw_supplier_id is not None:
        df[C.SUPPLIER] = (
            df[s.supplier_id]
            if s.supplier_id in cfg.privacy.pseudonymize_columns
            else pseudonymize(raw_supplier_id, salt)
        )
    else:
        df[C.SUPPLIER] = None

    if s.supplier_name and s.supplier_name in df:
        names = clean_categorical(df[s.supplier_name])
        df[C.SUPPLIER_DISPLAY] = (
            mask_person_names(
                names,
                mask_name,
                df[C.SUPPLIER],
                labels=df[C.PERSON_TYPE]
                .map({NATURAL: "Persona natural"})
                .fillna("Proveedor sin clasificar"),
            )
            if cfg.privacy.mask_natural_person_names
            else names
        )
        if cfg.privacy.mask_natural_person_names:
            df[s.supplier_name] = df[C.SUPPLIER_DISPLAY]
    else:
        df[C.SUPPLIER_DISPLAY] = df[C.SUPPLIER]

    if s.legal_rep_id and s.legal_rep_id in df:
        df[C.LEGAL_REP] = (
            df[s.legal_rep_id]
            if s.legal_rep_id in cfg.privacy.pseudonymize_columns
            else pseudonymize(df[s.legal_rep_id], salt)
        )
    else:
        df[C.LEGAL_REP] = None

    entity_col = s.entity_id if s.entity_id and s.entity_id in df else s.entity_name
    df[C.ENTITY] = clean_categorical(df[entity_col]) if entity_col and entity_col in df else None

    # De-duplicate: SECOP exports may contain the same contract more than once.
    before = len(df)
    df = df.drop_duplicates(subset=[C.ID], keep="last").reset_index(drop=True)
    if len(df) < before:
        logger.info("Removed %d duplicated contract ids", before - len(df))
    return df


def training_mask(df: pd.DataFrame) -> pd.Series:
    """Rows usable for training: valid target inside the configured range."""
    return df[C.VALUE].notna() & ~df[C.VALUE_OUT_OF_RANGE]
