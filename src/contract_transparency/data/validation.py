"""Data-quality checks run before training and scoring.

Errors stop the pipeline; warnings are logged and stored with the model
metadata so that a degraded extract is visible in the model history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from contract_transparency.config import Settings
from contract_transparency.data import columns as C


@dataclass
class ValidationReport:
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return asdict(self) | {"ok": self.ok}

    def raise_if_failed(self) -> None:
        if self.errors:
            raise ValueError("Data validation failed:\n- " + "\n- ".join(self.errors))


def validate_raw(df: pd.DataFrame, cfg: Settings) -> ValidationReport:
    """Schema checks on the raw extract (before cleaning)."""
    s = cfg.schema_
    rep = ValidationReport(n_rows=len(df))
    if df.empty:
        rep.errors.append("Dataset is empty")
        return rep
    for required in (s.id, s.target):
        if required not in df.columns:
            rep.errors.append(f"Required column missing: '{required}'")
    optional = [
        *s.text,
        *s.categorical,
        s.unspsc,
        s.duration,
        s.signed_date,
        s.supplier_id,
        s.supplier_doc_type,
        s.modality,
        s.contract_type,
    ]
    missing = sorted({c for c in optional if c and c not in df.columns})
    if missing:
        rep.warnings.append(f"Configured columns not found (features will be skipped): {missing}")
    leaked = [c for c in cfg.privacy.drop_columns if c in df.columns]
    if leaked:
        rep.errors.append(f"PII columns still present after loading: {leaked}")
    return rep


def validate_clean(df: pd.DataFrame, cfg: Settings, for_training: bool = True) -> ValidationReport:
    """Content checks on the cleaned table."""
    rep = ValidationReport(n_rows=len(df))
    if df.empty:
        rep.errors.append("No rows after cleaning")
        return rep
    value_na = float(df[C.VALUE].isna().mean())
    out_of_range = float(df[C.VALUE_OUT_OF_RANGE].mean())
    signed_na = float(df[C.SIGNED_AT].isna().mean())
    text_empty = float((df[C.TEXT].str.len() == 0).mean())
    dup = float(df[C.ID].duplicated().mean())
    rep.stats = {
        "share_value_missing": value_na,
        "share_value_out_of_range": out_of_range,
        "share_signed_date_missing": signed_na,
        "share_text_empty": text_empty,
        "share_duplicated_ids": dup,
    }
    if C.PERSON_TYPE in df.columns:
        shares = df[C.PERSON_TYPE].value_counts(normalize=True)
        for kind in ("natural", "juridica", "desconocido"):
            rep.stats[f"share_person_{kind}"] = float(shares.get(kind, 0.0))
        if "person_type_agreement" in df.attrs and df.attrs["person_type_agreement"] is not None:
            rep.stats["person_type_number_rule_agreement"] = float(
                df.attrs["person_type_agreement"]
            )
        if rep.stats["share_person_desconocido"] > 0.1:
            rep.warnings.append(
                f"{rep.stats['share_person_desconocido']:.1%} of suppliers could not be classified as "
                "natural/juridica: check schema.supplier_doc_type / supplier_id"
            )
    if for_training:
        usable = int((df[C.VALUE].notna() & ~df[C.VALUE_OUT_OF_RANGE]).sum())
        rep.stats["n_trainable"] = usable
        if usable < 200:
            rep.errors.append(
                f"Only {usable} rows with a valid contract value; need >= 200 to train"
            )
        if cfg.evaluation.split == "temporal" and signed_na > 0.5:
            rep.errors.append(
                "More than 50% of signing dates are missing: temporal split impossible "
                "(fix `schema.signed_date` or use evaluation.split=random)"
            )
    if value_na > 0.2:
        rep.warnings.append(f"{value_na:.1%} of contract values could not be parsed")
    if out_of_range > 0.2:
        rep.warnings.append(f"{out_of_range:.1%} of contract values outside [min_value, max_value]")
    if text_empty > 0.3:
        rep.warnings.append(f"{text_empty:.1%} of contracts have no description")
    return rep
