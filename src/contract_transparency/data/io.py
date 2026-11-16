"""Reading and writing tables (CSV / Parquet / Excel)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from contract_transparency.config import Settings

logger = logging.getLogger(__name__)


def read_contracts(path: str | Path, cfg: Settings, nrows: int | None = None) -> pd.DataFrame:
    """Read a raw SECOP extract.

    * Every column is read as text (``dtype=str``); typing is done explicitly in
      :mod:`cleaning`, which avoids pandas' mixed-type guessing on 500k+ rows.
    * Columns listed in ``privacy.drop_columns`` are excluded **at read time**
      for CSV files, so that personal data never reaches memory.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Download SECOP II from datos.gov.co or run "
            "`ct demo-data` to generate a synthetic dataset."
        )
    nrows = nrows if nrows is not None else cfg.data.nrows
    drop = set(cfg.privacy.drop_columns)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".gz", ".zip"}:
        df = pd.read_csv(
            path,
            sep=cfg.data.csv_sep,
            encoding=cfg.data.encoding,
            dtype=str,
            nrows=nrows,
            usecols=lambda c: c not in drop,
            keep_default_na=True,
            low_memory=False,
        )
    elif suffix == ".json":
        # JSON saved from the datos.gov.co API (e.g. .../jbjy-vk9h.json?$limit=10000).
        from contract_transparency.data.download import read_api_json

        df = read_api_json(path, cfg)
        if nrows:
            df = df.head(nrows)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
        df = df.drop(columns=[c for c in df.columns if c in drop])
        if nrows:
            df = df.head(nrows)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, dtype=str, nrows=nrows)
        df = df.drop(columns=[c for c in df.columns if c in drop])
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    df.columns = [str(c).strip() for c in df.columns]
    logger.info("Loaded %s rows x %s columns from %s", len(df), df.shape[1], path)
    return df


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    fmt: str | None = None,
    csv_sep: str = ";",
    manifest: dict | None = None,
) -> Path:
    """Write a table. Excel files get the notice as their FIRST sheet ("AVISO")
    and a column dictionary; other formats rely on the ``.manifest.json`` sidecar."""
    path = Path(path)
    fmt = fmt or path.suffix.lstrip(".") or "csv"
    path = path.with_suffix("." + fmt)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    elif fmt == "xlsx":
        with pd.ExcelWriter(path) as writer:
            if manifest:
                notice = [manifest.get("notice_es", ""), "", manifest.get("notice_en", "")]
                if manifest.get("synthetic_data"):
                    notice = [
                        "DATOS SINTÉTICOS - NO REALES / SYNTHETIC DATA - NOT REAL",
                        "",
                        *notice,
                    ]
                notice += ["", manifest.get("how_to_verify", "")]
                pd.DataFrame({"AVISO": notice}).to_excel(writer, sheet_name="AVISO", index=False)
            df.to_excel(writer, sheet_name="resultados", index=False)
            if manifest:
                cols = pd.DataFrame.from_dict(manifest.get("columns", {}), orient="index")
                cols.rename_axis("columna").to_excel(writer, sheet_name="diccionario")
    else:
        # utf-8-sig so that Excel (es-CO) opens accents correctly.
        df.to_csv(path, index=False, sep=csv_sep, encoding="utf-8-sig")
    logger.info("Wrote %s rows to %s", len(df), path)
    return path
