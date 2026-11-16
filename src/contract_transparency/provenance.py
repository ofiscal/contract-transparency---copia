"""Public-data provenance: every result must be traceable to an official public source.

Policy implemented here (see docs/DATA_PROTECTION.md):

* **Link, don't copy.** Each scored contract carries ``secop_url`` (the
  official SECOP process page) and ``open_data_url`` (the exact record in the
  datos.gov.co API). Anyone can verify the inputs of every result without
  trusting this project.
* Rows without a valid link to an allowed official domain are **not
  published** (``public_data.require_secop_url``); they are counted in the
  manifest instead.
* Every output file has a sidecar manifest (``<file>.manifest.json``) with the
  notice, the sources (URL, publisher, retrieval date, SHA-256 of the file
  used), the model version, and a column dictionary saying which columns are
  copied verbatim from the public source and which ones are statistical
  estimates computed by this project.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pandas as pd

from contract_transparency import __version__
from contract_transparency.config import Settings

SOURCE_SIDECAR_SUFFIX = ".source.json"
MANIFEST_SUFFIX = ".manifest.json"

DERIVED_COLUMNS: dict[str, str] = {
    "review_priority": "Orden sugerido de revisión según el índice de alerta (1 = primero).",
    "alert_index": "Índice estadístico 0-1: combinación ponderada de las señales sig_*.",
    "n_signals": "Número de señales activas.",
    "signals": "Lista de señales activas.",
    "predicted_value": "Valor esperado estimado por el modelo para un contrato con estas características (COP).",
    "log_residual": "log(1+valor real) - log(1+valor esperado).",
    "pct_error": "(valor real - valor esperado) / valor esperado.",
    "excess_value": "Valor real por encima del esperado (COP); 0 si no lo supera.",
    "excess_ratio": "Fórmula heredada `flagging`: (real - esperado) / (real + 1).",
    "weighted_excess": "Fórmula heredada `flagging`: excess_ratio x (real - esperado).",
    "tail_probability": "Probabilidad estimada de que un contrato comparable sea al menos así de costoso respecto a lo esperado. Valores pequeños = atípico.",
    "within_tolerance": "El valor esperado está dentro de +-tolerancia del valor real.",
    "days_to_next_restriction": "Días desde la firma hasta el inicio del siguiente periodo de Ley de Garantías.",
    "supplier": "Nombre del proveedor (empresas) o 'Persona natural <seudónimo>'.",
    "supplier_pseudonym": "Seudónimo HMAC del documento del proveedor (no reversible sin la clave).",
    "natural_person": "El proveedor es persona natural (regla de clasificación, ver person_type).",
    "person_type": "Tipo de proveedor: natural, juridica o desconocido, según el tipo y número de documento publicados (regla de numeración colombiana) y el nombre.",
    "model_segment": "Modelo que estimó el valor esperado: natural, juridica o _global.",
    "contract_value": "Valor del contrato publicado en SECOP, convertido a número (COP).",
    "duration_days": "Duración del contrato publicada, convertida a días.",
    "secop_url": "Enlace oficial al proceso en SECOP (fuente para verificar).",
    "open_data_url": "Enlace al registro exacto en la API de datos abiertos (datos.gov.co).",
    "model_version": "Versión del modelo del registro que produjo el resultado.",
    "aviso": "Recordatorio: indicador estadístico, no prueba.",
}


# --------------------------------------------------------------------------- #
# hashing & code version
# --------------------------------------------------------------------------- #
def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_version() -> dict[str, str | None]:
    commit = os.environ.get("CT_GIT_COMMIT")  # baked into the Docker image
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        except Exception:  # git not installed or not a repository
            commit = None
    return {"package": f"contract-transparency {__version__}", "git_commit": commit}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# audit links
# --------------------------------------------------------------------------- #
def valid_official_url(url: object, allowed_domains: list[str]) -> str | None:
    """Return the URL if it is http(s) and its host belongs to an allowed domain."""
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"https", "http"}:
        return None
    if any(host == d or host.endswith("." + d) for d in allowed_domains):
        return url
    return None


def open_data_url(cfg: Settings, contract_id: str) -> str:
    p = cfg.public_data
    return f"{p.api_base.rstrip('/')}/resource/{p.dataset_id}.json?{p.id_api_field}={quote(str(contract_id), safe='')}"


def audit_links(df: pd.DataFrame, ids: pd.Series, cfg: Settings) -> pd.DataFrame:
    """``secop_url`` (validated official link) and ``open_data_url`` per row."""
    url_col = cfg.schema_.url
    raw = df[url_col] if url_col and url_col in df.columns else pd.Series(None, index=df.index)
    allowed = cfg.public_data.allowed_url_domains
    return pd.DataFrame(
        {
            "secop_url": raw.map(lambda u: valid_official_url(u, allowed)),
            "open_data_url": ids.map(lambda i: open_data_url(cfg, i) if pd.notna(i) else None),
        },
        index=df.index,
    )


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #
def read_source_sidecar(data_path: str | Path) -> dict[str, Any] | None:
    side = Path(str(data_path) + SOURCE_SIDECAR_SUFFIX)
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    return None


def write_source_sidecar(data_path: str | Path, info: dict[str, Any]) -> Path:
    side = Path(str(data_path) + SOURCE_SIDECAR_SUFFIX)
    side.write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return side


def contracts_source(cfg: Settings, path: str | Path, n_rows: int) -> dict[str, Any]:
    """Provenance of a raw contracts file (API download sidecar or manual export)."""
    p = cfg.public_data
    info: dict[str, Any] = {
        "name": p.dataset_name,
        "publisher": p.publisher,
        "portal_url": p.portal_url,
        "dataset_id": p.dataset_id,
        "file": Path(path).name,
        "file_sha256": file_sha256(path),
        "rows_read": n_rows,
        "synthetic": p.synthetic,
    }
    sidecar = read_source_sidecar(path)
    if sidecar:
        info["retrieval"] = sidecar
    else:
        info["retrieval"] = {
            "method": "synthetic generator" if p.synthetic else "manual export from the portal",
            "note": "Add a `<file>.source.json` (written by `ct download`) for exact query provenance.",
        }
    return info


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def column_dictionary(columns: list[str], cfg: Settings) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    source_name = f"{cfg.public_data.dataset_name} (datos.gov.co)"
    redacted = set(cfg.schema_.text) if cfg.privacy.redact_free_text else set()
    for c in columns:
        if c == "secop_url":
            out[c] = {
                "origin": "source (validated)",
                "description": f"Columna '{cfg.schema_.url}' de {source_name}: enlace oficial al "
                "proceso en SECOP; solo se publican enlaces a dominios oficiales.",
            }
        elif c in DERIVED_COLUMNS:
            out[c] = {"origin": "derived", "description": DERIVED_COLUMNS[c]}
        elif c.startswith("sig_"):
            out[c] = {
                "origin": "derived",
                "description": f"Señal estadística '{c[len('sig_') :]}' (regla o modelo); no implica irregularidad.",
            }
        elif c in redacted:
            out[c] = {
                "origin": "source (redacted)",
                "description": f"{source_name}; números de identificación y cifras enmascarados.",
            }
        else:
            out[c] = {"origin": "source", "description": f"Copiado de {source_name}."}
    return out


def build_manifest(
    cfg: Settings,
    columns: list[str],
    sources: list[dict[str, Any]],
    model_meta: dict[str, Any],
    rows: dict[str, int],
) -> dict[str, Any]:
    return {
        "notice_es": cfg.public_data.notice_es,
        "notice_en": cfg.public_data.notice_en,
        "synthetic_data": cfg.public_data.synthetic,
        "generated_at": _now(),
        "generator": code_version(),
        "model": {
            "version": model_meta.get("version"),
            "config_fingerprint": model_meta.get("config_fingerprint"),
            "train_dates": model_meta.get("data", {}).get("train_dates"),
            "test_metrics": model_meta.get("metrics", {}).get("test"),
        },
        "sources": sources,
        "rows": rows,
        "how_to_verify": (
            "Para cada fila: abra `secop_url` (o `open_data_url`) y compare los datos publicados "
            "con las columnas de origen 'source'. Las columnas 'derived' son estimaciones "
            "estadísticas de este proyecto, no datos oficiales."
        ),
        "columns": column_dictionary(columns, cfg),
    }


def write_manifest(output_path: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(str(output_path) + MANIFEST_SUFFIX)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path
