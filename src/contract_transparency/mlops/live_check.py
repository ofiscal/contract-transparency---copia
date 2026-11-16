"""End-to-end verification against the LIVE public sources.

``ct live-check`` runs the complete pipeline on a fresh sample downloaded from
datos.gov.co, inside an isolated work directory (its own snapshot, registry
and outputs; the production registry is never touched), and writes a
PASS/WARN/FAIL report (``report.json`` + ``report.md``). Every WARN/FAIL comes
with a hint, and the CLI prints them, so the report is actionable without
reading code.

Steps:
  environment        Python / platform / proxy & certificate settings (for diagnosis)
  api_metadata       dataset metadata readable (if blocked: fall back to a sample record)
  schema_*           configured columns exist in the live dataset
  json_field_names   names used when reading JSON saved from the API URL
  id_api_field       API field of the contract id (used by open_data_url)
  download           fresh sample (signing-date filter; :updated_at fallback)
  prepare            validation + cleaning
  records_import     official records (optional)
  train              training + promotion gates
  score              scoring; share of contracts with an official SECOP link
  audit_links        sampled open_data_url return that contract with the same value
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contract_transparency.config import PathsConfig, Settings
from contract_transparency.data.cleaning import parse_money
from contract_transparency.data.download import (
    DownloadError,
    api_field_map,
    client_from_config,
    download_contracts,
    needed_columns,
)

logger = logging.getLogger(__name__)

MIN_ROWS_TO_TRAIN = 200


class _Report:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(self, name: str, status: str, hint: str | None = None, **details: Any) -> None:
        step = {"step": name, "status": status, **details}
        if hint:
            step["hint"] = hint
        self.steps.append(step)
        level = logging.ERROR if status == "FAIL" else logging.INFO
        logger.log(
            level, "[%s] %s %s", status, name, {k: v for k, v in details.items() if k != "trace"}
        )

    @property
    def ok(self) -> bool:
        return all(s["status"] != "FAIL" for s in self.steps)

    def markdown(self) -> str:
        lines = [
            "# Live check report",
            "",
            "| Step | Status | Details | Hint |",
            "|---|---|---|---|",
        ]
        for s in self.steps:
            det = {k: v for k, v in s.items() if k not in {"step", "status", "hint", "trace"}}
            det_txt = json.dumps(det, ensure_ascii=False, default=str)[:500].replace("|", "\\|")
            lines.append(f"| {s['step']} | {s['status']} | `{det_txt}` | {s.get('hint', '')} |")
        traces = [s for s in self.steps if "trace" in s]
        for s in traces:
            lines += ["", f"## Traceback: {s['step']}", "", "```", s["trace"], "```"]
        lines += ["", f"**Overall: {'PASS' if self.ok else 'FAIL'}**"]
        return "\n".join(lines)


def _isolated(cfg: Settings, workdir: Path) -> Settings:
    paths = PathsConfig(
        raw_data=workdir / "raw" / "secop_sample.csv",
        processed_dir=workdir / "processed",
        outputs_dir=workdir / "outputs",
        registry_dir=workdir / "registry",
        secrets_dir=cfg.paths.secrets_dir,  # same key -> same pseudonyms
    )
    return cfg.model_copy(update={"paths": paths})


def _environment() -> dict[str, Any]:
    from contract_transparency.environment import installed_version

    proxy = {
        k: bool(os.environ.get(k)) for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "NO_PROXY")
    }
    certs = {
        k: os.environ.get(k)
        for k in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CT_CA_BUNDLE")
        if os.environ.get(k)
    }
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "contract_transparency": installed_version("contract-transparency"),
        "truststore": installed_version("truststore"),
        "proxy_env_set": {k: v for k, v in proxy.items() if v},
        "certificate_env": certs,
    }


def live_check(
    cfg: Settings,
    workdir: str | Path = "data/live_check",
    since: str | None = None,
    until: str | None = None,
    limit: int = 20_000,
    fiscal_bulletin: str | Path | None = None,
    verify_links: int = 5,
    api_base: str | None = None,
) -> dict[str, Any]:
    from contract_transparency.data.privacy import get_salt
    from contract_transparency.data.records import import_records
    from contract_transparency.mlops import pipeline

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if api_base:
        cfg = cfg.model_copy(
            update={"public_data": cfg.public_data.model_copy(update={"api_base": api_base})}
        )
    cfg = _isolated(cfg, workdir)
    since = since or (date.today() - timedelta(days=180)).isoformat()
    rep = _Report()
    client = client_from_config(cfg)
    s = cfg.schema_

    def finish() -> dict[str, Any]:
        out = {"ok": rep.ok, "workdir": str(workdir), "steps": rep.steps}
        (workdir / "report.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        (workdir / "report.md").write_text(rep.markdown(), encoding="utf-8")
        return out

    def crashed(step: str, exc: BaseException, hint: str) -> dict[str, Any]:
        rep.add(
            step, "FAIL", hint=hint, error=repr(exc)[:500], trace=traceback.format_exc()[-3000:]
        )
        return finish()

    rep.add("environment", "INFO", **_environment())

    # 1. metadata (fast probe: one retry) ----------------------------------------
    fmap: dict[str, str] = {}
    retries, client.max_retries = client.max_retries, min(client.max_retries, 1)
    try:
        meta = client.metadata()
        fmap = {c["name"].strip(): c["fieldName"] for c in meta.get("columns", [])}
        rep.add(
            "api_metadata",
            "PASS",
            dataset=meta.get("name"),
            columns=len(fmap),
            rows_updated_at=meta.get("rowsUpdatedAt"),
        )
    except (DownloadError, OSError) as exc:
        try:  # metadata endpoint blocked? the data endpoint may still work
            fields = client.sample_fields()
            fmap = {d: f for d, f in api_field_map(cfg).items() if f in fields}
            rep.add(
                "api_metadata",
                "WARN",
                error=str(exc)[:300],
                fallback="field names from a sample record",
                hint="The metadata endpoint failed but data is reachable; continuing.",
            )
        except (DownloadError, OSError) as exc2:
            hint = (
                "Check internet access to www.datos.gov.co (proxy: set HTTPS_PROXY). "
                "For certificate errors see the message; set SSL_CERT_FILE to your CA bundle."
            )
            rep.add("api_metadata", "FAIL", hint=hint, error=str(exc2)[:500])
            return finish()
    finally:
        client.max_retries = retries

    # 2. schema ----------------------------------------------------------------
    required_missing = [c for c in (s.id, s.target) if c not in fmap]
    other_missing = [c for c in needed_columns(cfg) if c not in fmap and c not in (s.id, s.target)]
    rep.add(
        "schema_required",
        "FAIL" if required_missing else "PASS",
        missing=required_missing,
        hint="Set schema.id / schema.target to the dataset's current column names."
        if required_missing
        else None,
        available_columns=sorted(fmap)[:80] if required_missing else None,
    )
    rep.add(
        "schema_optional",
        "WARN" if other_missing else "PASS",
        missing=other_missing,
        hint="Those features are skipped. Update `schema:` / `output:` in the config if they were renamed."
        if other_missing
        else None,
    )
    expected = api_field_map(cfg)
    wrong = {c: fmap[c] for c, f in expected.items() if c in fmap and fmap[c] != f}
    rep.add(
        "json_field_names",
        "WARN" if wrong else "PASS",
        mismatches=wrong,
        hint="Add to public_data.api_field_overrides in the config: "
        + ", ".join(f'"{c}": "{f}"' for c, f in wrong.items())
        if wrong
        else None,
    )
    real_id_field = fmap.get(s.id)
    if real_id_field and real_id_field != cfg.public_data.id_api_field:
        rep.add(
            "id_api_field",
            "WARN",
            configured=cfg.public_data.id_api_field,
            live=real_id_field,
            hint=f"Set public_data.id_api_field: {real_id_field}",
        )
        cfg = cfg.model_copy(
            update={
                "public_data": cfg.public_data.model_copy(update={"id_api_field": real_id_field})
            }
        )
    else:
        rep.add("id_api_field", "PASS" if real_id_field else "FAIL", field=real_id_field)
    if required_missing:
        return finish()

    # 3. download --------------------------------------------------------------
    try:
        info = download_contracts(
            cfg, cfg.paths.raw_data, since=since, until=until, limit=limit, client=client
        )
    except Exception as exc:  # report, never crash
        return crashed(
            "download", exc, "The data query failed; see the error. Try a different --since."
        )
    n = info["rows"]
    status = "PASS" if n >= 500 else ("WARN" if n >= MIN_ROWS_TO_TRAIN else "FAIL")
    rep.add(
        "download",
        status,
        rows=n,
        where=info["where"],
        date_filter=info["date_filter"],
        hint=None
        if status == "PASS"
        else f"Only {n} rows: use an earlier --since or a larger --limit.",
    )
    if status == "FAIL":
        return finish()

    # 4. prepare ---------------------------------------------------------------
    try:
        data, report = pipeline.prepare(cfg)
    except Exception as exc:  # report, never crash
        return crashed("prepare", exc, "Cleaning failed on real data; please send report.md.")
    clean = report["clean"]
    rep.add(
        "prepare",
        "PASS" if clean["ok"] else "FAIL",
        stats=clean["stats"],
        warnings=clean["warnings"],
        errors=clean["errors"],
    )
    if not clean["ok"]:
        return finish()

    # 5. official records ------------------------------------------------------
    if fiscal_bulletin:
        try:
            _, man = import_records(
                cfg, "contraloria_responsables_fiscales", fiscal_bulletin, get_salt(cfg)
            )
            rep.add(
                "records_import",
                "PASS" if man["rows_with_valid_id"] else "FAIL",
                rows=man["rows_in_file"],
                valid_ids=man["rows_with_valid_id"],
                id_column=man["id_column"],
                date_column=man["date_column"],
            )
        except Exception as exc:  # report, never crash
            rep.add(
                "records_import",
                "FAIL",
                error=str(exc)[:400],
                hint="Pin records.sources.contraloria_responsables_fiscales.id_column in the config.",
            )

    # 6. train -----------------------------------------------------------------
    try:
        res = pipeline.train(cfg, data)
    except Exception as exc:  # report, never crash
        return crashed("train", exc, "Training failed on real data; please send report.md.")
    rep.add(
        "train",
        "PASS" if res.promoted else "WARN",
        version=res.version,
        reasons=res.reasons,
        test=res.metrics["test"],
        baseline=res.metrics["baseline_test"],
        weak_labels=res.metrics.get("weak_labels_test"),
        hint=None
        if res.promoted
        else "Quality gates not met on this small sample (expected with few rows).",
    )
    if not res.promoted:
        pipeline.promote(
            cfg, res.version, reason="live-check (gates not met, promoted for inspection)"
        )

    # 7. score -----------------------------------------------------------------
    try:
        scored, path = pipeline.score(cfg, data=data, source=report["source"])
    except Exception as exc:  # report, never crash
        return crashed("score", exc, "Scoring failed on real data; please send report.md.")
    rows = scored.attrs["manifest"]["rows"]
    coverage = rows["published"] / max(rows["input"], 1)
    rep.add(
        "score",
        "PASS" if coverage >= 0.95 else "WARN",
        output=str(path),
        link_coverage=round(coverage, 4),
        hint=None if coverage >= 0.95 else "Many contracts lack a valid SECOP link (URLProceso).",
        **rows,
    )

    # 8. audit links -------------------------------------------------------------
    broken, changed, duplicated, checked = [], [], [], 0
    target_field = fmap.get(s.target, "")
    sample = scored.dropna(subset=["contract_value"])
    for _, row in sample.sample(min(verify_links, len(sample)), random_state=0).iterrows():
        cid = row[s.id]
        try:
            found = client.record(cfg.public_data.id_api_field, cid)
        except DownloadError as exc:
            broken.append({"id": cid, "error": str(exc)[:200]})
            continue
        checked += 1
        if not found:
            broken.append({"id": cid, "records_found": 0})
            continue
        if len(found) > 1:
            duplicated.append({"id": cid, "records_found": len(found)})
        values = parse_money(pd.Series([r.get(target_field) for r in found]))
        if not np.isclose(values, row["contract_value"]).any():
            changed.append(
                {"id": cid, "api_values": values.tolist(), "output_value": row["contract_value"]}
            )
    status = "FAIL" if broken else ("WARN" if changed or duplicated else "PASS")
    hint = None
    if broken:
        hint = "Some open_data_url links return nothing: check public_data.id_api_field."
    elif changed or duplicated:
        hint = (
            "Links work. Differences come from the source itself (contract published more than once, "
            "or its value updated after the download)."
        )
    rep.add(
        "audit_links",
        status,
        checked=checked,
        broken=broken,
        value_changed=changed,
        duplicated_ids=duplicated,
        hint=hint,
    )
    return finish()
