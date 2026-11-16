"""Command-line interface: ``ct <command> --help``.

Typical flows::

    ct demo-data --with-records             # synthetic data, no downloads
    ct run -c configs/demo.yaml             # prepare -> train -> promote -> score

    ct download --since 2025-01-01          # real SECOP II from datos.gov.co
    ct records import contraloria_responsables_fiscales boletin.xlsx
    ct prepare && ct train && ct score
    ct update data/raw/secop_2025_07.csv    # incremental: drift -> retrain? -> score
    ct live-check                           # verify everything against the live sources
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import typer

from contract_transparency.config import DEFAULT_CONFIG_PATH, Settings, load_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Contract transparency: statistical alerts for the review of SECOP II public contracts.",
)
records_app = typer.Typer(
    no_args_is_help=True, help="Official public records (Contraloría, SECOP sanctions)."
)
app.add_typer(records_app, name="records")

ConfigOpt = typer.Option(
    None, "--config", "-c", help="YAML config (default: $CT_CONFIG or configs/secop_ii.yaml)"
)


def _cfg(path: Path | None) -> Settings:
    path = path or Path(os.environ.get("CT_CONFIG", DEFAULT_CONFIG_PATH))
    if not Path(path).exists():
        raise typer.BadParameter(f"Config file not found: {path}")
    return load_config(path)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("matplotlib", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _echo_json(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _notice(cfg: Settings) -> None:
    """Printed after every command that produces results."""
    if cfg.public_data.synthetic:
        typer.secho(
            "DATOS SINTÉTICOS - NO REALES / SYNTHETIC DATA - NOT REAL", fg="yellow", bold=True
        )
    typer.secho(cfg.public_data.notice_es, fg="yellow")


# --------------------------------------------------------------------------- data
@app.command("demo-data")
def demo_data(
    rows: int = typer.Option(20_000, help="Number of synthetic contracts"),
    output: Path = typer.Option(Path("data/raw/demo_contracts.csv"), help="Output CSV"),
    seed: int = typer.Option(24111996),
    start: str = typer.Option("2021-01-01"),
    end: str = typer.Option("2023-12-31"),
    with_records: bool = typer.Option(
        False, help="Also write a synthetic Contraloría-like bulletin and import it"
    ),
    config: Path = ConfigOpt,
) -> None:
    """Generate a synthetic SECOP II-like dataset (fictitious people and companies)."""
    from contract_transparency.data.synthetic import generate_contracts, generate_fiscal_bulletin

    df = generate_contracts(rows, seed=seed, start=start, end=end)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    typer.echo(f"Wrote {len(df):,} synthetic contracts to {output}")
    if with_records:
        from contract_transparency.data.privacy import get_salt
        from contract_transparency.data.records import import_records

        bulletin = output.parent / "demo_boletin_responsables_fiscales.xlsx"
        generate_fiscal_bulletin(df, n=40, seed=seed).to_excel(bulletin, index=False, header=False)
        cfg = _cfg(config or Path("configs/demo.yaml"))
        _, man = import_records(cfg, "contraloria_responsables_fiscales", bulletin, get_salt(cfg))
        typer.echo(
            f"Wrote and imported synthetic bulletin {bulletin} ({man['rows_with_valid_id']} records)"
        )


@app.command()
def download(
    config: Path = ConfigOpt,
    since: str = typer.Option(None, help="Signed on/after YYYY-MM-DD"),
    until: str = typer.Option(None, help="Signed before YYYY-MM-DD"),
    limit: int = typer.Option(None, help="Max rows (default: all)"),
    output: Path = typer.Option(None, help="CSV path (default: paths.raw_data)"),
    api_base: str = typer.Option(None, help="Override the API base URL"),
) -> None:
    """Download SECOP II contracts from the datos.gov.co API (no personal-data columns)."""
    from contract_transparency.data.download import client_from_config, download_contracts

    cfg = _cfg(config)
    target = output or Path(cfg.paths.raw_data)
    info = download_contracts(cfg, target, since, until, limit, client_from_config(cfg, api_base))
    typer.echo(
        f"Downloaded {info['rows']:,} contracts to {target} (provenance: {target}.source.json)"
    )


@app.command()
def validate(
    config: Path = ConfigOpt,
    input: Path = typer.Option(None, help="Raw file (default: paths.raw_data)"),
    nrows: int = typer.Option(None),
) -> None:
    """Check that a raw extract matches the configured schema (no files written)."""
    from contract_transparency.data.io import read_contracts
    from contract_transparency.data.validation import validate_raw

    cfg = _cfg(config)
    raw = read_contracts(input or cfg.paths.raw_data, cfg, nrows)
    rep = validate_raw(raw, cfg)
    _echo_json(rep.to_dict())
    raise typer.Exit(code=0 if rep.ok else 1)


@app.command()
def prepare(
    config: Path = ConfigOpt, input: Path = typer.Option(None), nrows: int = typer.Option(None)
) -> None:
    """Read + anonymise + clean + validate a raw extract and save a snapshot."""
    from contract_transparency.mlops import pipeline

    _, report = pipeline.prepare(_cfg(config), input, nrows)
    _echo_json({"source": report["source"], "clean": report["clean"]})


@records_app.command("import")
def records_import(
    source: str = typer.Argument(..., help="Record source, e.g. contraloria_responsables_fiscales"),
    file: Path = typer.Argument(..., help="File downloaded from the official publication"),
    config: Path = ConfigOpt,
) -> None:
    """Import an official record file (keeps only pseudonymised IDs + dates)."""
    from contract_transparency.data.privacy import get_salt
    from contract_transparency.data.records import import_records

    cfg = _cfg(config)
    _, manifest = import_records(cfg, source, file, get_salt(cfg))
    _echo_json(manifest)


@records_app.command("list")
def records_list(config: Path = ConfigOpt) -> None:
    """Configured record sources and which ones have been imported."""
    from contract_transparency.data.records import load_records

    cfg = _cfg(config)
    loaded = load_records(cfg)
    for name, src in cfg.records.sources.items():
        status = (
            f"imported ({len(loaded[name][0])} keys, file {loaded[name][1]['file']})"
            if name in loaded
            else "not imported"
        )
        typer.echo(f"- {name}: {src.title} - {src.publisher} [{status}]\n    {src.official_url}")


# --------------------------------------------------------------------------- models
@app.command()
def train(config: Path = ConfigOpt) -> None:
    """Train a new model version; auto-promote it if it passes the quality gates."""
    from contract_transparency.mlops import pipeline

    res = pipeline.train(_cfg(config))
    _echo_json(
        {
            "version": res.version,
            "promoted": res.promoted,
            "reasons": res.reasons,
            "test": res.metrics["test"],
            "baseline_test": res.metrics["baseline_test"],
            "detection_test": res.metrics.get("detection_test"),
            "weak_labels_test": res.metrics.get("weak_labels_test"),
        }
    )


@app.command()
def models(config: Path = ConfigOpt) -> None:
    """List registered model versions (champion marked)."""
    from contract_transparency.mlops.pipeline import get_registry

    summary = get_registry(_cfg(config)).summary()
    typer.echo("Registry is empty" if summary.empty else summary.to_string(index=False))


@app.command()
def promote(version: str, config: Path = ConfigOpt, reason: str = typer.Option("manual")) -> None:
    """Make VERSION the champion (also used for rollbacks)."""
    from contract_transparency.mlops import pipeline

    pipeline.promote(_cfg(config), version, reason)
    typer.echo(f"{version} is now the champion")


# --------------------------------------------------------------------------- results
@app.command()
def score(
    config: Path = ConfigOpt,
    input: Path = typer.Option(
        None, help="Raw extract to score (default: the processed snapshot from `ct prepare`)"
    ),
    version: str = typer.Option("champion"),
    output: Path = typer.Option(None),
    top: int = typer.Option(10, help="Rows to print"),
) -> None:
    """Score contracts: expected price, statistical alerts and alert index for review."""
    from contract_transparency.mlops import pipeline

    cfg = _cfg(config)
    data, source = None, None
    if input is None and pipeline.processed_path(cfg).exists():
        data, source = pipeline.load_processed(cfg), pipeline.snapshot_source(cfg)
    scored, path = pipeline.score(
        cfg, input_path=input, data=data, version=version, output_path=output, source=source
    )
    cols = [
        "review_priority",
        cfg.schema_.id,
        "contract_value",
        "predicted_value",
        "alert_index",
        "secop_url",
    ]
    typer.echo(scored[[c for c in cols if c in scored]].head(top).to_string(index=False))
    rows = scored.attrs["manifest"]["rows"]
    typer.echo(
        f"\n{rows['price_deviations']} price alerts / {rows['published']:,} published contracts "
        f"({rows['excluded_not_auditable']} without official link excluded) -> {path}\n"
        f"Provenance & notice: {path}.manifest.json"
    )
    _notice(cfg)


@app.command()
def drift(
    config: Path = ConfigOpt,
    input: Path = typer.Option(None),
    version: str = typer.Option("champion"),
) -> None:
    """Population Stability Index of new data vs. the model's training data."""
    from contract_transparency.mlops import pipeline

    cfg = _cfg(config)
    data, _ = pipeline.prepare(cfg, input, save=False)
    report, retrain = pipeline.check_drift(cfg, data, version)
    typer.echo(report.to_string(index=False))
    typer.echo(f"\nRetrain recommended: {retrain}")


@app.command()
def update(
    new_input: Path, config: Path = ConfigOpt, force_retrain: bool = typer.Option(False)
) -> None:
    """Incremental update with a new extract: new/modified contracts, drift, retrain, score."""
    from contract_transparency.mlops import pipeline

    cfg = _cfg(config)
    summary = pipeline.update(cfg, new_input, force_retrain)
    summary.pop("drift", None)
    _echo_json(summary)
    _notice(cfg)


@app.command()
def run(
    config: Path = ConfigOpt, input: Path = typer.Option(None), nrows: int = typer.Option(None)
) -> None:
    """Full pipeline: prepare -> train -> (promote) -> score."""
    from contract_transparency.mlops import pipeline

    cfg = _cfg(config)
    data, report = pipeline.prepare(cfg, input, nrows)
    res = pipeline.train(cfg, data)
    typer.echo(f"Model {res.version} promoted={res.promoted}: {'; '.join(res.reasons)}")
    scored, path = pipeline.score(cfg, data=data, source=report["source"])
    rows = scored.attrs["manifest"]["rows"]
    typer.echo(
        f"{rows['price_deviations']} price alerts / {rows['published']:,} published contracts -> {path}"
    )
    _notice(cfg)


@app.command("live-check")
def live_check_cmd(
    config: Path = ConfigOpt,
    workdir: Path = typer.Option(Path("data/live_check")),
    since: str = typer.Option(None, help="Default: 180 days ago"),
    until: str = typer.Option(None),
    limit: int = typer.Option(20_000),
    fiscal_bulletin: Path = typer.Option(None, help="Contraloría bulletin file to import"),
    verify_links: int = typer.Option(5),
    api_base: str = typer.Option(None, help="Override the API base URL"),
) -> None:
    """Verify the complete pipeline against the LIVE public sources (needs internet)."""
    from contract_transparency.mlops.live_check import live_check

    out = live_check(
        _cfg(config), workdir, since, until, limit, fiscal_bulletin, verify_links, api_base
    )
    colors = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "INFO": "blue"}
    for step in out["steps"]:
        typer.secho(f"[{step['status']:4}] {step['step']}", fg=colors.get(step["status"]))
        if step["status"] in {"WARN", "FAIL"}:
            for key in (
                "error",
                "missing",
                "mismatches",
                "broken",
                "value_changed",
                "errors",
                "rows",
            ):
                if step.get(key):
                    typer.echo(
                        f"       {key}: {json.dumps(step[key], ensure_ascii=False, default=str)[:400]}"
                    )
            if step.get("hint"):
                typer.secho(f"       -> {step['hint']}", fg="yellow")
    typer.echo(f"\nOverall: {'PASS' if out['ok'] else 'FAIL'} - full report: {workdir}/report.md")
    if not out["ok"]:
        typer.echo(
            "If the cause is unclear, please share that file (it contains no personal data)."
        )
    raise typer.Exit(code=0 if out["ok"] else 1)


@app.command()
def doctor(
    config: Path = ConfigOpt,
    network: bool = typer.Option(False, help="Also check access to the datos.gov.co API"),
) -> None:
    """Check the environment after installing/upgrading packages (versions, model, folders)."""
    from contract_transparency.environment import doctor as run_doctor

    rep = run_doctor(_cfg(config), network=network)
    colors = {"PASS": "green", "WARN": "yellow", "FAIL": "red", "INFO": "blue"}
    for c in rep.checks:
        typer.secho(f"[{c.status:4}] {c.name}: {c.detail}", fg=colors[c.status])
        if c.hint:
            typer.echo(f"       -> {c.hint}")
    typer.echo(f"\nOverall: {'OK' if rep.ok else 'PROBLEMS FOUND'}")
    raise typer.Exit(code=0 if rep.ok else 1)


@app.command()
def benchmark(
    config: Path = ConfigOpt,
    variants: str = typer.Option(None, help="Comma-separated variants (default: all CPU variants)"),
    nrows: int = typer.Option(None, help="Use only the first N contracts of the snapshot"),
    output: Path = typer.Option(None, help="Markdown report (default: <outputs_dir>/benchmark.md)"),
) -> None:
    """Compare model architectures on accuracy, detection and cost (uses the processed snapshot)."""
    from contract_transparency.mlops import pipeline
    from contract_transparency.mlops.benchmark import VARIANTS, run_benchmark, to_markdown

    cfg = _cfg(config)
    data = pipeline.load_processed(cfg)
    if nrows:
        data = data.head(nrows)
    chosen = [v.strip() for v in variants.split(",")] if variants else None
    result = run_benchmark(cfg, data, chosen)
    md = to_markdown(result)
    legend = "\n".join(f"- `{k}`: {d}" for k, (d, _) in VARIANTS.items())
    path = output or Path(cfg.paths.outputs_dir) / "benchmark.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Benchmark ({len(data):,} contracts)\n\n{md}\n\n{legend}\n", encoding="utf-8"
    )
    typer.echo(md)
    typer.echo(f"\nReport: {path}")


# --------------------------------------------------------------------------- analysis
@app.command()
def suppliers(
    config: Path = ConfigOpt,
    since: str = typer.Option(None, help="YYYY-MM-DD"),
    top: int = typer.Option(20),
    keep_universities: bool = typer.Option(False),
) -> None:
    """Ranking of suppliers by contracted value (natural persons masked)."""
    from contract_transparency.analysis.contractors import DEFAULT_EXCLUDE, top_suppliers
    from contract_transparency.mlops.pipeline import load_processed

    df = load_processed(_cfg(config))
    res = top_suppliers(
        df, n=top, since=since, exclude_regex=None if keep_universities else DEFAULT_EXCLUDE
    )
    typer.echo(res.to_string(index=False))


@app.command()
def sustainability(
    config: Path = ConfigOpt,
    limit: int = typer.Option(500),
    output: Path = typer.Option(None),
) -> None:
    """Zero-shot sustainability tags for contract descriptions (needs `.[nlp]`)."""
    from contract_transparency.analysis.sustainability import classify_sustainability
    from contract_transparency.data import columns as C
    from contract_transparency.data.io import write_table
    from contract_transparency.mlops.pipeline import load_processed

    cfg = _cfg(config)
    df = load_processed(cfg).head(limit)
    tags = classify_sustainability(df[C.TEXT].tolist())
    tags.insert(0, "ID Contrato", df[C.ID].to_numpy())
    path = write_table(
        tags,
        output or Path(cfg.paths.outputs_dir) / "sustainability",
        cfg.output.format,
        cfg.output.csv_sep,
    )
    typer.echo(f"Wrote {len(tags)} rows to {path}")


@app.command("app")
def run_app(
    config: Path = ConfigOpt,
    port: int = typer.Option(8501),
    address: str = typer.Option("localhost", help="Use 0.0.0.0 inside Docker"),
) -> None:
    """Launch the Streamlit dashboard."""
    app_file = Path(__file__).parent / "app" / "streamlit_app.py"
    env = dict(os.environ)
    if config:
        env["CT_CONFIG"] = str(config)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_file)]
    cmd += ["--server.port", str(port), "--server.address", address]
    # headless: no interactive first-run prompt; open the URL yourself.
    cmd += ["--server.headless", "true", "--browser.gatherUsageStats", "false"]
    typer.echo(f"Dashboard on http://localhost:{port}  (Ctrl+C to stop)")
    raise typer.Exit(subprocess.call(cmd, env=env))


def main() -> None:
    """Console entry point: expected problems are shown as a short message
    (set CT_DEBUG=1 for the full traceback)."""
    from pydantic import ValidationError

    from contract_transparency.data.download import DownloadError
    from contract_transparency.environment import ModelCompatibilityError
    from contract_transparency.mlops.registry import IntegrityError

    expected = (
        ModelCompatibilityError,
        IntegrityError,
        DownloadError,
        LookupError,
        FileNotFoundError,
        ValidationError,
        ValueError,
    )
    try:
        app()
    except expected as exc:
        if os.environ.get("CT_DEBUG"):
            raise
        typer.secho(f"Error: {exc}", fg="red", err=True)
        typer.echo("(set CT_DEBUG=1 to see the full traceback)", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
