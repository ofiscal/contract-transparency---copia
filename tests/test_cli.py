import json

import pytest
from typer.testing import CliRunner

from contract_transparency.cli import app
from tests.conftest import write_cli_config

runner = CliRunner()


def test_help():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0 and "demo-data" in res.output


def test_missing_config():
    res = runner.invoke(app, ["models", "-c", "does/not/exist.yaml"])
    assert res.exit_code != 0


@pytest.mark.slow
def test_cli_end_to_end(tmp_path):
    cfg = write_cli_config(tmp_path)
    assert (
        runner.invoke(
            app, ["demo-data", "--rows", "2500", "--output", str(tmp_path / "raw.csv")]
        ).exit_code
        == 0
    )
    res = runner.invoke(app, ["validate", "-c", str(cfg)])
    assert res.exit_code == 0 and json.loads(res.output)["ok"]

    res = runner.invoke(app, ["run", "-c", str(cfg)])
    assert res.exit_code == 0, res.output
    assert "promoted=True" in res.output

    res = runner.invoke(app, ["models", "-c", str(cfg)])
    assert "v0001" in res.output and "True" in res.output

    # default input = processed snapshot (raw file moved away to prove it is not re-read)
    raw, moved = tmp_path / "raw.csv", tmp_path / "raw_moved.csv"
    raw.rename(moved)
    out = tmp_path / "scored.csv"
    res = runner.invoke(app, ["score", "-c", str(cfg), "--output", str(out), "--top", "3"])
    assert res.exit_code == 0 and out.exists(), res.output
    # explicit raw extract
    out2 = tmp_path / "scored_raw.csv"
    res = runner.invoke(
        app, ["score", "-c", str(cfg), "--input", str(moved), "--output", str(out2)]
    )
    assert res.exit_code == 0 and out2.exists(), res.output
    moved.rename(raw)

    res = runner.invoke(app, ["train", "-c", str(cfg)])
    assert res.exit_code == 0 and json.loads(res.output)["version"] == "v0002"
    res = runner.invoke(app, ["promote", "v0001", "-c", str(cfg), "--reason", "rollback"])
    assert res.exit_code == 0 and "v0001 is now the champion" in res.output

    res = runner.invoke(app, ["suppliers", "-c", str(cfg), "--top", "5"])
    assert res.exit_code == 0 and "supplier_pseudonym" in res.output
    res = runner.invoke(app, ["drift", "-c", str(cfg)])
    assert res.exit_code == 0 and "Retrain recommended" in res.output


def test_all_commands_are_registered():
    """Guards against a command being dropped by an edit (it happened once)."""
    res = runner.invoke(app, ["--help"])
    expected = {
        "demo-data",
        "download",
        "validate",
        "prepare",
        "train",
        "models",
        "promote",
        "score",
        "drift",
        "update",
        "run",
        "live-check",
        "doctor",
        "suppliers",
        "sustainability",
        "app",
        "records",
        "benchmark",
    }
    missing = {c for c in expected if c not in res.output}
    assert not missing, missing
