"""Packaging & dependency policy (static checks; no Docker or network needed)."""

import re
import shutil
import subprocess

import pytest
import yaml

from tests.conftest import REPO


def _pyproject() -> str:
    return (REPO / "pyproject.toml").read_text()


def _core_dependencies() -> list[str]:
    # (tomllib is Python >= 3.11; CI also runs 3.10)
    block = re.search(r"^dependencies = \[(.*?)^\]", _pyproject(), re.M | re.S)
    return re.findall(r'"([^"]+)"', block.group(1))


def test_dependencies_have_minimums_and_no_upper_caps():
    """Flexibility policy: upgrading a package must never be blocked by this
    project. Minimums only (tested by `make test-lowest`), no `<`, `<=`, `==`, `~=`."""
    for dep in _core_dependencies():
        assert ">=" in dep, f"{dep}: declare a tested minimum version"
        assert not re.search(r"<|==|~=|!=", dep), f"{dep}: no upper caps / exact pins"


def test_uv_lock_present_and_in_sync():
    lock = REPO / "uv.lock"
    assert lock.exists(), "commit uv.lock (the known-good set of versions)"
    names = {re.split(r"[<>=\[ ]", d)[0].lower() for d in _core_dependencies()}
    locked = set(re.findall(r'^name = "([^"]+)"', lock.read_text(), re.M))
    assert names <= locked
    if shutil.which("uv"):
        subprocess.run(["uv", "lock", "--check"], cwd=REPO, check=True, capture_output=True)


def test_image_installs_the_lock_as_non_root():
    d = (REPO / "Dockerfile").read_text()
    assert re.search(r"^USER\s+ct\s*$", d, re.M)
    assert "uv sync --frozen --no-dev" in d
    assert "HEALTHCHECK" in d and 'ENTRYPOINT ["ct"]' in d


def test_image_never_receives_data_models_or_secrets():
    ignore = {line.strip() for line in (REPO / ".dockerignore").read_text().splitlines()}
    for must in ("data", "models", ".secrets", ".env", "legacy", ".git", "mlruns"):
        assert must in ignore, must
    copies = re.findall(r"^COPY\s+(.+)$", (REPO / "Dockerfile").read_text(), re.M)
    sources = {tok for line in copies for tok in line.split()[:-1]}
    assert sources <= {"pyproject.toml", "uv.lock", "README.md", "LICENSE", "src", "configs"}


def test_compose_binds_dashboard_to_localhost_and_uses_volumes():
    compose = yaml.safe_load((REPO / "docker-compose.yml").read_text())
    assert all(p.startswith("127.0.0.1:") for p in compose["services"]["app"]["ports"])
    vols = " ".join(compose["x-common"]["volumes"])
    for v in ("./data:/app/data", "./models:/app/models", "./.secrets:/app/.secrets"):
        assert v in vols


def test_makefile_uses_uv_and_docker_is_optional():
    mk = (REPO / "Makefile").read_text()
    for target in ("install", "demo", "app", "test", "upgrade", "test-lowest", "test-latest"):
        assert re.search(rf"^{target}:", mk, re.M), target
    recipes = re.findall(r"^([a-z-]+):.*?\n((?:\t.*\n?)*)", mk, re.M)
    for name, body in recipes:
        if "docker" in body and not name.startswith("docker"):
            pytest.fail(f"target {name} requires docker")


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")
def test_setup_scripts_never_install_system_wide():
    subprocess.run(["bash", "-n", str(REPO / "scripts" / "setup.sh")], check=True)
    for script in ("setup.sh", "setup.ps1"):
        text = (REPO / "scripts" / script).read_text()
        assert "--break-system-packages" not in text and "sudo pip" not in text
        assert "uv sync" in text
