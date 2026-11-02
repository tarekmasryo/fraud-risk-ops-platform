"""Test gate failure detection; actual Docker semantics run in the CI Docker job."""

import shutil

import pytest

from scripts.check_docker_context import CANARIES, ROOT, authorized_files, digests, verify_export


def test_required_build_inputs_are_explicitly_authorized():
    names = set(authorized_files())
    required = {
        "Dockerfile",
        ".dockerignore",
        "docker-compose.yml",
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "pytest.ini",
        ".coveragerc",
        "README.md",
        "LICENSE",
        "security-exceptions.json",
        "app.py",
        "api.py",
        "streamlit_app.py",
        "tests/fixtures/reference_predictions.json",
    }
    for directory in ("src", "tests", "scripts"):
        required.update(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / directory).rglob("*.py")
            if "__pycache__" not in p.parts
        )
    required.update(
        f"artifacts/{name}"
        for name in (
            "metadata.json",
            "policy.json",
            "thresholds.json",
            "rf_calibrated.joblib",
            "xgb_calibrated.joblib",
            "rf_pipe.joblib",
            "xgb_pipe.joblib",
        )
    )
    assert required == names
    assert not names.intersection(CANARIES)


@pytest.mark.parametrize("rule", ["!src/", "!src/**/", "!tests/**/*.py", "!../outside"])
def test_gate_rejects_broad_or_invalid_authorizations(tmp_path, rule):
    (tmp_path / ".dockerignore").write_text(f"**\n{rule}\n")
    with pytest.raises(ValueError, match="one existing file"):
        authorized_files(tmp_path)


@pytest.mark.parametrize("failure", ["none", "unexpected", "missing", "changed"])
def test_export_verifier_detects_filtering_or_content_failure(tmp_path, failure):
    name = "app.py"
    shutil.copyfile(ROOT / name, tmp_path / name)
    expected = digests(ROOT, [name])
    if failure == "unexpected":
        (tmp_path / "private.json").write_text("fictional canary")
    elif failure == "missing":
        (tmp_path / name).unlink()
    elif failure == "changed":
        (tmp_path / name).write_text("changed")
    if failure == "none":
        verify_export(tmp_path, expected)
    else:
        with pytest.raises(AssertionError):
            verify_export(tmp_path, expected)
