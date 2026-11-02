from datetime import date
from types import SimpleNamespace

import pytest

from scripts import audit_dependencies
from scripts.audit_dependencies import unaccepted_findings

TODAY = date(2026, 11, 2)
EXCEPTION = {
    "id": "CVE-2024-5206",
    "package": "scikit-learn",
    "version": "1.4.2",
    "expires": "2026-12-02",
}


def report(package="scikit-learn", version="1.4.2", identifier="PYSEC-2024-110"):
    return {
        "dependencies": [
            {
                "name": package,
                "version": version,
                "vulns": [{"id": identifier, "aliases": ["CVE-2024-5206"]}],
            }
        ]
    }


def test_reviewed_alias_is_accepted():
    assert unaccepted_findings(report(), [EXCEPTION], TODAY) == []


@pytest.mark.parametrize("changes", [{"package": "pillow"}, {"version": "1.4.1"}])
def test_exception_is_scoped_to_package_and_version(changes):
    assert unaccepted_findings(report(**changes), [EXCEPTION], TODAY)


def test_unrelated_vulnerability_is_blocked():
    data = report()
    data["dependencies"][0]["vulns"] = [{"id": "new-vulnerability"}]
    assert unaccepted_findings(data, [EXCEPTION], TODAY)


def test_expiry_blocks_even_when_vulnerability_disappears():
    data = report()
    data["dependencies"][0]["vulns"] = []
    assert unaccepted_findings(data, [EXCEPTION], date(2026, 12, 2))


def test_incomplete_audit_is_blocked():
    assert unaccepted_findings(
        {"dependencies": [{"name": "unavailable", "skip_reason": "lookup failed"}]}, [], TODAY
    )
    with pytest.raises(ValueError, match="no dependencies"):
        unaccepted_findings({"dependencies": []}, [], TODAY)


def test_only_the_local_editable_project_is_excluded():
    data = {
        "dependencies": [
            {"name": "fraud-risk-ops-platform", "skip_reason": "distribution marked as editable"}
        ]
    }
    assert unaccepted_findings(data, [], TODAY) == []
    data["dependencies"][0]["name"] = "third-party-editable"
    assert unaccepted_findings(data, [], TODAY)


@pytest.mark.parametrize(
    ("returncode", "output"),
    [(2, ""), (1, "not json"), (1, '{"dependencies": [{"name": "safe", "vulns": []}]}')],
)
def test_audit_process_failures_are_blocking(monkeypatch, returncode, output):
    monkeypatch.setattr(
        audit_dependencies.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=returncode, stdout=output, stderr="audit failure"
        ),
    )
    assert audit_dependencies.main() != 0
