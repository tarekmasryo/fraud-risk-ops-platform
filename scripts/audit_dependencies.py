"""Audit the installed environment; allow only reviewed, version-scoped exceptions."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def unaccepted_findings(report: dict, exceptions: list[dict], today: date) -> list[str]:
    failures = []
    for exception in exceptions:
        if date.fromisoformat(exception["expires"]) <= today:
            failures.append(f"Expired security exception: {exception['id']}")
    dependencies = report["dependencies"]
    if not dependencies:
        raise ValueError("Audit returned no dependencies")
    for dependency in dependencies:
        if "skip_reason" in dependency:
            # The local source project has no published advisory identity. No third-party
            # editable or otherwise unaudited distribution is exempted here.
            if (
                dependency["name"] == "fraud-risk-ops-platform"
                and dependency["skip_reason"] == "distribution marked as editable"
            ):
                continue
            failures.append(f"Unaudited dependency: {dependency['name']}")
            continue
        for vulnerability in dependency["vulns"]:
            identifiers = {vulnerability["id"], *vulnerability.get("aliases", [])}
            accepted = any(
                exception["id"] in identifiers
                and exception["package"] == dependency["name"]
                and exception["version"] == dependency["version"]
                and date.fromisoformat(exception["expires"]) > today
                for exception in exceptions
            )
            label = f"{dependency['name']}=={dependency['version']}: {vulnerability['id']}"
            if accepted:
                print(f"Reviewed exception: {label}")
            else:
                failures.append(label)
    return failures


def main() -> int:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip_audit",
            "--local",
            "--skip-editable",
            "--format=json",
            "--progress-spinner=off",
            "--cache-dir",
            str(ROOT / ".cache" / "pip-audit"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode not in (0, 1):
        return result.returncode
    try:
        report = json.loads(result.stdout)
        exceptions = json.loads((ROOT / "security-exceptions.json").read_text(encoding="utf-8"))
        failures = unaccepted_findings(report, exceptions, date.today())
        # Exit 1 without findings is an audit failure, not a clean environment.
        if result.returncode and not any(d.get("vulns") for d in report["dependencies"]):
            failures.append("pip-audit failed without reporting vulnerabilities")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Security audit could not be evaluated: {exc}", file=sys.stderr)
        return 1
    for failure in failures:
        print(f"BLOCKED: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
