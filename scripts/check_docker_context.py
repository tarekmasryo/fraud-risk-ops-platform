"""Verify Docker's actual context filtering using synthetic canaries and image files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
CANARIES = (
    "src/local-config.json",
    "tests/private.json",
    "scripts/local-secret.txt",
    "artifacts/local-model.joblib",
    "creditcard.csv",
    "data/private.csv",
    "ops.db",
    "ops.db-wal",
    "data/private.sqlite3",
    ".env",
    "src/.env",
    ".tmp/private.txt",
    "src/.tmp/private.txt",
    "tests/cache.tmp",
    "scripts/local-config.json",
    "src/unapproved.py",
    "tests/unapproved.py",
    "scripts/unapproved.py",
    "artifacts/private.json",
)


def authorized_files(root: Path = ROOT) -> list[str]:
    """Read exact authorizations; Docker remains the filtering implementation."""
    rules = [
        line.strip()
        for line in (root / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rules or rules[0] != "**":
        raise ValueError("Build context must default to deny")
    result = []
    for rule in rules[1:]:
        name = rule.removeprefix("!")
        path = PurePosixPath(name)
        if (
            not rule.startswith("!")
            or any(char in name for char in "*?[]\\")
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
            or not (root / name).is_file()
        ):
            raise ValueError(f"Authorization must name one existing file: {rule}")
        result.append(name)
    if set(result) & set(CANARIES):
        raise ValueError("Canary paths must never be authorized")
    return result


def digests(root: Path, names: list[str]) -> dict[str, str]:
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def verify_export(root: Path, expected: dict[str, str]) -> None:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(expected):
        raise AssertionError(
            f"Context files differ: missing={sorted(set(expected) - actual)}, "
            f"unexpected={sorted(actual - set(expected))}"
        )
    if digests(root, sorted(actual)) != expected:
        raise AssertionError("Context file content differs from authorized inputs")


IMAGE_CHECK = """
import hashlib, json, pathlib, sys
expected, forbidden = json.load(sys.stdin)
root = pathlib.Path('/app')
for name, digest in expected.items():
    path = root / name
    assert path.is_file(), f'Missing required image file: {name}'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'Changed image input: {name}'
for name in forbidden:
    assert not (root / name).exists(), f'Forbidden image file: {name}'
print(f'PASS: {len(expected)} required image inputs match; {len(forbidden)} forbidden paths absent')
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", help="Inspect already-built test/API/UI images")
    args = parser.parse_args()
    names = authorized_files()
    expected = digests(ROOT, names)
    if args.images:
        for image in args.images:
            print(f"Inspecting {image}", flush=True)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-i",
                    "--network=none",
                    "--entrypoint",
                    "python",
                    image,
                    "-c",
                    IMAGE_CHECK,
                ],
                input=json.dumps([expected, CANARIES]),
                text=True,
                check=True,
                timeout=120,
            )
        return
    # Never copy real local state or secrets. Only reviewed inputs and fictional canaries.
    with tempfile.TemporaryDirectory(prefix="fraud-context-") as temporary:
        workspace = Path(temporary)
        context = workspace / "input"
        for name in names:
            target = context / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        for name in CANARIES:
            target = context / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("DISPOSABLE CONTEXT CANARY - NO PRIVATE DATA\n")
        output = workspace / "output"
        subprocess.run(
            [
                "docker",
                "build",
                "--file",
                "-",
                "--output",
                f"type=local,dest={output}",
                str(context),
            ],
            input="FROM scratch\nCOPY . /context/\n",
            text=True,
            check=True,
            timeout=300,
        )
        verify_export(output / "context", expected)
        print(
            f"PASS: Docker exported exactly {len(expected)} authorized files with matching hashes; "
            f"all {len(CANARIES)} canaries excluded"
        )


if __name__ == "__main__":
    main()
