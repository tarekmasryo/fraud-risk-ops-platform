"""Exercise Git checkout conversion, including Windows autocrlf behavior."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("git"), "Git is required to exercise checkout conversion")
class ArtifactCheckoutTest(unittest.TestCase):
    def test_autocrlf_checkout_preserves_manifest_protected_bytes(self) -> None:
        metadata = json.loads((ROOT / "artifacts/metadata.json").read_text(encoding="utf-8"))
        expected = metadata["artifact_integrity"]["expected_sha256"]

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            checkout = Path(directory) / "checkout"
            repository.mkdir()
            checkout.mkdir()

            def git(*arguments: str) -> None:
                subprocess.run(
                    ["git", "-C", str(repository), *arguments],
                    check=True,
                    capture_output=True,
                )

            git("init", "--quiet")
            git("config", "core.autocrlf", "true")
            git("config", "core.safecrlf", "false")
            shutil.copyfile(ROOT / ".gitattributes", repository / ".gitattributes")
            (repository / "artifacts").mkdir()
            for name in expected:
                shutil.copyfile(ROOT / "artifacts" / name, repository / "artifacts" / name)
            git("add", ".")
            git("checkout-index", "--all", f"--prefix={checkout.as_posix()}/")

            for name, digest in expected.items():
                with self.subTest(artifact=name):
                    artifact_bytes = (checkout / "artifacts" / name).read_bytes()
                    actual = hashlib.sha256(artifact_bytes).hexdigest()
                    self.assertEqual(actual, digest, f"Git checkout changed {name}")


if __name__ == "__main__":
    unittest.main()
