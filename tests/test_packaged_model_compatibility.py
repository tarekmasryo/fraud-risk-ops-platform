"""Regression evidence for the unchanged reference models and security exception."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_dashboard.core.artifacts import load_bundle, project_root


def _fitted_objects(value):
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, np.ndarray) and item.dtype == object:
            pending.extend(item.flat)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            pending.extend(vars(item).values())


def test_packaged_predictions_and_text_vectorizer_exception():
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "reference_predictions.json").read_text()
    )
    for name, expected in reference["sha256"].items():
        assert (
            hashlib.sha256((project_root() / "artifacts" / name).read_bytes()).hexdigest()
            == expected
        )
    bundle = load_bundle()
    assert bundle["schema"]["features"] == reference["features"]
    frame = pd.DataFrame(reference["records"], columns=reference["features"])
    assert set(bundle["models"]) == set(reference["probabilities"])
    for family, model in bundle["models"].items():
        for fitted in _fitted_objects(model):
            assert not type(fitted).__module__.startswith("sklearn.feature_extraction.text"), (
                "Reassess CVE-2024-5206 exception: a packaged model contains text feature extraction"
            )
        actual = model.predict_proba(frame)[:, 1]
        assert np.isfinite(actual).all()
        assert ((actual >= 0) & (actual <= 1)).all()
        np.testing.assert_allclose(actual, reference["probabilities"][family], rtol=1e-7, atol=1e-9)
