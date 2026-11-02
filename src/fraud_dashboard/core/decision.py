from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def stable_input_hash(record: dict[str, Any]) -> str:
    """Return a stable privacy-preserving hash for an input record."""

    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def risk_band(score: float) -> str:
    value = float(score)
    if value >= 0.90:
        return "critical"
    if value >= 0.65:
        return "high"
    if value >= 0.20:
        return "medium"
    return "low"


def decision_from_score(score: float, threshold: float) -> tuple[str, bool]:
    """Map a risk score to the public reference decision contract."""

    review_required = float(score) >= float(threshold)
    return ("review" if review_required else "approve", review_required)


def reason_codes_for_record(record: dict[str, Any], *, score: float, threshold: float) -> list[str]:
    """Generate deterministic operational reason hints.

    These are not model explanations. They are simple review hints that make the
    public release auditable without claiming SHAP/explainability support.
    """

    codes: list[str] = []
    amount = pd.to_numeric(pd.Series([record.get("Amount", 0.0)]), errors="coerce").fillna(0.0)
    amount_value = float(amount.iloc[0])
    if amount_value >= 1000.0:
        codes.append("amount_high")
    elif amount_value >= 250.0:
        codes.append("amount_elevated")

    if float(score) >= float(threshold):
        codes.append("score_above_policy_threshold")
    else:
        codes.append("score_below_policy_threshold")

    codes.append(f"risk_band_{risk_band(float(score))}")
    return codes
