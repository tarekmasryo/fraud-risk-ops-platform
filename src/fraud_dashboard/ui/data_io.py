from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fraud_dashboard.data.synthetic import SyntheticCreditcardSpec, generate_synthetic_creditcard
from fraud_dashboard.data.validation import coerce_numeric, validate_finite, validate_no_nan

DEFAULT_DATA_PATHS = [
    "data/demo_creditcard.csv",
    "creditcard.csv",
    "data/creditcard.csv",
    "/mnt/data/creditcard.csv",
]


def read_csv_any(file) -> pd.DataFrame:
    name = getattr(file, "name", "") or ""
    data = file.read() if hasattr(file, "read") else file
    buf = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else None
    if isinstance(file, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(file))
    if name.endswith(".gz") and buf is not None:
        buf.seek(0)
        return pd.read_csv(buf, compression="gzip")
    if buf is not None:
        buf.seek(0)
        return pd.read_csv(buf)
    return pd.read_csv(file)


@st.cache_data(show_spinner=False)
def try_load_default_dataset() -> tuple[pd.DataFrame | None, str]:
    for p in DEFAULT_DATA_PATHS:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, f"Loaded default dataset: {Path(p).name}"
            except Exception:
                continue
    spec = SyntheticCreditcardSpec()
    df = generate_synthetic_creditcard(spec)
    return (
        df,
        "Loaded synthetic sample dataset. Upload a compatible CSV for project-specific results.",
    )


def validate_ui_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reject unusable numeric input before charts, engineering or evaluation."""
    if df.empty:
        raise ValueError("CSV contains no data rows. Upload a header and at least one record.")
    columns = [c for c in ["Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]] if c in df]
    out = coerce_numeric(df, columns)
    validate_no_nan(out, columns)
    validate_finite(out, columns)
    if "Class" in out and not out["Class"].isin([0, 1]).all():
        raise ValueError("Class must contain only 0 (non-fraud) or 1 (fraud).")
    if "Amount" in out and (out["Amount"] < 0).any():
        raise ValueError("Amount must be non-negative.")
    return out


def get_active_dataframe(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        try:
            df = validate_ui_dataframe(read_csv_any(uploaded_file))
            return df, f"Using uploaded dataset: {getattr(uploaded_file, 'name', 'uploaded')}"
        except Exception as e:
            st.error(f"Invalid CSV: {e}. Check the header, delimiters, quoting and numeric values.")
            return pd.DataFrame(), "Upload failed."
    df0, msg = try_load_default_dataset()
    if df0 is None:
        return pd.DataFrame(), msg
    return df0, msg
