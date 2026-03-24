"""
data_validation.py
==================
Data loading, cleaning, and derived-field calculation.
All column expectations and business logic stay here so app.py stays clean.
"""

import pandas as pd
import streamlit as st
import logging
from config import (
    DATE_FORMATS,
    DUPLICATE_SUBSET,
    SEVERITY_PRESENCE_WEIGHT,
    SEVERITY_CROP_WEIGHT,
    SEVERITY_HOUSE_WEIGHT,
    SEVERITY_INJURY_WEIGHT,
)

logger = logging.getLogger(__name__)


def _read_csv_robust(source):
    """Try UTF-8 then Latin-1 when reading a CSV file or uploaded buffer."""
    try:
        return pd.read_csv(source)
    except UnicodeDecodeError:
        try:
            source.seek(0)
        except AttributeError:
            pass
        return pd.read_csv(source, encoding="ISO-8859-1")


def _parse_dates_robust(series):
    """
    Try multiple date formats in order; return a datetime Series.
    Logs a warning with a count of rows that could not be parsed.
    """
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        success = parsed.notna().sum()
        if success > 0:
            failed = parsed.isna().sum()
            if failed:
                logger.warning("Date parsing with format %s: %d rows failed.", fmt, failed)
            return parsed
    # Last-resort: let pandas infer
    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")


def _parse_hour_robust(time_series):
    """
    Try multiple time formats; return an integer hour Series.
    Returns None if parsing fails completely (caller should warn the user).
    """
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        parsed = pd.to_datetime(time_series, format=fmt, errors="coerce")
        if parsed.notna().any():
            return parsed.dt.hour.fillna(-1).astype(int)
    return None


def load_and_clean_sightings(uploaded_csv):
    """
    Full pipeline: read → date parse → time parse → dedup → numeric cleanup
    → derived fields.

    Returns (df_clean, warnings_list) where warnings_list is a list of
    human-readable strings suitable for st.warning().
    """
    warnings = []

    # ── 1. Read ───────────────────────────────────────────────
    df = _read_csv_robust(uploaded_csv)
    df.columns = [c.strip() for c in df.columns]

    # ── 2. Date ───────────────────────────────────────────────
    df["Date"] = _parse_dates_robust(df["Date"])
    bad_dates = df["Date"].isna().sum()
    if bad_dates:
        warnings.append(f"⚠️ {bad_dates} rows dropped — date could not be parsed.")
    df = df.dropna(subset=["Date"])

    # ── 3. Time / Hour ────────────────────────────────────────
    if "Time" in df.columns:
        hours = _parse_hour_robust(df["Time"])
        if hours is not None:
            df["Hour"] = hours
            unresolved = (hours == -1).sum()
            if unresolved:
                warnings.append(
                    f"⚠️ {unresolved} rows had unparseable Time values; "
                    "hour recorded as -1 (excluded from nocturnal metric)."
                )
        else:
            df["Hour"] = -1
            warnings.append(
                "⚠️ Time column could not be parsed in any known format. "
                "Nocturnal and hourly metrics will be unavailable."
            )
    else:
        df["Hour"] = -1

    # ── 4. Coordinate validation ──────────────────────────────
    for col in ("Latitude", "Longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_coords = df[["Latitude", "Longitude"]].isna().any(axis=1).sum()
    if bad_coords:
        warnings.append(f"⚠️ {bad_coords} rows dropped — invalid coordinates.")
    df = df.dropna(subset=["Latitude", "Longitude"])

    # ── 5. Text normalisation ─────────────────────────────────
    for col in ("Range", "Beat", "Division"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().replace("Nan", "Unknown")

    for col in ("Division", "Range", "Beat"):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ── 6. Deduplication (robust subset) ─────────────────────
    dedup_cols = [c for c in DUPLICATE_SUBSET if c in df.columns]
    initial_len = len(df)
    df = df.drop_duplicates(subset=dedup_cols)
    removed = initial_len - len(df)
    if removed:
        warnings.append(f"🧹 {removed} duplicate entries removed.")

    # ── 7. Numeric cleanup ────────────────────────────────────
    count_cols = ["Total Count", "Male Count", "Female Count", "Calf Count",
                  "Crop Damage", "House Damage", "Injury"]
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── 8. Derived fields ─────────────────────────────────────
    df = calculate_derived_fields(df)

    return df, warnings


def calculate_derived_fields(df):
    """Add Unknown Count, Severity Score, Is_Night. Pure transforms, no drops."""

    # Unknown demographics
    classified = df.get("Male Count", 0) + df.get("Female Count", 0) + df.get("Calf Count", 0)
    df["Unknown Count"] = (df["Total Count"] - classified).clip(lower=0)

    # Severity score — single canonical formula from config weights
    df["Severity Score"] = (
        SEVERITY_PRESENCE_WEIGHT
        + (df["Crop Damage"]  > 0).astype(float) * SEVERITY_CROP_WEIGHT
        + (df["House Damage"] > 0).astype(float) * SEVERITY_HOUSE_WEIGHT
        + (df["Injury"]       > 0).astype(float) * SEVERITY_INJURY_WEIGHT
    )

    # Nocturnal flag (exclude -1 sentinel hours from hour-parse failures)
    df["Is_Night"] = df["Hour"].apply(
        lambda x: 1 if x != -1 and (x >= 18 or x <= 6) else 0
    )

    return df
