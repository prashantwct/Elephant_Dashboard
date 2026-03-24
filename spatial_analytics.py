"""
spatial_analytics.py
====================
Authoritative spatial helper functions. All distance calculations in the
app should import from here so the Earth-radius constant stays consistent.
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from config import (
    EARTH_RADIUS_KM,
    AFFECTED_VILLAGE_RADIUS_KM,
    SEVERITY_CROP_WEIGHT,
    SEVERITY_HOUSE_WEIGHT,
    SEVERITY_INJURY_WEIGHT,
    REFUGE_DAY_START_HOUR,
    REFUGE_DAY_END_HOUR,
    REFUGE_MAX_SEVERITY,
)


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorised Haversine distance between two points (or arrays of points).
    All inputs in decimal degrees; returns distance in kilometres.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def calculate_affected_villages(sightings_df, villages_df, radius_km=AFFECTED_VILLAGE_RADIUS_KM):
    """
    Vectorised: for every sighting row, find villages within *radius_km*.
    Returns sightings_df with a new 'Affected Villages' column (comma-separated
    village names, or "None").

    Replaces the original row-by-row apply() implementation — O(n) matrix
    multiply instead of O(n*m) Python loop.
    """
    if villages_df is None or sightings_df.empty:
        sightings_df = sightings_df.copy()
        sightings_df["Affected Villages"] = "None"
        return sightings_df

    v_lons = villages_df["Longitude"].values
    v_lats = villages_df["Latitude"].values
    v_names = villages_df["Village"].values

    s_lons = sightings_df["Longitude"].values
    s_lats = sightings_df["Latitude"].values

    # Full distance matrix: shape (n_villages, n_sightings)
    dist_matrix = haversine_np(
        v_lons[:, np.newaxis], v_lats[:, np.newaxis],
        s_lons[np.newaxis, :],  s_lats[np.newaxis, :],
    )

    # For each sighting (column), collect village names within radius
    within = dist_matrix <= radius_km  # bool matrix

    result = []
    for col_idx in range(within.shape[1]):
        nearby = v_names[within[:, col_idx]]
        result.append(", ".join(sorted(nearby)) if len(nearby) else "None")

    sightings_df = sightings_df.copy()
    sightings_df["Affected Villages"] = result
    return sightings_df


def identify_risk_villages(sightings_df, villages_df, damage_rad_km, presence_rad_km, cons_days):
    """
    Identifies at-risk villages based on two criteria:
      1. Damage Radius  — damage event within *damage_rad_km* of village.
      2. Presence Streak — elephant present within *presence_rad_km* for
         *cons_days* consecutive days ending on the most recent date.

    Returns a list of dicts: {Village, Lat, Lon, Reason, Color, Radius}.
    """
    if villages_df is None or sightings_df.empty:
        return []

    v_lons  = villages_df["Longitude"].values
    v_lats  = villages_df["Latitude"].values
    v_names = villages_df["Village"].values

    s_lons  = sightings_df["Longitude"].values
    s_lats  = sightings_df["Latitude"].values

    # Full distance matrix: shape (n_villages, n_sightings)
    dists = haversine_np(
        v_lons[:, np.newaxis], v_lats[:, np.newaxis],
        s_lons[np.newaxis, :],  s_lats[np.newaxis, :],
    )

    affected_list = []
    added_villages = set()

    # ── Criteria 1: Damage ────────────────────────────────────
    damage_mask = (sightings_df["Crop Damage"] > 0) | (sightings_df["House Damage"] > 0)
    if damage_mask.any():
        dmg_dists = dists[:, damage_mask.values]
        close_enough = np.min(dmg_dists, axis=1) <= damage_rad_km
        for idx in np.where(close_enough)[0]:
            name = v_names[idx]
            affected_list.append({
                "Village": name,
                "Lat":     v_lats[idx],
                "Lon":     v_lons[idx],
                "Reason":  f"Damage Reported (<{damage_rad_km} km)",
                "Color":   "red",
                "Radius":  damage_rad_km * 1000,
            })
            added_villages.add(name)

    # ── Criteria 2: Consecutive Presence ─────────────────────
    max_date = sightings_df["Date"].max()
    dates    = sightings_df["Date"].values
    streak   = np.ones(len(villages_df), dtype=bool)

    for i in range(cons_days):
        check_date = max_date - timedelta(days=i)
        idx_date   = np.where(dates == np.datetime64(check_date))[0]
        if len(idx_date) == 0:
            streak[:] = False
            break
        daily_close = np.any(dists[:, idx_date] <= presence_rad_km, axis=1)
        streak &= daily_close

    for idx in np.where(streak)[0]:
        name = v_names[idx]
        if name not in added_villages:
            affected_list.append({
                "Village": name,
                "Lat":     v_lats[idx],
                "Lon":     v_lons[idx],
                "Reason":  f"{cons_days}-Day Presence (<{presence_rad_km} km)",
                "Color":   "orange",
                "Radius":  presence_rad_km * 1000,
            })

    return affected_list


def identify_daytime_refuges(df):
    """
    Identifies staging/refuge beats by filtering for daylight sightings
    (REFUGE_DAY_START_HOUR–REFUGE_DAY_END_HOUR) with low conflict severity.

    Returns a DataFrame with columns:
        Division, Range, Beat, Persistence Score, Avg Group Size,
        Sighting Frequency, Latitude, Longitude

    Column names are explicit and match every caller.
    """
    day_df = df[
        (df["Hour"] >= REFUGE_DAY_START_HOUR) &
        (df["Hour"] <= REFUGE_DAY_END_HOUR)
    ].copy()

    if day_df.empty:
        return pd.DataFrame()

    # Foraging-sign weight using official severity weights for consistency
    def refuge_weight(row):
        score = 1.0
        if row["Sighting Type"] == "Direct":
            score += 1.5
        detail = str(row.get("Sighting Type Detail", ""))
        if "brokenBranches" in detail:
            score += 1.5
        if "dung" in detail:
            score += 0.5
        return score

    day_df["_weight"] = day_df.apply(refuge_weight, axis=1)

    low_conflict = day_df[day_df["Severity Score"] <= REFUGE_MAX_SEVERITY]
    if low_conflict.empty:
        return pd.DataFrame()

    summary = (
        low_conflict
        .groupby(["Division", "Range", "Beat"])
        .agg(
            **{
                "Persistence Score": ("_weight", "sum"),
                "Avg Group Size":    ("Total Count", "mean"),
                "Sighting Frequency":("_weight", "count"),
                "Latitude":          ("Latitude",  "mean"),
                "Longitude":         ("Longitude", "mean"),
            }
        )
        .reset_index()
        .sort_values("Persistence Score", ascending=False)
    )

    return summary
