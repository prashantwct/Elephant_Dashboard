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
    REFUGE_CONFLICT_SEARCH_RADIUS_KM,
    REFUGE_SPATIAL_GRID_RESOLUTION_KM,
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


# ── degrees-per-km helpers (approximate, fine for <50 km scales) ──────────
_KM_PER_DEG_LAT = EARTH_RADIUS_KM * np.pi / 180          # ~111.2 km / °lat
def _km_to_deg_lat(km): return km / _KM_PER_DEG_LAT
def _km_to_deg_lon(km, lat_deg):
    return km / (_KM_PER_DEG_LAT * np.cos(np.radians(lat_deg)))


def infer_refuges_from_conflict_proximity(df, search_radius_km=None, grid_res_km=None):
    """
    Infers likely daytime refuge zones from spatial proximity to conflict events,
    *independently of whether any field entry was logged at that location*.

    Method
    ------
    1. Extract all conflict events (crop/house/injury damage) with their
       coordinates and a weighted severity score.
    2. Build a regular lat/lon grid that covers the bounding box of all
       conflict events, extended by *search_radius_km* on every side.
       Grid spacing = *grid_res_km*.
    3. For every grid cell, compute a **Conflict Attraction Score**:
         - Inverse-distance weighted sum of nearby conflict severities.
         - Cells closer to dense, high-severity conflicts score higher.
         - Cells that already ARE conflict sites are penalised (elephants
           don't shelter where they raid).
    4. Overlay observed daytime entries (if any) to boost grid cells that
       have both spatial proximity to conflicts AND evidence of actual use.
    5. Return the top candidate cells as a DataFrame, annotated with the
       nearest known Beat (from actual entries) for operational labelling.

    Parameters
    ----------
    df              : full filtered sightings DataFrame (all hours/types)
    search_radius_km: max distance from a conflict to look for refuges
                      (default: REFUGE_CONFLICT_SEARCH_RADIUS_KM from config)
    grid_res_km     : spatial grid resolution in km
                      (default: REFUGE_SPATIAL_GRID_RESOLUTION_KM from config)

    Returns
    -------
    DataFrame with columns:
        Latitude, Longitude, Conflict Attraction Score, Observation Boost,
        Combined Score, Nearest Beat, Nearest Beat Distance (km),
        Nearby Conflict Count, Nearby Conflict Severity
    Sorted by Combined Score descending.
    Returns empty DataFrame if no conflict events exist.
    """
    if search_radius_km is None:
        search_radius_km = REFUGE_CONFLICT_SEARCH_RADIUS_KM
    if grid_res_km is None:
        grid_res_km = REFUGE_SPATIAL_GRID_RESOLUTION_KM

    # ── 1. Conflict events ────────────────────────────────────────────────
    conflict_mask = (
        (df["Crop Damage"]  > 0) |
        (df["House Damage"] > 0) |
        (df["Injury"]       > 0)
    )
    conflicts = df[conflict_mask].copy()
    if conflicts.empty:
        return pd.DataFrame()

    # Weighted severity: house > crop > injury ordering already in config weights
    conflicts["_conflict_weight"] = (
        (conflicts["Crop Damage"]  > 0).astype(float) * SEVERITY_CROP_WEIGHT  +
        (conflicts["House Damage"] > 0).astype(float) * SEVERITY_HOUSE_WEIGHT +
        (conflicts["Injury"]       > 0).astype(float) * SEVERITY_INJURY_WEIGHT
    )

    c_lats = conflicts["Latitude"].values
    c_lons = conflicts["Longitude"].values
    c_wts  = conflicts["_conflict_weight"].values

    # ── 2. Build bounding-box grid ────────────────────────────────────────
    pad_lat = _km_to_deg_lat(search_radius_km)
    # Use the mean conflict latitude for lon conversion (good enough at these scales)
    mean_lat = float(np.mean(c_lats))
    pad_lon  = _km_to_deg_lon(search_radius_km, mean_lat)

    lat_min = c_lats.min() - pad_lat
    lat_max = c_lats.max() + pad_lat
    lon_min = c_lons.min() - pad_lon
    lon_max = c_lons.max() + pad_lon

    deg_step_lat = _km_to_deg_lat(grid_res_km)
    deg_step_lon = _km_to_deg_lon(grid_res_km, mean_lat)

    grid_lats = np.arange(lat_min, lat_max + deg_step_lat, deg_step_lat)
    grid_lons = np.arange(lon_min, lon_max + deg_step_lon, deg_step_lon)

    # Cartesian product → all grid cells
    g_lats, g_lons = np.meshgrid(grid_lats, grid_lons, indexing="ij")
    g_lats = g_lats.ravel()
    g_lons = g_lons.ravel()
    n_cells = len(g_lats)

    # ── 3. Conflict Attraction Score (inverse-distance weighted) ──────────
    # Distance matrix: (n_cells, n_conflicts)
    dist_gc = haversine_np(
        g_lons[:, np.newaxis], g_lats[:, np.newaxis],
        c_lons[np.newaxis, :],  c_lats[np.newaxis, :],
    )

    # Only conflicts within search_radius_km influence a cell
    in_radius = dist_gc <= search_radius_km          # bool (n_cells, n_conflicts)

    # Inverse-distance weight; add small epsilon to avoid divide-by-zero
    # at exact conflict locations
    idw = np.where(in_radius, c_wts[np.newaxis, :] / (dist_gc + 0.05), 0.0)

    attraction_score   = idw.sum(axis=1)             # (n_cells,)
    nearby_conf_count  = in_radius.sum(axis=1)       # (n_cells,)
    nearby_conf_sev    = np.where(in_radius, c_wts[np.newaxis, :], 0.0).sum(axis=1)

    # Penalise cells that sit directly on top of conflict sites
    # (elephants raid at night, rest elsewhere during day)
    on_conflict = (dist_gc <= grid_res_km).any(axis=1)
    attraction_score[on_conflict] *= 0.3

    # Keep only cells that have at least one conflict in range
    valid = nearby_conf_count > 0
    if not valid.any():
        return pd.DataFrame()

    g_lats = g_lats[valid]
    g_lons = g_lons[valid]
    attraction_score  = attraction_score[valid]
    nearby_conf_count = nearby_conf_count[valid]
    nearby_conf_sev   = nearby_conf_sev[valid]

    # ── 4. Observation Boost from logged daytime entries ──────────────────
    day_df = df[
        (df["Hour"] >= REFUGE_DAY_START_HOUR) &
        (df["Hour"] <= REFUGE_DAY_END_HOUR)   &
        (df["Severity Score"] <= REFUGE_MAX_SEVERITY)
    ]

    obs_boost = np.zeros(len(g_lats))
    if not day_df.empty:
        obs_lats = day_df["Latitude"].values
        obs_lons = day_df["Longitude"].values

        dist_go = haversine_np(
            g_lons[:, np.newaxis], g_lats[:, np.newaxis],
            obs_lons[np.newaxis, :], obs_lats[np.newaxis, :],
        )
        # Any observed daytime presence within one grid cell = boost
        obs_boost = (dist_go <= grid_res_km).sum(axis=1).astype(float)

    # ── 5. Combined Score (normalised) ────────────────────────────────────
    max_attr = attraction_score.max() if attraction_score.max() > 0 else 1.0
    max_obs  = obs_boost.max()         if obs_boost.max()  > 0 else 1.0

    # 70 % spatial inference, 30 % observational evidence
    combined = 0.70 * (attraction_score / max_attr) + 0.30 * (obs_boost / max_obs)

    # ── 6. Label each cell with nearest Beat from actual entries ──────────
    beats = df.dropna(subset=["Beat", "Latitude", "Longitude"]).copy()
    if not beats.empty:
        b_lats  = beats["Latitude"].values
        b_lons  = beats["Longitude"].values
        b_names = beats["Beat"].values

        dist_gb = haversine_np(
            g_lons[:, np.newaxis], g_lats[:, np.newaxis],
            b_lons[np.newaxis, :],  b_lats[np.newaxis, :],
        )
        nearest_idx      = dist_gb.argmin(axis=1)
        nearest_beat     = b_names[nearest_idx]
        nearest_beat_dist = dist_gb[np.arange(len(g_lats)), nearest_idx]
    else:
        nearest_beat      = np.full(len(g_lats), "Unknown")
        nearest_beat_dist = np.full(len(g_lats), np.nan)

    # ── 7. Assemble result DataFrame ──────────────────────────────────────
    result = pd.DataFrame({
        "Latitude":                   g_lats,
        "Longitude":                  g_lons,
        "Conflict Attraction Score":  attraction_score,
        "Observation Boost":          obs_boost,
        "Combined Score":             combined,
        "Nearest Beat":               nearest_beat,
        "Nearest Beat Distance (km)": nearest_beat_dist.round(2),
        "Nearby Conflict Count":      nearby_conf_count,
        "Nearby Conflict Severity":   nearby_conf_sev.round(1),
    })

    return (
        result
        .sort_values("Combined Score", ascending=False)
        .reset_index(drop=True)
    )
