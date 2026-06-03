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


# ══════════════════════════════════════════════════════════════
# HERD CLASSIFICATION
# ══════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════
# HERD CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def classify_herds(df,
                   spatial_gap_km=5.0,
                   temporal_gap_days=3,
                   observation_window_days=30,
                   min_herd_size=1):
    """
    Estimates distinct elephant herds from repeated guard sighting logs,
    operating on a sliding time window so that seasonal re-use of the same
    area does not artificially merge unrelated herd visits.

    DESIGN RATIONALE
    ----------------
    Guard entries are NOT one-per-herd.  The same physical group may generate
    many entries across beats on the same day, and a herd may be logged on
    and off for weeks as it moves through a landscape.

    Crucially, herds are ALWAYS MOVING.  A spatial cluster seen in January
    and a cluster in the same spot in June are completely separate events and
    must not be merged.  Operating on the full dataset at once — even with a
    temporal gap threshold — wrongly links them if there happens to be a
    chain of intermediate sightings in that area over months.

    ALGORITHM — sliding-window two-pass spatiotemporal union-find
    -------------------------------------------------------------
    Step 0 — Slice the dataset into overlapping windows of
             `observation_window_days`.  Each window is independent:
             herds that persist across a window boundary are re-identified
             in the next window.  Window stride = observation_window_days / 2
             (50 % overlap) so no herd event is split by a window edge.

    Within each window:

    Pass 1 — SPATIAL DEDUPLICATION within each calendar day
      Any two records on the same day within `spatial_gap_km` are merged
      into one cluster (same physical group).  Centroid = cluster mean.

    Pass 2 — TEMPORAL LINKING across days
      Day-cluster centroids are linked forward in time: if centroid A on
      day D and centroid B on day D+k are within `spatial_gap_km` AND the
      gap between A's last sighting and B's first sighting is ≤
      `temporal_gap_days` days, they belong to the same herd presence event.
      This correctly handles missed-day gaps (guard didn't patrol) without
      linking visits months apart.

    After all windows, duplicate herd events (same records captured in the
    overlapping portion of two windows) are merged by record-set overlap.

    Parameters
    ----------
    df                      : cleaned sightings DataFrame from data_validation
    spatial_gap_km          : max km between records to be the same group
                              (default 5 km — typical daily patch size)
    temporal_gap_days       : max days between consecutive cluster sightings
                              before a herd chain is broken (default 3 days)
    observation_window_days : width of each time slice in days (default 30).
                              Increase for sparse data; decrease for dense
                              continuous monitoring.
    min_herd_size           : minimum Total Count to include a record

    Returns
    -------
    tuple (sightings_df_with_herd_id, herds_summary_df)
    """
    if df.empty:
        return df.copy(), pd.DataFrame()

    # ── 0. Prepare ───────────────────────────────────────────
    work = df.copy().reset_index(drop=True)
    work["_dt"] = work["Date"] + pd.to_timedelta(
        work["Hour"].clip(lower=0), unit="h"
    )
    work = work[work["Total Count"] >= min_herd_size].reset_index(drop=True)
    if work.empty:
        return df.copy(), pd.DataFrame()

    work["_date"] = work["_dt"].dt.normalize()

    # ── Union-Find ────────────────────────────────────────────
    def make_uf(n):
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
        return parent, find, union

    # ── Sliding window setup ──────────────────────────────────
    all_dates  = work["_date"].sort_values().unique()
    date_min   = pd.Timestamp(all_dates[0])   # ensure pd.Timestamp, not numpy.datetime64
    date_max   = pd.Timestamp(all_dates[-1])
    total_days = (date_max - date_min).days + 1

    stride_days = max(1, observation_window_days // 2)
    window_td   = pd.Timedelta(days=observation_window_days)
    stride_td   = pd.Timedelta(days=stride_days)
    gap_td      = pd.Timedelta(days=temporal_gap_days)

    # Each element: list of work-df row indices belonging to one herd event
    all_herd_record_sets = []

    window_start = date_min
    while window_start <= date_max:
        window_end = window_start + window_td

        mask = (work["_date"] >= window_start) & (work["_date"] < window_end)
        w = work[mask].copy()
        if w.empty:
            window_start += stride_td
            continue

        w_idx = w.index.tolist()           # positions in work df
        w     = w.reset_index(drop=False)  # keep 'index' col = work row index
        w     = w.rename(columns={"index": "_work_idx"})
        nw    = len(w)

        lats_w = w["Latitude"].values
        lons_w = w["Longitude"].values

        # Pass 1: spatial dedup within each day
        parent_w, find_w, union_w = make_uf(nw)
        for _day, day_rows in w.groupby("_date"):
            idxs = list(day_rows.index)  # positions within w
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if haversine_np(lons_w[i], lats_w[i], lons_w[j], lats_w[j]) <= spatial_gap_km:
                        union_w(i, j)

        # Build day-level clusters within this window
        root_to_dc = {}
        dc_counter = 0
        dc_ids = np.zeros(nw, dtype=int)
        for i in range(nw):
            r = find_w(i)
            if r not in root_to_dc:
                root_to_dc[r] = dc_counter
                dc_counter += 1
            dc_ids[i] = root_to_dc[r]
        w["_dc"] = dc_ids

        day_clusters = (
            w.groupby(["_date", "_dc"])
            .agg(
                clat      = ("Latitude",    "mean"),
                clon      = ("Longitude",   "mean"),
                first_dt  = ("_dt",         "min"),
                last_dt   = ("_dt",         "max"),
                work_idxs = ("_work_idx",   list),
            )
            .reset_index()
            .sort_values("first_dt")
            .reset_index(drop=True)
        )

        # Pass 2: temporal linking across days
        nc = len(day_clusters)
        parent_dc, find_dc, union_dc = make_uf(nc)
        clats_dc  = day_clusters["clat"].values
        clons_dc  = day_clusters["clon"].values
        # Keep as Python list of pd.Timestamp so arithmetic stays in pandas
        # (numpy.datetime64 subtraction returns numpy.timedelta64 which raises
        # TypeError when compared to pd.Timedelta in NumPy 2.x / Python 3.13)
        last_dts_list  = day_clusters["last_dt"].tolist()
        first_dts_list = day_clusters["first_dt"].tolist()

        for i in range(nc):
            for j in range(i + 1, nc):
                time_gap = first_dts_list[j] - last_dts_list[i]  # pd.Timedelta
                if time_gap > gap_td:
                    break  # sorted by first_dt; no later j qualifies
                if time_gap < pd.Timedelta(0):
                    continue  # overlapping same-day clusters; handled in pass 1
                if haversine_np(clons_dc[i], clats_dc[i], clons_dc[j], clats_dc[j]) <= spatial_gap_km:
                    union_dc(i, j)

        # Collect record sets per herd event in this window
        hroot_to_wids = {}
        for i, row in day_clusters.iterrows():
            r = find_dc(i)
            if r not in hroot_to_wids:
                hroot_to_wids[r] = []
            hroot_to_wids[r].extend(row["work_idxs"])

        for wids in hroot_to_wids.values():
            all_herd_record_sets.append(set(wids))

        window_start += stride_td

    # ── De-duplicate overlapping window results ───────────────
    # Two herd events from different windows are the same physical herd if
    # they share ≥50 % of their records (the overlapping window portion).
    # Merge them with another union-find pass.
    if not all_herd_record_sets:
        return df.copy(), pd.DataFrame()

    nh = len(all_herd_record_sets)
    parent_h, find_h, union_h = make_uf(nh)
    for i in range(nh):
        for j in range(i + 1, nh):
            si, sj = all_herd_record_sets[i], all_herd_record_sets[j]
            overlap = len(si & sj)
            if overlap > 0 and overlap / min(len(si), len(sj)) >= 0.50:
                union_h(i, j)

    # Assign final herd IDs to every work-df row
    work["Herd ID"] = pd.NA
    hroot_to_final = {}
    final_counter  = 0
    for i, record_set in enumerate(all_herd_record_sets):
        r = find_h(i)
        if r not in hroot_to_final:
            hroot_to_final[r] = final_counter
            final_counter += 1
        hid = hroot_to_final[r]
        for widx in record_set:
            work.at[widx, "Herd ID"] = hid

    work["Herd ID"] = work["Herd ID"].fillna(-1).astype(int)

    # ── Build per-herd summary ────────────────────────────────
    records_out = []

    for hid, grp in work[work["Herd ID"] >= 0].groupby("Herd ID"):
        grp = grp.sort_values("_dt")
        n   = len(grp)

        start_dt   = grp["_dt"].min()
        end_dt     = grp["_dt"].max()
        duration_h = (end_dt - start_dt).total_seconds() / 3600

        c_lat = grp["Latitude"].mean()
        c_lon = grp["Longitude"].mean()

        div   = grp["Division"].mode()[0] if "Division" in grp.columns else "Unknown"
        rng   = grp["Range"].mode()[0]    if "Range"    in grp.columns else "Unknown"
        beat  = grp["Beat"].mode()[0]     if "Beat"     in grp.columns else "Unknown"
        beats_visited = ", ".join(sorted(grp["Beat"].dropna().unique())) if "Beat" in grp.columns else ""

        sev_sum   = grp["Severity Score"].sum()
        crop_dmg  = int(grp["Crop Damage"].sum())
        house_dmg = int(grp["House Damage"].sum())
        injury    = int(grp["Injury"].sum())

        total_count = int(grp["Total Count"].max())
        male_c      = int(grp["Male Count"].max())
        female_c    = int(grp["Female Count"].max())
        calf_c      = int(grp["Calf Count"].max())
        unk_c       = int(grp["Unknown Count"].max()) if "Unknown Count" in grp.columns else 0

        # Composition
        classified_total = male_c + female_c + calf_c
        if classified_total > 0:
            male_pct   = male_c   / classified_total
            female_pct = female_c / classified_total
            if male_pct >= 0.70 and calf_c == 0:
                composition = "Bull Group"
            elif calf_c > 0 and female_c > 0 and (female_c / calf_c) <= 4:
                composition = "Nursery Herd"
            elif calf_c > 0 and male_c > 0:
                composition = "Mixed Herd"
            elif calf_c > 0:
                composition = "Nursery Herd"
            elif female_pct >= 0.90 and male_c == 0:
                composition = "Female Group"
            elif male_c > 0 and female_c > 0:
                composition = "Mixed Herd"
            elif male_c > 0:
                composition = "Bull Group"
            else:
                composition = "Unclassified"
        else:
            composition = "Unclassified"

        # Movement
        if n >= 2:
            g_lats = grp["Latitude"].values
            g_lons = grp["Longitude"].values
            path_dists = [
                haversine_np(g_lons[k], g_lats[k], g_lons[k+1], g_lats[k+1])
                for k in range(len(g_lats) - 1)
            ]
            total_path   = sum(path_dists)
            displacement = haversine_np(g_lons[0], g_lats[0], g_lons[-1], g_lats[-1])
        else:
            total_path   = 0.0
            displacement = 0.0

        if crop_dmg > 0 or house_dmg > 0:
            movement = "Raiding"
        elif displacement > 3 * spatial_gap_km:
            movement = "Transiting"
        elif displacement <= spatial_gap_km / 2:
            movement = "Stationary"
        else:
            movement = "Ranging"

        # Temporal
        valid_hours = grp[grp["Hour"] != -1]["Hour"].values
        if len(valid_hours) > 0:
            nh2 = len(valid_hours)
            crep_mask  = (
                ((valid_hours >= 5)  & (valid_hours <= 7)) |
                ((valid_hours >= 17) & (valid_hours <= 19))
            )
            night_mask = (valid_hours >= 20) | (valid_hours <= 4)
            day_mask   = (valid_hours >= 8)  & (valid_hours <= 16)
            if crep_mask.sum()  / nh2 >= 0.60:
                temporal = "Crepuscular"
            elif night_mask.sum() / nh2 >= 0.70:
                temporal = "Nocturnal"
            elif day_mask.sum()   / nh2 >= 0.70:
                temporal = "Diurnal"
            else:
                temporal = "Mixed"
        else:
            temporal = "Unknown"

        # Risk
        near_village = bool(grp["Near Village"].any()) \
                       if "Near Village" in grp.columns else False
        if injury > 0 or sev_sum > 25:
            risk = "Critical"
        elif crop_dmg > 0 or house_dmg > 0 or sev_sum > 10:
            risk = "High"
        elif near_village or sev_sum > 5:
            risk = "Moderate"
        else:
            risk = "Low"

        records_out.append({
            "Herd ID":              hid,
            "Start Time":           start_dt,
            "End Time":             end_dt,
            "Duration (hrs)":       round(duration_h, 1),
            "Records":              n,
            "Total Count (max)":    total_count,
            "Centroid Lat":         round(c_lat, 5),
            "Centroid Lon":         round(c_lon, 5),
            "Division":             div,
            "Range":                rng,
            "Beat":                 beat,
            "Beats Visited":        beats_visited,
            "Composition":          composition,
            "Movement":             movement,
            "Temporal Pattern":     temporal,
            "Conflict Risk":        risk,
            "Severity Score (sum)": round(sev_sum, 1),
            "Crop Damage":          crop_dmg,
            "House Damage":         house_dmg,
            "Injury":               injury,
            "Male Count":           male_c,
            "Female Count":         female_c,
            "Calf Count":           calf_c,
            "Unknown Count":        unk_c,
            "Displacement (km)":    round(displacement, 2),
            "Total Path (km)":      round(total_path, 2),
        })

    herds_df = pd.DataFrame(records_out)

    # Merge Herd ID back to original df
    df_out = df.copy()
    df_out["_dt"] = df_out["Date"] + pd.to_timedelta(
        df_out["Hour"].clip(lower=0), unit="h"
    )
    df_out = df_out.merge(
        work[["Latitude", "Longitude", "_dt", "Herd ID"]],
        on=["Latitude", "Longitude", "_dt"],
        how="left",
    ).drop(columns=["_dt"])

    return df_out, herds_df
