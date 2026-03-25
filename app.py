"""
Elephant Sighting & Conflict Command Centre
===========================================
Main Streamlit application.

Fixes applied vs original:
  - config.py imported and used everywhere (no more inline magic numbers)
  - spatial_analytics.py is the single source of distance/refuge logic
  - data_validation.py owns all loading / parsing / cleaning
  - identify_daytime_refuges returns correct column names (Persistence Score,
    Avg Group Size, Sighting Frequency) — no more KeyError crash in tab_refuge
  - Explanation panel de-duplicated
  - Session state initialised once, at the top
  - Metrics displayed with st.metric() (proper delta display + accessibility)
  - Map auto-centres on filtered data centroid (falls back to config default)
  - haversine uses single EARTH_RADIUS_KM constant from config
  - Severity Score formula defined once in data_validation/config
  - Date parsing tries multiple formats; failed rows are reported, not silently dropped
  - Hour parse failure sets sentinel -1; metrics exclude it cleanly
  - Duplicate detection uses 6-column robust subset including coordinates
  - Animated charts use weekly frames to avoid 365-frame lag
  - Heavy spatial functions wrapped with st.cache_data
  - Bare except replaced with except Exception as e
  - KML ignored-name list sourced from config
  - Report generator receives explicit figure args (no locals() lookup)
  - Staff tab: divisions below MIN_STAFF_PER_DIVISION are flagged, with toggle to show all
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import zipfile
import os
import xml.etree.ElementTree as ET
import glob
import plotly.express as px
from datetime import timedelta
import numpy as np
import logging

import config
from data_validation import load_and_clean_sightings
from spatial_analytics import (
    calculate_affected_villages,
    identify_risk_villages,
    identify_daytime_refuges,
    infer_refuges_from_conflict_proximity,
    classify_herds,
)

logging.basicConfig(level=logging.WARNING)

# ══════════════════════════════════════════════════════════════
# 1. PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=config.APP_TITLE,
    layout=config.LAYOUT,
    page_icon=config.PAGE_ICON,
)

# ══════════════════════════════════════════════════════════════
# 2. SESSION STATE  (initialised once at top level)
# ══════════════════════════════════════════════════════════════

if "map_filter"       not in st.session_state: st.session_state.map_filter       = "All"
if "hotspot_beat"     not in st.session_state: st.session_state.hotspot_beat     = None
if "selected_village" not in st.session_state: st.session_state.selected_village = None

# ══════════════════════════════════════════════════════════════
# 3. CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_village_data(filepath="centroids.csv"):
    """Load village centroid CSV; returns DataFrame or None."""
    if not os.path.exists(filepath):
        return None
    try:
        v_df = pd.read_csv(filepath)
    except UnicodeDecodeError:
        v_df = pd.read_csv(filepath, encoding="ISO-8859-1")
    except Exception as e:
        logging.warning("Village data load failed: %s", e)
        return None

    v_df.columns = [c.strip() for c in v_df.columns]
    rename_map = {
        "Lat": "Latitude", "lat": "Latitude",
        "Lon": "Longitude", "Long": "Longitude", "lon": "Longitude",
        "Name": "Village", "village": "Village", "VILLAGE": "Village",
    }
    v_df = v_df.rename(columns=rename_map)
    if {"Latitude", "Longitude", "Village"}.issubset(v_df.columns):
        return v_df
    return None


@st.cache_data
def load_map_data_as_geojson(folder_path="."):
    """
    Scan folder for KMZ/KML files and return a list of GeoJSON Feature dicts.
    Silently skips unreadable files and logs the error.
    """
    features = []
    files = (
        glob.glob(os.path.join(folder_path, "*.kmz")) +
        glob.glob(os.path.join(folder_path, "*.kml"))
    )

    for file_path in files:
        try:
            content = None
            filename = os.path.basename(file_path)

            if file_path.endswith(".kmz"):
                with zipfile.ZipFile(file_path, "r") as z:
                    kml_files = [f for f in z.namelist() if f.endswith(".kml")]
                    if kml_files:
                        with z.open(kml_files[0]) as f:
                            content = f.read()
            else:
                with open(file_path, "rb") as f:
                    content = f.read()

            if not content:
                continue

            tree = ET.fromstring(content)
            ns   = {"kml": "http://www.opengis.net/kml/2.2"}

            for placemark in tree.findall(".//kml:Placemark", ns):
                name  = _get_feature_name(placemark, ns)
                polys = _parse_kml_coordinates(placemark, ns)
                for poly_coords in polys:
                    features.append({
                        "type": "Feature",
                        "properties": {"name": name, "source": filename},
                        "geometry":   {"type": "Polygon", "coordinates": [poly_coords]},
                    })
        except Exception as e:
            logging.warning("Could not parse %s: %s", file_path, e)

    return features


@st.cache_data
def cached_calculate_affected_villages(sightings_hash, sightings_df, villages_df, radius_km):
    """Cache wrapper for the vectorised village-distance function."""
    return calculate_affected_villages(sightings_df, villages_df, radius_km)


@st.cache_data
def cached_identify_risk_villages(sightings_hash, sightings_df, villages_df,
                                   damage_rad, presence_rad, cons_days):
    """Cache wrapper for risk-village identification."""
    return identify_risk_villages(sightings_df, villages_df, damage_rad, presence_rad, cons_days)


@st.cache_data
def cached_identify_daytime_refuges(sightings_hash, df):
    """Cache wrapper for refuge detection."""
    return identify_daytime_refuges(df)


@st.cache_data
def cached_infer_refuges_from_conflict_proximity(sightings_hash, df, search_radius_km, grid_res_km):
    """Cache wrapper for spatial conflict-proximity refuge inference."""
    return infer_refuges_from_conflict_proximity(df, search_radius_km, grid_res_km)


@st.cache_data
def cached_classify_herds(sightings_hash, df, spatial_gap_km, temporal_gap_hours, min_size):
    """Cache wrapper for herd classification."""
    return classify_herds(df, spatial_gap_km, temporal_gap_hours, min_size)


# ══════════════════════════════════════════════════════════════
# 4. KML HELPERS  (module-level, not cached — pure functions)
# ══════════════════════════════════════════════════════════════

def _get_feature_name(placemark, ns):
    name_tag = placemark.find("kml:name", ns)
    if name_tag is not None and name_tag.text:
        val = name_tag.text.strip().upper()
        if val not in config.KML_IGNORED_NAMES:
            return name_tag.text.strip().title()

    ext = placemark.find("kml:ExtendedData", ns)
    if ext is not None:
        for sd in ext.findall(".//kml:SimpleData", ns):
            if sd.get("name", "").lower() in ("range_nm", "beat_nm", "name"):
                return sd.text.title() if sd.text else "Unknown Area"

    return "Unknown Area"


def _parse_kml_coordinates(placemark, ns):
    polygons = []
    for polygon in placemark.findall(".//kml:Polygon", ns):
        outer = polygon.find(
            ".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns
        )
        if outer is not None and outer.text:
            pts = []
            for point in outer.text.strip().split():
                try:
                    parts = point.split(",")
                    lon, lat = float(parts[0]), float(parts[1])
                    pts.append([lon, lat])
                except (ValueError, IndexError):
                    continue
            if pts:
                polygons.append(pts)
    return polygons


# ══════════════════════════════════════════════════════════════
# 5. REPORT GENERATION
# ══════════════════════════════════════════════════════════════

def generate_full_html_report(df, map_object, fig_trend, fig_demog,
                               fig_hourly, fig_damage, fig_sunburst,
                               start_date, end_date):
    """Generates a standalone HTML report with embedded charts and map."""

    total_sightings   = len(df)
    cumulative_count  = int(df["Total Count"].sum())
    conflict_count    = int(((df["Crop Damage"] > 0) | (df["House Damage"] > 0) | (df["Injury"] > 0)).sum())
    severity_score    = int(df["Severity Score"].sum())

    narrative = f"""
    <h2 style="color:#2c3e50;">🐘 Elephant Monitoring &amp; Conflict Report</h2>
    <p><b>Report Period:</b> {start_date} to {end_date}</p>
    <div style="background:#ecf0f1;padding:20px;border-radius:8px;border-left:5px solid #2980b9;">
        <h4>📋 Executive Summary</h4>
        <p>Total field entries: <b>{total_sightings}</b> &mdash;
           cumulative elephant presence: <b>{cumulative_count}</b>.</p>
        <p><b>Conflict Severity Score:</b> {severity_score}
           from <b>{conflict_count}</b> conflict incidents.</p>
    </div><br>
    """

    methodology = f"""
    <div style="margin-top:20px;border:1px solid #ddd;padding:15px;background:#fafafa;font-size:.9em;">
        <h5>ℹ️ Methodology &amp; Definitions</h5>
        <ul>
            <li><b>Conflict Severity Score:</b>
                Presence×{config.SEVERITY_PRESENCE_WEIGHT} +
                Crop×{config.SEVERITY_CROP_WEIGHT} +
                House×{config.SEVERITY_HOUSE_WEIGHT} +
                Injury×{config.SEVERITY_INJURY_WEIGHT}</li>
            <li><b>HEC Ratio:</b> % of entries with a conflict event.</li>
            <li><b>Nocturnal Activity:</b> Sightings between 18:00 and 06:00.</li>
        </ul>
    </div><br>
    """

    map_html   = map_object.get_root().render()
    map_iframe = (
        f'<iframe srcdoc="{map_html.replace(chr(34), "&quot;")}" '
        f'width="100%" height="500px" style="border:1px solid #ddd;border-radius:4px;"></iframe>'
    )

    def _chart(fig):
        return fig.to_html(full_html=False, include_plotlyjs="cdn") if fig else "<p>No data.</p>"

    dmg_table = (
        df.groupby(["Division", "Range"])
        .agg({"Crop Damage": lambda x: (x > 0).sum(),
              "House Damage": lambda x: (x > 0).sum(),
              "Injury": lambda x: (x > 0).sum()})
        .reset_index()
    )
    dmg_table = dmg_table[
        (dmg_table["Crop Damage"] > 0) |
        (dmg_table["House Damage"] > 0) |
        (dmg_table["Injury"] > 0)
    ]
    table_html = dmg_table.to_html(classes="table table-striped table-bordered", index=False)

    return f"""
    <html><head><title>Elephant Report</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    <style>
        body{{padding:40px;font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;}}
        h3{{margin-top:40px;border-bottom:2px solid #eee;padding-bottom:10px;color:#34495e;}}
        .footer{{margin-top:50px;text-align:center;color:#7f8c8d;font-size:12px;
                 border-top:1px solid #eee;padding-top:20px;}}
    </style></head>
    <body><div class="container">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
            <img src="{config.LOGO_MP}" width="80" alt="MP Forest">
            <img src="{config.LOGO_WCT}" width="150" alt="WCT Mumbai">
        </div>
        {narrative}{methodology}
        <h3>📍 Spatial Distribution Map</h3>{map_iframe}
        <h3>📊 Population &amp; Temporal Analysis</h3>
        <div class="row">
            <div class="col-md-6">{_chart(fig_trend)}</div>
            <div class="col-md-6">{_chart(fig_demog)}</div>
        </div>
        <h3>🔥 Conflict &amp; Activity Analysis</h3>
        <div class="row">
            <div class="col-md-6">{_chart(fig_damage)}</div>
            <div class="col-md-6">{_chart(fig_hourly)}</div>
        </div>
        <div class="row"><div class="col-md-12">{_chart(fig_sunburst)}</div></div>
        <h3>⚠️ Detailed Damage Data</h3>{table_html}
        <div class="footer">
            <p>Generated by Elephant Sighting &amp; Conflict Dashboard</p>
            <p>Intellectual Property of Wildlife Conservation Trust, Mumbai</p>
        </div>
    </div></body></html>
    """


# ══════════════════════════════════════════════════════════════
# 6. SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    c1, c2 = st.columns([1, 1])
    c1.markdown(f'<img src="{config.LOGO_WCT}" width="100%">', unsafe_allow_html=True)
    c2.markdown(f'<img src="{config.LOGO_MP}"  width="80%">',  unsafe_allow_html=True)
    st.markdown("---")
    st.info("© **Wildlife Conservation Trust, Mumbai**\n\nDeveloped for MP Forest Department Elephant Monitoring.")

    st.divider()
    st.subheader("🏘️ Risk Parameters")
    with st.expander("Configure Logic", expanded=True):
        p_dmg_rad = st.slider(
            "Damage Radius (km)",
            config.DAMAGE_RADIUS_MIN_KM, config.DAMAGE_RADIUS_MAX_KM,
            config.DAMAGE_RADIUS_DEFAULT_KM, config.DAMAGE_RADIUS_STEP_KM,
            key="slider_dmg",
            help="Alert village if damage occurs within this distance.",
        )
        p_pres_rad = st.slider(
            "Presence Radius (km)",
            config.PRESENCE_RADIUS_MIN_KM, config.PRESENCE_RADIUS_MAX_KM,
            config.PRESENCE_RADIUS_DEFAULT_KM, config.PRESENCE_RADIUS_STEP_KM,
            key="slider_pres",
            help="Alert village if elephant present within this distance.",
        )
        p_days = st.slider(
            "Consecutive Days",
            config.CONSECUTIVE_DAYS_MIN, config.CONSECUTIVE_DAYS_MAX,
            config.CONSECUTIVE_DAYS_DEFAULT,
            key="slider_days",
            help="Number of consecutive days required for presence alert.",
        )

    st.divider()
    st.header("👥 User Registry")
    uploaded_users = st.file_uploader("Upload Staff List (CSV)", type=["csv"], key="user_upload")

# ══════════════════════════════════════════════════════════════
# 7. MAIN TITLE & STATIC DATA LOAD
# ══════════════════════════════════════════════════════════════

st.title(f"🐘 {config.APP_TITLE}")

geojson_features = load_map_data_as_geojson(".")
village_df       = load_village_data("centroids.csv")

if geojson_features:
    with st.expander(f"✅ Map Data Loaded ({len(geojson_features)} regions)"):
        st.write("Boundary data ready.")

if village_df is not None:
    st.sidebar.success(f"✅ Village Data Loaded ({len(village_df)} villages)")
else:
    st.sidebar.warning("⚠️ 'centroids.csv' not found. Village analysis disabled.")

# ══════════════════════════════════════════════════════════════
# 8. FILE UPLOAD & PROCESSING
# ══════════════════════════════════════════════════════════════

uploaded_csv = st.file_uploader("Upload Sightings Report (CSV)", type=["csv"])

if uploaded_csv is not None:

    # ── A. LOAD & CLEAN ───────────────────────────────────────
    df_raw, load_warnings = load_and_clean_sightings(uploaded_csv)

    for w in load_warnings:
        st.sidebar.warning(w)

    # ── B. VILLAGE PROXIMITY ──────────────────────────────────
    if village_df is not None:
        _hash = str(len(df_raw)) + str(df_raw["Date"].max())
        df_raw = cached_calculate_affected_villages(
            _hash, df_raw, village_df, config.AFFECTED_VILLAGE_RADIUS_KM
        )
        df_raw["Near Village"] = df_raw["Affected Villages"] != "None"

    # ── C. SIDEBAR FILTERS ────────────────────────────────────
    with st.sidebar:
        st.header("Filters")
        min_date = df_raw["Date"].min().date()
        max_date = df_raw["Date"].max().date()
        date_range = st.date_input("Date Range", [min_date, max_date],
                                    min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start, end = date_range
        else:
            start, end = min_date, max_date

        df = df_raw[
            (df_raw["Date"].dt.date >= start) &
            (df_raw["Date"].dt.date <= end)
        ].copy()

        # Comparison period (same length, immediately prior)
        delta_days = (end - start).days + 1
        prev_end   = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=delta_days - 1)
        df_prev = df_raw[
            (df_raw["Date"].dt.date >= prev_start) &
            (df_raw["Date"].dt.date <= prev_end)
        ].copy()

        divisions  = ["All"] + sorted(df["Division"].unique().tolist())
        sel_div    = st.selectbox("Filter Division", divisions)
        if sel_div != "All":
            df      = df[df["Division"] == sel_div]
            df_prev = df_prev[df_prev["Division"] == sel_div]

        ranges    = ["All"] + sorted(df["Range"].unique().tolist())
        sel_range = st.selectbox("Filter Range", ranges)
        if sel_range != "All":
            df      = df[df["Range"] == sel_range]
            df_prev = df_prev[df_prev["Range"] == sel_range]

        st.divider()
        st.header("Map Settings")
        map_mode = st.radio("Visualization Mode:", ["Pins", "Heatmap"], horizontal=True)

    # ── D. OPERATIONAL METRICS (Row 1) ────────────────────────
    n_sightings      = len(df)
    n_cumulative     = int(df["Total Count"].sum())
    n_direct         = int((df["Sighting Type"] == "Direct").sum()) if "Sighting Type" in df.columns else 0
    n_males          = int(df["Male Count"].sum())
    n_calves         = int(df["Calf Count"].sum())
    n_conflict_events = int(
        ((df["Crop Damage"] > 0) | (df["House Damage"] > 0) | (df["Injury"] > 0)).sum()
    )

    st.markdown("### 📈 Operational Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def _metric_btn(col, label, value, filter_val, btn_key):
        col.metric(label, value)
        if col.button("View on map", key=btn_key):
            st.session_state.map_filter = filter_val
            st.rerun()

    _metric_btn(c1, "📝 Entries",    n_sightings,       "All",      "btn_entries")
    _metric_btn(c2, "🐾 Cumulative", n_cumulative,      "All",      "btn_cumulative")
    _metric_btn(c3, "👁️ Direct",     n_direct,          "Direct",   "btn_direct")
    _metric_btn(c4, "🔥 Conflicts",  n_conflict_events, "Conflict", "btn_conflict")
    _metric_btn(c5, "♂️ Males",      n_males,           "Males",    "btn_males")
    _metric_btn(c6, "👶 Calves",     n_calves,          "Calves",   "btn_calves")

    # ── E. STRATEGIC INDICATORS (Row 2) ───────────────────────
    st.markdown("### 🛡️ Strategic Indicators")
    k1, k2, k3, k4 = st.columns(4)

    prev_sight = len(df_prev)

    curr_sev  = df["Severity Score"].sum()
    prev_sev  = df_prev["Severity Score"].sum()
    k1.metric("🚨 Severity Score", f"{curr_sev:.1f}", delta=f"{curr_sev - prev_sev:+.1f}")
    if k1.button("View severity map", key="btn_sev"):
        st.session_state.map_filter = "Severity_View"
        st.rerun()

    curr_hec = (n_conflict_events / n_sightings * 100) if n_sightings > 0 else 0
    prev_conf = int(((df_prev["Crop Damage"] > 0) | (df_prev["House Damage"] > 0) | (df_prev["Injury"] > 0)).sum())
    prev_hec  = (prev_conf / prev_sight * 100) if prev_sight > 0 else 0
    k2.metric("⚠️ HEC Ratio", f"{curr_hec:.1f}%", delta=f"{curr_hec - prev_hec:+.1f}%")
    if k2.button("View conflict map", key="btn_hec"):
        st.session_state.map_filter = "Conflict"
        st.rerun()

    valid_hours   = df[df["Hour"] != -1]
    curr_night    = (valid_hours["Is_Night"].sum() / len(valid_hours) * 100) if len(valid_hours) > 0 else 0
    prev_valid    = df_prev[df_prev["Hour"] != -1]
    prev_night    = (prev_valid["Is_Night"].sum() / len(prev_valid) * 100) if len(prev_valid) > 0 else 0
    k3.metric("🌙 Night Activity", f"{curr_night:.1f}%", delta=f"{curr_night - prev_night:+.1f}%")
    if k3.button("View night map", key="btn_night"):
        st.session_state.map_filter = "Night_View"
        st.rerun()

    if not df.empty:
        beat_stats = df.groupby("Beat")["Severity Score"].sum().reset_index()
        beat_stats  = beat_stats.sort_values("Severity Score", ascending=False)
        top_beat    = beat_stats.iloc[0]["Beat"] if beat_stats.iloc[0]["Severity Score"] > 0 else "None"
        st.session_state.hotspot_beat = top_beat if top_beat != "None" else None
    else:
        top_beat = "None"

    k4.metric("🔁 Hotspot Beat", top_beat)
    if k4.button("Focus hotspot", key="btn_hotspot"):
        st.session_state.map_filter = "Hotspot_View"
        st.rerun()

    # Single explanation panel (was duplicated)
    filter_labels = {
        "Severity_View": "🔴 **Viewing Severity:** Markers sized by Severity Score.",
        "Night_View":    "🟣 **Viewing Night Activity:** Sightings between 6 PM and 6 AM.",
        "Hotspot_View":  f"🎯 **Viewing Hotspot:** Focusing on **{top_beat}** (Highest Severity).",
        "Conflict":      "🔥 **Viewing Conflicts:** Crop, House, or Injury incidents.",
        "Direct":        "👁️ **Viewing Direct Sightings.**",
        "Males":         "♂️ **Viewing Male sightings.**",
        "Calves":        "👶 **Viewing Calf sightings.**",
    }
    st.info(filter_labels.get(st.session_state.map_filter, "📍 **Viewing All Data.**"))

    # ══════════════════════════════════════════════════════════
    # 9. TABS
    # ══════════════════════════════════════════════════════════

    st.divider()
    tab_map, tab_charts, tab_refuge, tab_herds, tab_data, tab_staff = st.tabs([
        "🗺️ Live Map", "📊 Analytics", "🐘 Daytime Refuges", "🐘 Herd Classification", "📋 Data & Reports", "👥 Staff Registry"
    ])

    # ── TAB 1: MAP ────────────────────────────────────────────
    with tab_map:
        map_df = df.copy()
        _hash  = str(len(map_df)) + str(map_df["Date"].max())
        risk_villages = (
            cached_identify_risk_villages(
                _hash, map_df, village_df,
                p_dmg_rad, p_pres_rad, p_days
            )
            if village_df is not None else []
        )

        c_map, c_list = st.columns([3, 1])

        with c_list:
            st.markdown("### 🛰️ ACTIVE INTEL")
            if st.button("📡 RESET SYSTEM VIEW", use_container_width=True):
                st.session_state.selected_village = None
                st.rerun()
            st.divider()

            if not risk_villages:
                st.info("No active threats.")
            else:
                for i, v in enumerate(risk_villages):
                    is_active = (v["Village"] == st.session_state.selected_village)
                    btn_type  = "primary" if is_active else "secondary"
                    if st.button(
                        f"{'📍' if is_active else '🏘️'} {v['Village']}\n({v['Reason']})",
                        key=f"hud_v_{i}", use_container_width=True, type=btn_type,
                    ):
                        st.session_state.selected_village = v["Village"]
                        st.rerun()

        with c_map:
            # Auto-centre on filtered data; fall back to config default
            if not map_df.empty:
                map_center = [map_df["Latitude"].mean(), map_df["Longitude"].mean()]
            else:
                map_center = config.DEFAULT_MAP_CENTER

            # ── Apply map_filter to select which rows to plot ──
            active_filter = st.session_state.map_filter
            if active_filter == "Direct":
                plot_df = map_df[map_df.get("Sighting Type", pd.Series()) == "Direct"] \
                          if "Sighting Type" in map_df.columns else map_df
            elif active_filter == "Conflict":
                plot_df = map_df[
                    (map_df["Crop Damage"] > 0) |
                    (map_df["House Damage"] > 0) |
                    (map_df["Injury"] > 0)
                ]
            elif active_filter == "Males":
                plot_df = map_df[map_df["Male Count"] > 0]
            elif active_filter == "Calves":
                plot_df = map_df[map_df["Calf Count"] > 0]
            elif active_filter == "Night_View":
                plot_df = map_df[map_df["Is_Night"] == 1]
            elif active_filter == "Hotspot_View" and st.session_state.hotspot_beat:
                plot_df = map_df[map_df["Beat"] == st.session_state.hotspot_beat]
            else:
                plot_df = map_df  # All / Severity_View uses all rows

            # Re-centre on filtered subset when a specific filter is active
            if active_filter not in ("All", None) and not plot_df.empty:
                map_center = [plot_df["Latitude"].mean(), plot_df["Longitude"].mean()]
                zoom = 11  # zoom in on the filtered area
            else:
                zoom = config.DEFAULT_MAP_ZOOM

            m = folium.Map(location=map_center, zoom_start=zoom,
                           tiles="CartoDB dark_matter")

            folium.TileLayer(
                tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                attr="Google", name="🛰️ Satellite Hybrid", overlay=False,
            ).add_to(m)
            folium.TileLayer("CartoDB dark_matter", name="🌃 Tactical Dark", overlay=False).add_to(m)

            sector_grp = folium.FeatureGroup(name="🗺️ Sector Boundaries", show=True).add_to(m)
            refuge_grp = folium.FeatureGroup(name="🟢 Active Refuges",    show=True).add_to(m)
            threat_grp = folium.FeatureGroup(name="🚨 Threat Matrix",     show=True).add_to(m)
            filter_grp = folium.FeatureGroup(name="🔎 Filtered View",     show=True).add_to(m)

            if geojson_features:
                folium.GeoJson(
                    data={"type": "FeatureCollection", "features": geojson_features},
                    style_function=lambda x: {
                        "fillColor": "#00f2ff", "color": "#00f2ff",
                        "weight": 1, "fillOpacity": 0.05,
                    },
                ).add_to(sector_grp)

            ref_df = cached_identify_daytime_refuges(_hash, df)
            if not ref_df.empty:
                for _, ref in ref_df.head(10).iterrows():
                    folium.CircleMarker(
                        location=[ref["Latitude"], ref["Longitude"]],
                        radius=ref["Persistence Score"] * 1.5,
                        color="#39FF14", fill=True,
                        fill_color="#39FF14", fill_opacity=0.3,
                        tooltip=f"<b>REFUGE: {ref['Beat']}</b>",
                    ).add_to(refuge_grp)

            # Always show high-severity threats
            for _, row in map_df[map_df["Severity Score"] > 5].iterrows():
                folium.CircleMarker(
                    location=[row["Latitude"], row["Longitude"]],
                    radius=8, color="#FF3131", fill=True,
                    fill_color="#FF3131", fill_opacity=0.8,
                    popup=f"SEVERITY: {row['Severity Score']}",
                ).add_to(threat_grp)

            # ── Render filtered markers ────────────────────────
            if active_filter == "Severity_View":
                # Size markers proportional to severity score
                for _, row in plot_df.iterrows():
                    r = max(4, min(20, row["Severity Score"] * 1.5))
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=r, color="#FF3131", fill=True,
                        fill_color="#FF3131",
                        fill_opacity=0.6,
                        tooltip=f"Severity: {row['Severity Score']:.1f} | {row.get('Beat','')}"
                    ).add_to(filter_grp)
            elif active_filter == "Night_View":
                for _, row in plot_df.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=6, color="#9B59B6", fill=True,
                        fill_color="#9B59B6", fill_opacity=0.7,
                        tooltip=f"Hour: {row['Hour']}:00 | {row.get('Beat','')}"
                    ).add_to(filter_grp)
            elif active_filter == "Conflict":
                for _, row in plot_df.iterrows():
                    tip = (f"Crop: {int(row['Crop Damage'])} | "
                           f"House: {int(row['House Damage'])} | "
                           f"Injury: {int(row['Injury'])} | {row.get('Beat','')}")
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=9, color="#FF3131", fill=True,
                        fill_color="#FF6B35", fill_opacity=0.85,
                        tooltip=tip
                    ).add_to(filter_grp)
            elif active_filter in ("Direct", "Males", "Calves"):
                color_map = {"Direct": "#00f2ff", "Males": "#1f77b4", "Calves": "#2ca02c"}
                c = color_map.get(active_filter, "#ffffff")
                for _, row in plot_df.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=6, color=c, fill=True,
                        fill_color=c, fill_opacity=0.75,
                        tooltip=f"{active_filter} | Count: {int(row['Total Count'])} | {row.get('Beat','')}"
                    ).add_to(filter_grp)
            elif active_filter == "Hotspot_View":
                for _, row in plot_df.iterrows():
                    folium.CircleMarker(
                        location=[row["Latitude"], row["Longitude"]],
                        radius=7, color="#EF9F27", fill=True,
                        fill_color="#EF9F27", fill_opacity=0.8,
                        tooltip=f"Beat: {row.get('Beat','')} | Severity: {row['Severity Score']:.1f}"
                    ).add_to(filter_grp)
            else:
                # All — use heatmap or pins based on sidebar setting
                if map_mode == "Heatmap" and not plot_df.empty:
                    heat_data = plot_df[["Latitude", "Longitude"]].dropna().values.tolist()
                    HeatMap(heat_data, radius=15, blur=20).add_to(filter_grp)
                else:
                    for _, row in plot_df.iterrows():
                        folium.CircleMarker(
                            location=[row["Latitude"], row["Longitude"]],
                            radius=5, color="#00f2ff", fill=True,
                            fill_color="#00f2ff", fill_opacity=0.5,
                            tooltip=f"{row.get('Beat','')} | {row.get('Sighting Type','')}"
                        ).add_to(filter_grp)

            legend_html = """
            <div style="position:fixed;bottom:30px;left:30px;width:185px;
                        background:rgba(15,15,15,0.85);color:#00f2ff;
                        border:1px solid #00f2ff;z-index:9999;
                        font-family:'Courier New',monospace;
                        padding:10px;border-radius:2px;font-size:11px;">
            <b style="letter-spacing:2px;">HUD: MONITORING</b>
            <hr style="border-color:#00f2ff;opacity:.3;">
            <span style="color:#39FF14;">●</span> REFUGE_DETECTED<br>
            <span style="color:#FF3131;">●</span> THREAT_INCIDENT<br>
            <span style="color:#9B59B6;">●</span> NIGHT_ACTIVITY<br>
            <span style="color:#EF9F27;">●</span> HOTSPOT_BEAT<br>
            <span style="color:#00f2ff;">—</span> SECTOR_LIMIT
            </div>"""
            m.get_root().html.add_child(folium.Element(legend_html))
            folium.LayerControl(position="topright", collapsed=False).add_to(m)

            st_folium(m, width=None, height=650, use_container_width=True)

    # ── TAB 2: ANALYTICS ──────────────────────────────────────
    with tab_charts:
        st.subheader("📊 Analytics Dashboard")

        # Initialise chart variables so report generator always has valid refs
        fig_sun    = None
        fig_hourly = None
        fig_trend  = None
        fig_demog  = None
        fig_damage = None

        r1c1, r1c2 = st.columns(2)

        with r1c1:
            st.markdown("**Hierarchy Drill-Down**")
            if not df.empty:
                sb_df  = df.copy()
                sb_val = "Total Count"
                sb_title = "Sighting Distribution"

                if st.session_state.map_filter == "Conflict":
                    sb_df = sb_df[(sb_df["Crop Damage"] > 0) | (sb_df["House Damage"] > 0) | (sb_df["Injury"] > 0)]
                    sb_df = sb_df.copy(); sb_df["Incidents"] = 1; sb_val = "Incidents"; sb_title = "Conflict Hierarchy"
                elif st.session_state.map_filter == "Direct":
                    sb_df = sb_df[sb_df["Sighting Type"] == "Direct"]; sb_title = "Direct Sightings"
                elif st.session_state.map_filter == "Males":
                    sb_df = sb_df[sb_df["Male Count"] > 0]; sb_val = "Male Count"; sb_title = "Male Population"

                if not sb_df.empty:
                    fig_sun = px.sunburst(sb_df, path=["Division", "Range", "Beat"],
                                          values=sb_val, title=sb_title)
                    st.plotly_chart(fig_sun, use_container_width=True)
                else:
                    st.info("No data for this view.")
            else:
                st.info("No data available.")

        with r1c2:
            st.markdown("**Hourly Activity (0–24 h)**")
            valid_hour_df = df[df["Hour"] != -1]
            if not valid_hour_df.empty:
                h_counts = (
                    valid_hour_df["Hour"]
                    .value_counts()
                    .reindex(range(24), fill_value=0)
                    .reset_index()
                )
                h_counts.columns = ["Hour", "Count"]
                fig_hourly = px.bar(h_counts, x="Hour", y="Count",
                                    title="Activity Peaks",
                                    color="Count", color_continuous_scale="Viridis")
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.info("No time data available.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Trend Analysis**")
            if not df.empty:
                daily     = df.groupby("Date").size().reset_index(name="Count")
                fig_trend = px.line(daily, x="Date", y="Count", markers=True,
                                    title="Daily Sighting Trend")
                st.plotly_chart(fig_trend, use_container_width=True)

        with c2:
            st.markdown("**Demographics**")
            if not df.empty:
                demog = (
                    df[["Male Count", "Female Count", "Calf Count", "Unknown Count"]]
                    .sum()
                    .reset_index()
                )
                demog.columns = ["Type", "Count"]
                demog = demog[demog["Count"] > 0]
                fig_demog = px.pie(
                    demog, values="Count", names="Type", hole=0.4,
                    color="Type",
                    color_discrete_map={
                        "Male Count":    "#1f77b4",
                        "Female Count":  "#e377c2",
                        "Calf Count":    "#2ca02c",
                        "Unknown Count": "#7f7f7f",
                    },
                )
                st.plotly_chart(fig_demog, use_container_width=True)

        c4 = st.columns(1)[0]
        with c4:
            st.markdown("**Conflict Type Breakdown**")
            damage_sums = (
                df[["Crop Damage", "House Damage", "Injury"]]
                .apply(lambda x: (x > 0).sum())
                .reset_index()
            )
            damage_sums.columns = ["Damage Type", "Incidents"]
            damage_sums = damage_sums[damage_sums["Incidents"] > 0]
            if not damage_sums.empty:
                fig_damage = px.pie(
                    damage_sums, values="Incidents", names="Damage Type",
                    color="Damage Type",
                    color_discrete_map={
                        "Crop Damage":  "orange",
                        "House Damage": "red",
                        "Injury":       "darkred",
                    },
                )
                st.plotly_chart(fig_damage, use_container_width=True)
            else:
                st.info("No conflicts reported.")

        st.divider()
        st.subheader("🏆 Division Comparison: Cumulative Reports")

        if not df.empty:
            div_daily = df.groupby(["Date", "Division"]).size().reset_index(name="Daily")
            div_pivot = (
                div_daily
                .pivot(index="Date", columns="Division", values="Daily")
                .fillna(0)
                .reindex(pd.date_range(div_daily["Date"].min(), div_daily["Date"].max()), fill_value=0)
            )
            div_cum  = div_pivot.cumsum()
            div_long = div_cum.stack().reset_index()
            div_long.columns = ["Date", "Division", "Cumulative Reports"]

            # Use weekly frames to avoid 365-frame lag
            div_long["Week"] = div_long["Date"].dt.to_period("W").dt.start_time
            div_weekly = div_long.groupby(["Week", "Division"])["Cumulative Reports"].max().reset_index()
            div_weekly["Week_Str"] = div_weekly["Week"].dt.strftime("%Y-%m-%d")

            fig_div_anim = px.bar(
                div_weekly, x="Division", y="Cumulative Reports",
                color="Division", animation_frame="Week_Str",
                range_y=[0, div_weekly["Cumulative Reports"].max() * 1.1],
                title="Cumulative Reports by Division (weekly)",
            )
            fig_div_anim.update_layout(showlegend=False)
            fig_div_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
            st.plotly_chart(fig_div_anim, use_container_width=True)

        st.divider()
        st.subheader("⏳ Temporal Dynamics: Range Growth Animation")

        if not df.empty:
            div_options       = sorted(df["Division"].unique().tolist())
            selected_div_anim = st.selectbox("Select Division:", div_options, key="anim_div_filter")
            subset            = df[df["Division"] == selected_div_anim].copy()

            if not subset.empty:
                daily_counts = subset.groupby(["Date", "Range"]).size().reset_index(name="Daily")
                pivoted      = (
                    daily_counts
                    .pivot(index="Date", columns="Range", values="Daily")
                    .fillna(0)
                    .reindex(pd.date_range(daily_counts["Date"].min(), daily_counts["Date"].max()), fill_value=0)
                )
                cum_pivoted = pivoted.cumsum()
                cum_long    = cum_pivoted.stack().reset_index()
                cum_long.columns = ["Date", "Range", "Cumulative Entries"]

                # Weekly frames
                cum_long["Week"]     = cum_long["Date"].dt.to_period("W").dt.start_time
                cum_weekly           = cum_long.groupby(["Week", "Range"])["Cumulative Entries"].max().reset_index()
                cum_weekly["Week_Str"] = cum_weekly["Week"].dt.strftime("%Y-%m-%d")

                fig_anim = px.bar(
                    cum_weekly, x="Range", y="Cumulative Entries",
                    color="Range", animation_frame="Week_Str",
                    range_y=[0, cum_weekly["Cumulative Entries"].max() * 1.1],
                    title=f"Cumulative Entry Growth by Range ({selected_div_anim})",
                )
                fig_anim.update_layout(showlegend=False)
                fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 200
                st.plotly_chart(fig_anim, use_container_width=True)
            else:
                st.warning(f"No data for Division: {selected_div_anim}")

    # ── TAB 3: DAYTIME REFUGES ────────────────────────────────
    with tab_refuge:
        st.subheader("🐘 Daytime Refuge Analysis")

        # ── Mode selector ─────────────────────────────────────
        refuge_mode = st.radio(
            "Analysis mode",
            ["📋 Entry-based (logged sightings)", "🗺️ Spatial inference (conflict proximity)"],
            horizontal=True,
            key="refuge_mode",
        )

        st.divider()

        # ══════════════════════════════════════════════════════
        # MODE A: Entry-based (original logic)
        # ══════════════════════════════════════════════════════
        if refuge_mode == "📋 Entry-based (logged sightings)":
            st.markdown(
                "Identifies staging beats from **logged field entries** during daylight hours "
                f"({config.REFUGE_DAY_START_HOUR}:00–{config.REFUGE_DAY_END_HOUR}:00) "
                "with low conflict scores. Prioritises foraging signs (broken branches, dung). "
                "Only beats with actual sighting records appear here."
            )

            refuge_df = cached_identify_daytime_refuges(_hash, df)

            if not refuge_df.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_refuge = px.bar(
                        refuge_df.head(10),
                        x="Beat",
                        y="Persistence Score",
                        color="Avg Group Size",
                        title="Top 10 Daytime Staging Beats (by Persistence Score)",
                        hover_data=["Division", "Range", "Sighting Frequency"],
                        color_continuous_scale="Greens",
                    )
                    st.plotly_chart(fig_refuge, use_container_width=True)

                with col2:
                    st.metric("Primary Refuge Beat", refuge_df.iloc[0]["Beat"])
                    st.write("**High-Confidence Refuges**")
                    st.dataframe(
                        refuge_df[["Division", "Range", "Beat", "Persistence Score"]].head(15),
                        hide_index=True,
                        use_container_width=True,
                    )

                st.info(
                    "💡 **Operational Insight:** Direct morning patrols toward these Beats "
                    "to intercept herds before they move toward agricultural fields at dusk."
                )
            else:
                st.warning(
                    f"Insufficient daylight data ({config.REFUGE_DAY_START_HOUR}:00–"
                    f"{config.REFUGE_DAY_END_HOUR}:00) within the selected filters to identify refuges."
                )

        # ══════════════════════════════════════════════════════
        # MODE B: Spatial inference from conflict proximity
        # ══════════════════════════════════════════════════════
        else:
            st.markdown(
                "Infers likely refuge zones **purely from spatial relationships** — "
                "no daytime field entry is required at the candidate location. "
                "The model finds areas within striking distance of conflict sites "
                "that are *not* themselves conflict zones, ranking them by how much "
                "accumulated conflict pressure surrounds them. "
                "Logged daytime observations (if any) add a secondary boost."
            )

            with st.expander("⚙️ Inference parameters", expanded=False):
                col_p1, col_p2 = st.columns(2)
                search_radius_km = col_p1.slider(
                    "Search radius around conflicts (km)",
                    min_value=2.0, max_value=20.0,
                    value=config.REFUGE_CONFLICT_SEARCH_RADIUS_KM,
                    step=0.5, key="inf_search_radius",
                    help="How far from a conflict event to look for refuge zones.",
                )
                grid_res_km = col_p2.slider(
                    "Grid resolution (km)",
                    min_value=0.25, max_value=2.0,
                    value=config.REFUGE_SPATIAL_GRID_RESOLUTION_KM,
                    step=0.25, key="inf_grid_res",
                    help="Smaller = finer map, slower compute.",
                )
                top_n = col_p1.slider(
                    "Top N candidate zones to show",
                    min_value=5, max_value=50, value=20, step=5,
                    key="inf_top_n",
                )

            infer_hash = f"{_hash}_{search_radius_km}_{grid_res_km}"
            with st.spinner("Running spatial inference…"):
                inf_df = cached_infer_refuges_from_conflict_proximity(
                    infer_hash, df, search_radius_km, grid_res_km
                )

            if inf_df.empty:
                st.warning(
                    "No conflict events found in the filtered data — "
                    "spatial inference requires at least one crop/house/injury record."
                )
            else:
                top_inf = inf_df.head(top_n).copy()

                # ── Summary metrics ───────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Candidate zones identified", len(inf_df))
                m2.metric(
                    "Top zone — nearest beat",
                    top_inf.iloc[0]["Nearest Beat"],
                )
                m3.metric(
                    "Conflicts near top zone",
                    f"{top_inf.iloc[0]['Nearby Conflict Count']:.0f}",
                )

                st.divider()

                # ── How to read this ──────────────────────────
                with st.expander("📖 How to read this analysis", expanded=True):
                    st.markdown("""
**What the model does:**
It draws a grid across the landscape around every conflict site and scores each cell based on:
- How many conflict events are nearby (crop raids, house damage, injuries)
- How severe those conflicts are (weighted by type)
- Whether the cell itself is a conflict site — if yes, score is reduced 80% (elephants don't rest where they raid)
- Whether any ranger actually logged a daytime observation nearby (adds a 30% boost)

**How to interpret the map:**
- 🔴 **Red dots** = actual conflict sites (crop/house/injury events in your data)
- 🟡→🟠 **Yellow-orange circles** = inferred refuge zones ranked by Combined Score
- **Circle size** = total severity of nearby conflicts (bigger = more pressure around that zone)
- **Colour intensity** = Combined Score (darker = higher confidence this is a refuge)

**What to do with it:**
Zones with **high Combined Score but zero Observation Boost** are unsurveyed areas the model flags as likely refuges. These are your highest-priority new morning patrol locations.
                    """)

                # ── Interactive Folium map ────────────────────
                st.markdown("**Interactive refuge zone map** — zoom, click markers for details")

                conflict_pts = df[
                    (df["Crop Damage"] > 0) | (df["House Damage"] > 0) | (df["Injury"] > 0)
                ].copy()

                ref_map_center = [
                    float(top_inf["Latitude"].mean()),
                    float(top_inf["Longitude"].mean()),
                ]
                ref_m = folium.Map(
                    location=ref_map_center,
                    zoom_start=10,
                    tiles="CartoDB dark_matter",
                )
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                    attr="Google", name="🛰️ Satellite", overlay=False,
                ).add_to(ref_m)
                folium.TileLayer("CartoDB dark_matter", name="🌃 Dark", overlay=False).add_to(ref_m)

                refuge_zone_grp   = folium.FeatureGroup(name="🟡 Inferred Refuges", show=True).add_to(ref_m)
                conflict_site_grp = folium.FeatureGroup(name="🔴 Conflict Sites",   show=True).add_to(ref_m)

                # Normalise score to [0,1] for colour intensity
                max_score = top_inf["Combined Score"].max()

                def _score_to_color(score, max_s):
                    """Map normalised score to yellow→red hex."""
                    t = score / max(max_s, 1e-9)
                    r = int(255)
                    g = int(200 * (1 - t))
                    b = 0
                    return f"#{r:02x}{g:02x}{b:02x}"

                for _, zone in top_inf.iterrows():
                    color = _score_to_color(zone["Combined Score"], max_score)
                    obs   = zone["Observation Boost"]
                    obs_tag = f"✅ {obs:.0f} logged observations" if obs > 0 else "⚠️ No logged observations (unsurveyed)"
                    popup_html = f"""
                    <div style='font-family:sans-serif;font-size:13px;min-width:200px'>
                    <b>Nearest Beat:</b> {zone['Nearest Beat']}<br>
                    <b>Distance to beat:</b> {zone['Nearest Beat Distance (km)']:.1f} km<br>
                    <hr style='margin:4px 0'>
                    <b>Combined Score:</b> {zone['Combined Score']:.3f}<br>
                    <b>Conflict Attraction:</b> {zone['Conflict Attraction Score']:.2f}<br>
                    <b>{obs_tag}</b><br>
                    <hr style='margin:4px 0'>
                    <b>Nearby conflicts:</b> {zone['Nearby Conflict Count']:.0f}<br>
                    <b>Nearby severity:</b> {zone['Nearby Conflict Severity']:.1f}
                    </div>"""
                    radius = 6 + zone["Nearby Conflict Severity"] / 5
                    folium.CircleMarker(
                        location=[zone["Latitude"], zone["Longitude"]],
                        radius=min(radius, 20),
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.6 + 0.35 * (zone["Combined Score"] / max(max_score, 1e-9)),
                        popup=folium.Popup(popup_html, max_width=260),
                        tooltip=f"Score: {zone['Combined Score']:.3f} | Beat: {zone['Nearest Beat']}",
                    ).add_to(refuge_zone_grp)

                # Conflict sites as red markers
                for _, cp in conflict_pts.iterrows():
                    popup_html = f"""
                    <div style='font-family:sans-serif;font-size:13px'>
                    <b>Conflict site</b><br>
                    Beat: {cp.get('Beat','?')}<br>
                    Crop: {int(cp['Crop Damage'])} | House: {int(cp['House Damage'])} | Injury: {int(cp['Injury'])}<br>
                    Severity: {cp['Severity Score']:.1f}
                    </div>"""
                    folium.CircleMarker(
                        location=[cp["Latitude"], cp["Longitude"]],
                        radius=7,
                        color="#E24B4A",
                        fill=True,
                        fill_color="#E24B4A",
                        fill_opacity=0.85,
                        popup=folium.Popup(popup_html, max_width=220),
                        tooltip=f"Conflict | Severity: {cp['Severity Score']:.1f} | {cp.get('Beat','')}",
                    ).add_to(conflict_site_grp)

                folium.LayerControl(position="topright", collapsed=False).add_to(ref_m)
                st_folium(ref_m, width=None, height=550, use_container_width=True)

                # ── Score breakdown chart ─────────────────────
                st.markdown("**Score breakdown** — each dot is one candidate zone")
                fig_scores = px.scatter(
                    top_inf,
                    x="Conflict Attraction Score",
                    y="Observation Boost",
                    color="Combined Score",
                    size="Nearby Conflict Count",
                    hover_name="Nearest Beat",
                    color_continuous_scale="YlOrRd",
                    title="Conflict Attraction vs Ranger Observations  (bubble = nearby conflict count)",
                    labels={
                        "Conflict Attraction Score": "← Spatial attraction from conflicts →",
                        "Observation Boost":         "← Direct observation evidence →",
                    },
                )
                fig_scores.add_annotation(
                    x=top_inf["Conflict Attraction Score"].max() * 0.7,
                    y=top_inf["Observation Boost"].max() * 0.05,
                    text="High attraction, unsurveyed<br>→ Priority patrol zones",
                    showarrow=False,
                    font=dict(size=11, color="#EF9F27"),
                    align="center",
                )
                st.plotly_chart(fig_scores, use_container_width=True)

                # ── Detail table ──────────────────────────────
                with st.expander("📋 Full candidate zone table"):
                    display_cols = [
                        "Nearest Beat", "Nearest Beat Distance (km)",
                        "Combined Score", "Conflict Attraction Score",
                        "Observation Boost",
                        "Nearby Conflict Count", "Nearby Conflict Severity",
                        "Latitude", "Longitude",
                    ]
                    st.dataframe(
                        top_inf[display_cols]
                        .style.background_gradient(subset=["Combined Score"], cmap="YlOrRd"),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.info(
                    "💡 **Priority patrol zones:** Look for dots in the bottom-right of the scatter chart — "
                    "high conflict attraction but zero observation boost. "
                    "These are areas the model flags as likely refuges that rangers have **never surveyed**."
                )

    # ── TAB 4: HERD CLASSIFICATION ────────────────────────────
    with tab_herds:
        st.subheader("🐘 Herd Classification")
        st.markdown(
            "Groups individual sighting records into discrete herd events using "
            "spatiotemporal chain-linking, then classifies each herd by **composition**, "
            "**movement pattern**, **temporal activity**, and **conflict risk**."
        )

        with st.expander("⚙️ Classification parameters", expanded=False):
            hc1, hc2, hc3 = st.columns(3)
            h_spatial = hc1.slider(
                "Spatial gap (km)", 0.5, 10.0,
                config.HERD_SPATIAL_GAP_KM, 0.5,
                key="h_spatial",
                help="Max distance between consecutive records to stay in the same herd.",
            )
            h_temporal = hc2.slider(
                "Temporal gap (hrs)", 1, 48,
                config.HERD_TEMPORAL_GAP_HOURS, 1,
                key="h_temporal",
                help="Max hours between consecutive records to stay in the same herd.",
            )
            h_min_size = hc3.slider(
                "Min herd size", 1, 10,
                config.HERD_MIN_SIZE, 1,
                key="h_min_size",
                help="Minimum Total Count to include a record.",
            )

        herd_hash = f"{_hash}_{h_spatial}_{h_temporal}_{h_min_size}"
        with st.spinner("Classifying herds…"):
            df_with_hids, herds_df = cached_classify_herds(
                herd_hash, df, h_spatial, h_temporal, h_min_size
            )

        if herds_df.empty:
            st.warning("No herds could be classified with current parameters and filters.")
        else:
            # ── Summary metrics ───────────────────────────────
            n_herds     = len(herds_df)
            n_critical  = int((herds_df["Conflict Risk"] == "Critical").sum())
            n_high      = int((herds_df["Conflict Risk"] == "High").sum())
            n_raiding   = int((herds_df["Movement"] == "Raiding").sum())
            avg_size    = herds_df["Total Count (max)"].mean()

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Total Herds",       n_herds)
            sm2.metric("Critical Risk",     n_critical)
            sm3.metric("High Risk",         n_high)
            sm4.metric("Raiding Herds",     n_raiding)
            sm5.metric("Avg Herd Size",     f"{avg_size:.1f}")

            st.divider()

            # ── Row 1: Classification breakdown charts ────────
            rc1, rc2, rc3, rc4 = st.columns(4)

            with rc1:
                comp_counts = herds_df["Composition"].value_counts().reset_index()
                comp_counts.columns = ["Composition", "Count"]
                fig_comp = px.pie(
                    comp_counts, values="Count", names="Composition",
                    title="Herd Composition",
                    color="Composition",
                    color_discrete_map=config.HERD_COMPOSITION_COLORS,
                    hole=0.4,
                )
                fig_comp.update_layout(showlegend=True, height=280, margin=dict(t=40,b=0,l=0,r=0))
                st.plotly_chart(fig_comp, use_container_width=True)

            with rc2:
                mov_counts = herds_df["Movement"].value_counts().reset_index()
                mov_counts.columns = ["Movement", "Count"]
                fig_mov = px.bar(
                    mov_counts, x="Movement", y="Count",
                    title="Movement Patterns",
                    color="Movement",
                    color_discrete_map={
                        "Raiding":    "#E24B4A",
                        "Transiting": "#EF9F27",
                        "Ranging":    "#378ADD",
                        "Stationary": "#639922",
                    },
                )
                fig_mov.update_layout(showlegend=False, height=280, margin=dict(t=40,b=0,l=0,r=0))
                st.plotly_chart(fig_mov, use_container_width=True)

            with rc3:
                temp_counts = herds_df["Temporal Pattern"].value_counts().reset_index()
                temp_counts.columns = ["Temporal", "Count"]
                fig_temp = px.bar(
                    temp_counts, x="Temporal", y="Count",
                    title="Temporal Activity",
                    color="Temporal",
                    color_discrete_map={
                        "Nocturnal":   "#9B59B6",
                        "Diurnal":     "#F1C40F",
                        "Crepuscular": "#E67E22",
                        "Mixed":       "#95A5A6",
                        "Unknown":     "#7f7f7f",
                    },
                )
                fig_temp.update_layout(showlegend=False, height=280, margin=dict(t=40,b=0,l=0,r=0))
                st.plotly_chart(fig_temp, use_container_width=True)

            with rc4:
                risk_counts = herds_df["Conflict Risk"].value_counts().reindex(
                    ["Critical", "High", "Moderate", "Low"], fill_value=0
                ).reset_index()
                risk_counts.columns = ["Risk", "Count"]
                fig_risk = px.bar(
                    risk_counts, x="Risk", y="Count",
                    title="Conflict Risk",
                    color="Risk",
                    color_discrete_map=config.HERD_RISK_COLORS,
                )
                fig_risk.update_layout(showlegend=False, height=280, margin=dict(t=40,b=0,l=0,r=0),
                                        xaxis=dict(categoryorder="array",
                                                   categoryarray=["Critical","High","Moderate","Low"]))
                st.plotly_chart(fig_risk, use_container_width=True)

            st.divider()

            # ── Row 2: Map + scatter ──────────────────────────
            map_col, scatter_col = st.columns([3, 2])

            with map_col:
                st.markdown("**Herd centroids — coloured by Conflict Risk**")
                herd_map_center = [
                    herds_df["Centroid Lat"].mean(),
                    herds_df["Centroid Lon"].mean(),
                ]
                hm = folium.Map(location=herd_map_center, zoom_start=10,
                                tiles="CartoDB dark_matter")
                folium.TileLayer(
                    tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
                    attr="Google", name="🛰️ Satellite", overlay=False,
                ).add_to(hm)
                folium.TileLayer("CartoDB dark_matter", name="🌃 Dark", overlay=False).add_to(hm)

                # One feature group per risk level
                risk_groups = {}
                for risk_level, color in config.HERD_RISK_COLORS.items():
                    grp = folium.FeatureGroup(
                        name=f"● {risk_level}", show=True
                    ).add_to(hm)
                    risk_groups[risk_level] = (grp, color)

                for _, h in herds_df.iterrows():
                    grp, color = risk_groups.get(
                        h["Conflict Risk"], (hm, "#aaaaaa")
                    )
                    popup_html = f"""
                    <div style='font-family:sans-serif;font-size:13px;min-width:210px'>
                    <b>Herd {int(h['Herd ID'])}</b><br>
                    <b>Beat:</b> {h['Beat']}<br>
                    <b>Composition:</b> {h['Composition']}<br>
                    <b>Movement:</b> {h['Movement']}<br>
                    <b>Temporal:</b> {h['Temporal Pattern']}<br>
                    <b>Risk:</b> {h['Conflict Risk']}<br>
                    <hr style='margin:4px 0'>
                    <b>Size:</b> {int(h['Total Count (max)'])} &nbsp;
                    <b>Records:</b> {int(h['Records'])}<br>
                    <b>Duration:</b> {h['Duration (hrs)']:.1f} hrs<br>
                    <b>Displacement:</b> {h['Displacement (km)']:.1f} km<br>
                    <b>Severity:</b> {h['Severity Score (sum)']:.1f}
                    </div>"""
                    radius = 5 + min(int(h["Total Count (max)"]) * 0.8, 15)
                    folium.CircleMarker(
                        location=[h["Centroid Lat"], h["Centroid Lon"]],
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.75,
                        popup=folium.Popup(popup_html, max_width=250),
                        tooltip=(
                            f"Herd {int(h['Herd ID'])} | "
                            f"{h['Conflict Risk']} | "
                            f"{h['Composition']} | "
                            f"Size: {int(h['Total Count (max)'])}"
                        ),
                    ).add_to(grp)

                folium.LayerControl(position="topright", collapsed=False).add_to(hm)
                st_folium(hm, width=None, height=500, use_container_width=True)

            with scatter_col:
                st.markdown("**Herd size vs duration — by composition**")
                fig_scatter = px.scatter(
                    herds_df,
                    x="Duration (hrs)",
                    y="Total Count (max)",
                    color="Composition",
                    symbol="Movement",
                    size="Severity Score (sum)",
                    size_max=20,
                    hover_name="Beat",
                    hover_data={
                        "Conflict Risk":    True,
                        "Temporal Pattern": True,
                        "Displacement (km)": True,
                        "Records":          True,
                    },
                    color_discrete_map=config.HERD_COMPOSITION_COLORS,
                    title="Size × Duration (symbol = movement, size = severity)",
                )
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()

            # ── Herd table with risk filter ───────────────────
            st.markdown("**Herd event table**")
            risk_filter = st.multiselect(
                "Filter by Conflict Risk",
                ["Critical", "High", "Moderate", "Low"],
                default=["Critical", "High"],
                key="herd_risk_filter",
            )
            display_herds = herds_df[herds_df["Conflict Risk"].isin(risk_filter)] \
                            if risk_filter else herds_df

            table_cols = [
                "Herd ID", "Beat", "Division", "Range",
                "Composition", "Movement", "Temporal Pattern", "Conflict Risk",
                "Total Count (max)", "Duration (hrs)", "Records",
                "Displacement (km)", "Severity Score (sum)",
                "Crop Damage", "House Damage", "Injury",
                "Start Time", "End Time",
            ]
            st.dataframe(
                display_herds[table_cols]
                .style.applymap(
                    lambda v: f"color: {config.HERD_RISK_COLORS.get(v, 'inherit')}; font-weight:600",
                    subset=["Conflict Risk"]
                )
                .background_gradient(subset=["Severity Score (sum)"], cmap="Reds"),
                use_container_width=True,
                hide_index=True,
            )

    # ── TAB 5: DATA & REPORTS ─────────────────────────────────
    with tab_data:
        st.subheader("📋 Data Tables & Reports")

        t1, t2 = st.tabs(["⚠️ Damage Report", "🏆 Leaderboards"])
        with t1:
            dmg = (
                df.groupby(["Division", "Range"])
                .agg({
                    "Crop Damage":  lambda x: (x > 0).sum(),
                    "House Damage": lambda x: (x > 0).sum(),
                    "Injury":       lambda x: (x > 0).sum(),
                })
                .reset_index()
            )
            dmg = dmg[(dmg["Crop Damage"] > 0) | (dmg["House Damage"] > 0) | (dmg["Injury"] > 0)]
            dmg.columns = ["Division", "Range", "🌾 Crop", "🏠 House", "🚑 Injury"]
            st.dataframe(dmg.style.background_gradient(cmap="Reds"), use_container_width=True)

        with t2:
            l1, l2 = st.columns(2)
            with l1:
                st.markdown("**Top Beats (Activity)**")
                if "Beat" in df.columns:
                    st.dataframe(
                        df["Beat"].value_counts().reset_index(name="Entries").head(10),
                        use_container_width=True, hide_index=True,
                    )
            with l2:
                st.markdown("**Top Reporters**")
                if "Created By" in df.columns:
                    st.dataframe(
                        df["Created By"].value_counts().reset_index(name="Entries").head(10),
                        use_container_width=True, hide_index=True,
                    )

        st.divider()
        st.subheader("📄 Report Generation")

        if st.button("🖨️ Generate Full Report"):
            # Pass explicit figure refs — no locals() lookup
            html_report = generate_full_html_report(
                df, m,
                fig_trend, fig_demog, fig_hourly,
                fig_damage, fig_sun,
                start, end,
            )
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name="Elephant_Monitoring_Report.html",
                mime="text/html",
            )

    # ── TAB 5: STAFF REGISTRY ─────────────────────────────────
    with tab_staff:
        if uploaded_users is not None:
            st.title("👥 Gaj Rakshak Staff Analytics")

            try:
                u_df = pd.read_csv(uploaded_users)
            except UnicodeDecodeError:
                uploaded_users.seek(0)
                u_df = pd.read_csv(uploaded_users, encoding="ISO-8859-1")
            except Exception as e:
                st.error(f"Could not read staff CSV: {e}")
                u_df = pd.DataFrame()

            if not u_df.empty:
                u_df.columns = [c.strip() for c in u_df.columns]

                if {"Division", "Post"}.issubset(u_df.columns):
                    u_df["Division"] = u_df["Division"].astype(str).str.title().str.strip()
                    u_df["Post"]     = u_df["Post"].astype(str).str.title().str.strip()
                    if "Range" in u_df.columns:
                        u_df["Range"] = u_df["Range"].astype(str).str.title().str.strip()

                    div_counts  = u_df["Division"].value_counts()
                    small_divs  = div_counts[div_counts < config.MIN_STAFF_PER_DIVISION].index.tolist()

                    show_all = False
                    if small_divs:
                        st.info(
                            f"ℹ️ {len(small_divs)} division(s) have fewer than "
                            f"{config.MIN_STAFF_PER_DIVISION} staff: {', '.join(small_divs)}."
                        )
                        show_all = st.toggle("Include small divisions in charts", value=False)

                    u_df_viz = u_df if show_all else u_df[~u_df["Division"].isin(small_divs)]

                    if not u_df_viz.empty:
                        total_users    = len(u_df_viz)
                        unique_divs    = u_df_viz["Division"].nunique()
                        unique_posts   = u_df_viz["Post"].nunique()
                        unique_ranges  = u_df_viz["Range"].nunique() if "Range" in u_df_viz.columns else 0

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("👥 Total Staff",       total_users)
                        m2.metric("🌲 Active Divisions",  unique_divs)
                        m3.metric("📍 Active Ranges",     unique_ranges)
                        m4.metric("🏷️ Designations",     unique_posts)

                        st.subheader("📊 Staff Distribution")
                        col1, col2 = st.columns(2)

                        with col1:
                            fig_users_div = px.histogram(
                                u_df_viz, x="Division", color="Post",
                                title="Staff Count by Division & Designation",
                                text_auto=True, barmode="stack",
                            )
                            st.plotly_chart(fig_users_div, use_container_width=True)

                        with col2:
                            post_counts = u_df_viz["Post"].value_counts().reset_index()
                            post_counts.columns = ["Post", "Count"]
                            fig_users_pie = px.pie(
                                post_counts, values="Count", names="Post",
                                hole=0.4, title="Overall Designation Split",
                            )
                            st.plotly_chart(fig_users_pie, use_container_width=True)

                        if "Range" in u_df_viz.columns:
                            st.subheader("🔍 Hierarchy Drill-Down")
                            tree_df = (
                                u_df_viz.fillna("Unknown")
                                .groupby(["Division", "Range", "Post"])
                                .size()
                                .reset_index(name="Count")
                            )
                            fig_tree = px.sunburst(
                                tree_df, path=["Division", "Range", "Post"],
                                values="Count", height=700,
                                title="Staff Deployment Hierarchy",
                            )
                            st.plotly_chart(fig_tree, use_container_width=True)
                    else:
                        st.warning("No staff data to display after filtering.")

                    with st.expander("📄 View Full Staff List"):
                        st.dataframe(u_df, use_container_width=True)

                else:
                    st.error("CSV must contain 'Division' and 'Post' columns.")
        else:
            st.info("👆 Upload a Staff List CSV in the sidebar to view Staff Analytics.")
