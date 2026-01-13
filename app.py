import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import zipfile
import os
import xml.etree.ElementTree as ET
import glob
import re
import plotly.express as px
import base64
from datetime import timedelta
import numpy as np

# ==========================================
# 1. CONFIGURATION & ASSETS
# ==========================================
st.set_page_config(page_title="Elephant Dashboard", layout="wide", page_icon="🐘")

# Logos (Hosted URLs for reliability)
LOGO_WCT = "https://www.wildlifeconservationtrust.org/wp-content/uploads/2016/09/wct-logo.png"
LOGO_MP = "https://upload.wikimedia.org/wikipedia/commons/2/23/Emblem_of_Madhya_Pradesh.svg"

# ==========================================
# 2. UTILITY FUNCTIONS (Spatial & Data)
# ==========================================

@st.cache_data
def load_village_data(filepath="centroids.csv"):
    """
    Loads village centroid data.
    Handles encoding errors (Excel often uses Latin-1).
    """
    if os.path.exists(filepath):
        try:
            # Try default UTF-8 first
            v_df = pd.read_csv(filepath)
        except UnicodeDecodeError:
            # Fallback for Excel-saved CSVs
            v_df = pd.read_csv(filepath, encoding='ISO-8859-1')
        except Exception:
            return None

        # Clean column names
        v_df.columns = [c.strip() for c in v_df.columns]
        
        # Normalize column names to standard
        rename_map = {
            'Lat': 'Latitude', 'lat': 'Latitude',
            'Lon': 'Longitude', 'Long': 'Longitude', 'lon': 'Longitude',
            'Name': 'Village', 'village': 'Village', 'VILLAGE': 'Village'
        }
        v_df = v_df.rename(columns=rename_map)
        
        # Validation
        if {'Latitude', 'Longitude', 'Village'}.issubset(v_df.columns):
            return v_df
            
    return None

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Vectorized Haversine formula to calculate distance between two points on Earth.
    Returns distance in kilometers.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def identify_risk_villages(sightings_df, villages_df, damage_rad_km, presence_rad_km, cons_days):
    """
    Identifies affected villages based on user-defined parameters.
    """
    if villages_df is None or sightings_df.empty:
        return []

    # Prepare Coordinates
    v_lons = villages_df['Longitude'].values
    v_lats = villages_df['Latitude'].values
    v_names = villages_df['Village'].values
    
    s_lons = sightings_df['Longitude'].values
    s_lats = sightings_df['Latitude'].values
    
    # Calculate Full Distance Matrix (Villages x Sightings)
    dists = haversine_np(v_lons[:, np.newaxis], v_lats[:, np.newaxis], s_lons[np.newaxis, :], s_lats[np.newaxis, :])
    
    affected_list = []
    
    # --- CRITERIA 1: DAMAGE (User-defined Radius) ---
    damage_mask = (sightings_df['Crop Damage'] > 0) | (sightings_df['House Damage'] > 0)
    if damage_mask.any():
        dmg_indices = np.where(damage_mask)[0]
        min_dmg_dists = np.min(dists[:, dmg_indices], axis=1)
        
        # Use slider value
        dmg_v_indices = np.where(min_dmg_dists <= damage_rad_km)[0]
        for idx in dmg_v_indices:
            affected_list.append({
                'Village': v_names[idx],
                'Lat': v_lats[idx],
                'Lon': v_lons[idx],
                'Reason': f'Damage Reported (<{damage_rad_km}km)',
                'Color': 'red',
                'Radius': damage_rad_km * 1000  # Convert to meters for Map
            })

    # --- CRITERIA 2: PRESENCE (User-defined Radius & Days) ---
    max_date = sightings_df['Date'].max()
    
    # Check consecutive days based on slider input
    # If slider is 3, we check: max_date, max_date-1, max_date-2
    dates = sightings_df['Date'].values
    has_streak_per_village = np.ones(len(villages_df), dtype=bool) # Start True
    
    for i in range(cons_days):
        check_date = max_date - timedelta(days=i)
        idx_date = np.where(dates == np.datetime64(check_date))[0]
        
        if len(idx_date) == 0:
            has_streak_per_village[:] = False # No data for this day, streak impossible
            break
            
        # Check if village is within Presence Radius for this day
        # min(dists) <= presence_rad_km
        daily_presence = np.any(dists[:, idx_date] <= presence_rad_km, axis=1)
        has_streak_per_village &= daily_presence
        
    # Add villages that maintained the streak
    for idx in np.where(has_streak_per_village)[0]:
        # Avoid duplicates if already caught by Damage
        if not any(d['Village'] == v_names[idx] for d in affected_list):
            affected_list.append({
                'Village': v_names[idx],
                'Lat': v_lats[idx],
                'Lon': v_lons[idx],
                'Reason': f'{cons_days}-Day Presence (<{presence_rad_km}km)',
                'Color': 'orange',
                'Radius': presence_rad_km * 1000 # Convert to meters for Map
            })
                
    return affected_list

def calculate_affected_villages(sightings_df, villages_df, radius_km=2.0):
    """
    Finds all villages within a specified radius (default 2km) for every sighting.
    Returns the dataframe with a new column 'Affected Villages'.
    """
    if villages_df is None or sightings_df.empty:
        return sightings_df
    
    # Prepare arrays for fast computation
    v_lons = villages_df['Longitude'].values
    v_lats = villages_df['Latitude'].values
    v_names = villages_df['Village'].values
    
    def get_villages_in_radius(row):
        # Calculate distance from this sighting to ALL villages
        dists = haversine_np(v_lons, v_lats, row['Longitude'], row['Latitude'])
        
        # Filter for villages within the radius
        mask = dists <= radius_km
        nearby_villages = v_names[mask]
        
        # Return comma-separated string of villages, or "None" if none found
        if len(nearby_villages) > 0:
            return ", ".join(sorted(nearby_villages))
        else:
            return "None"

    # Apply to all rows
    sightings_df['Affected Villages'] = sightings_df.apply(get_villages_in_radius, axis=1)
    return sightings_df
# ==========================================
# 3. KML/GEOJSON PARSING (Map Layers)
# ==========================================

def parse_kml_coordinates(placemark, ns):
    """Extracts polygon coordinates from KML Placemark."""
    polygons = []
    for polygon in placemark.findall('.//kml:Polygon', ns):
        outer_boundary = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
        if outer_boundary is not None and outer_boundary.text:
            coord_str = outer_boundary.text.strip()
            poly_points = []
            for point in coord_str.split():
                try:
                    p = point.split(',')
                    # KML is Lon,Lat. GeoJSON (in Folium) expects [Lat, Lon] for polygons usually, 
                    # but GeoJSON standard is [Lon, Lat]. We use [Lat, Lon] for Folium drawing logic later if needed,
                    # but here we follow standard KML extraction.
                    lon, lat = float(p[0]), float(p[1])
                    poly_points.append([lon, lat]) 
                except (ValueError, IndexError):
                    continue
            if poly_points:
                polygons.append(poly_points)
    return polygons

def get_feature_name(placemark, ns):
    """Extracts Name from KML tags."""
    # 1. Try Simple Name
    name_tag = placemark.find('kml:name', ns)
    if name_tag is not None and name_tag.text:
        val = name_tag.text.strip()
        # Filter out generic export names
        if val.upper() not in ["BTR", "SATNA", "0", "1", "NONE"]:
            return val.title()
    
    # 2. Try Extended Data (often shapefile attributes end up here)
    ext = placemark.find('kml:ExtendedData', ns)
    if ext is not None:
        for sd in ext.findall('.//kml:SimpleData', ns):
            if sd.get('name', '').lower() in ['range_nm', 'beat_nm', 'name']:
                return sd.text.title()
                
    return "Unknown Area"

@st.cache_data
def load_map_data_as_geojson(folder_path="."):
    """
    Scans folder for KML/KMZ files and compiles them into a GeoJSON FeatureCollection.
    Returns: (GeoJSON Dict, List of Beat Names)
    """
    features = []
    all_beats = set()
    
    files = glob.glob(os.path.join(folder_path, "*.kmz")) + glob.glob(os.path.join(folder_path, "*.kml"))
    
    for file_path in files:
        try:
            content = None
            filename = os.path.basename(file_path)
            
            # Handle KMZ (Zipped KML)
            if file_path.endswith('.kmz'):
                with zipfile.ZipFile(file_path, 'r') as z:
                    kml_inside = [f for f in z.namelist() if f.endswith('.kml')]
                    if kml_inside:
                        with z.open(kml_inside[0]) as f:
                            content = f.read()
            else:
                with open(file_path, 'rb') as f:
                    content = f.read()

            if content:
                tree = ET.fromstring(content)
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                
                for placemark in tree.findall('.//kml:Placemark', ns):
                    name = get_feature_name(placemark, ns)
                    
                    if name != "Unknown Area":
                        all_beats.add(name)

                    polys = parse_kml_coordinates(placemark, ns)
                    
                    for poly_coords in polys:
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "name": name,
                                "source": filename
                            },
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [poly_coords] 
                            }
                        }
                        features.append(feature)
        except Exception as e:
            # Silent fail to avoid disrupting UI, but normally log this
            continue

    return features

# ==========================================
# 4. REPORT GENERATION ENGINE
# ==========================================

def generate_full_html_report(df, map_object, fig_trend, fig_demog, fig_hourly, fig_damage, fig_sunburst, start_date, end_date):
    """Generates a standalone HTML report with embedded charts and map."""
    
    # 1. Statistics
    total_sightings = len(df)
    cumulative_count = int(df['Total Count'].sum())
    conflicts = df[(df['Crop Damage']>0) | (df['House Damage']>0) | (df['Injury']>0)]
    conflict_count = len(conflicts)
    severity_score = int(df['Severity Score'].sum())
    
    # 2. Text Content
    narrative = f"""
    <h2 style="color: #2c3e50;">🐘 Elephant Monitoring & Conflict Report</h2>
    <p><b>Report Period:</b> {start_date} to {end_date}</p>
    
    <div style="background-color: #ecf0f1; padding: 20px; border-radius: 8px; border-left: 5px solid #2980b9;">
        <h4>📋 Executive Summary</h4>
        <p>During this period, a total of <b>{total_sightings}</b> field entries were recorded, indicating a cumulative presence of <b>{cumulative_count}</b> elephants (based on direct sightings and indirect signs).</p>
        <p><b>Risk Assessment:</b> The calculated Conflict Severity Score is <b>{severity_score}</b>, derived from <b>{conflict_count}</b> specific conflict incidents (Crop/House/Injury).</p>
    </div>
    <br>
    """
    
    methodology = """
    <div style="margin-top: 20px; border: 1px solid #ddd; padding: 15px; background-color: #fafafa; font-size: 0.9em;">
        <h5>ℹ️ Methodology & Definitions</h5>
        <ul>
            <li><b>Conflict Severity Score:</b> Weighted Index. <i>Score = (Presence × 1) + (Crop Damage × 2.5) + (House Damage × 5) + (Human Injury × 20)</i>.</li>
            <li><b>HEC Ratio:</b> Percentage of entries that resulted in conflict events.</li>
            <li><b>Nocturnal Activity:</b> Sightings recorded between 18:00 (6 PM) and 06:00 (6 AM).</li>
        </ul>
    </div>
    <br>
    """
    
    # 3. Component Conversion (to HTML)
    map_html = map_object.get_root().render()
    # Safely escape quotes for srcdoc
    map_iframe = f"""<iframe srcdoc="{map_html.replace('"', '&quot;')}" width="100%" height="500px" style="border:1px solid #ddd; border-radius:4px;"></iframe>"""
    
    chart_trend = fig_trend.to_html(full_html=False, include_plotlyjs='cdn')
    chart_demog = fig_demog.to_html(full_html=False, include_plotlyjs='cdn')
    chart_damage = fig_damage.to_html(full_html=False, include_plotlyjs='cdn') if fig_damage else "<p>No conflict data.</p>"
    chart_sunburst = fig_sunburst.to_html(full_html=False, include_plotlyjs='cdn') if fig_sunburst else ""
    chart_hourly = fig_hourly.to_html(full_html=False, include_plotlyjs='cdn') if fig_hourly else "<p>No time data.</p>"
    
    # Data Table
    dmg_table = df.groupby(['Division', 'Range']).agg({'Crop Damage': lambda x: (x>0).sum(), 'House Damage': lambda x: (x>0).sum(), 'Injury': lambda x: (x>0).sum()}).reset_index()
    dmg_table = dmg_table[(dmg_table['Crop Damage']>0)|(dmg_table['House Damage']>0)|(dmg_table['Injury']>0)]
    table_html = dmg_table.to_html(classes='table table-striped table-bordered', index=False)

    # 4. Final Assembly
    full_html = f"""
    <html>
    <head>
        <title>Elephant Report</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>
            body{{padding: 40px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}}
            h3{{margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 10px; color: #34495e;}}
            .footer{{margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 12px; border-top: 1px solid #eee; padding-top: 20px;}}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <img src="{LOGO_MP}" width="80" alt="MP Forest">
                <img src="{LOGO_WCT}" width="150" alt="WCT Mumbai">
            </div>
            
            {narrative}
            {methodology}
            
            <h3>📍 Spatial Distribution Map</h3>
            {map_iframe}
            
            <h3>📊 Population & Temporal Analysis</h3>
            <div class="row">
                <div class="col-md-6">{chart_trend}</div>
                <div class="col-md-6">{chart_demog}</div>
            </div>
            
            <h3>🔥 Conflict & Activity Analysis</h3>
            <div class="row">
                <div class="col-md-6">{chart_damage}</div>
                <div class="col-md-6">{chart_hourly}</div>
            </div>
            
            <div class="row">
                <div class="col-md-12">{chart_sunburst}</div>
            </div>
            
            <h3>⚠️ Detailed Damage Data</h3>
            {table_html}
            
            <div class="footer">
                <p>Generated by Elephant Sighting & Conflict Dashboard</p>
                <p>Intellectual Property of Wildlife Conservation Trust, Mumbai</p>
            </div>
        </div>
    </body>
    </html>
    """
    return full_html

# ==========================================
# 5. SIDEBAR & SESSION STATE
# ==========================================

# Initialize Session State Variables
if 'map_filter' not in st.session_state:
    st.session_state.map_filter = 'All'
if 'hotspot_beat' not in st.session_state:
    st.session_state.hotspot_beat = None

# Sidebar Header
with st.sidebar:
    c1, c2 = st.columns([1, 1])
    c1.markdown(f'<img src="{LOGO_WCT}" width="100%">', unsafe_allow_html=True)
    c2.markdown(f'<img src="{LOGO_MP}" width="80%">', unsafe_allow_html=True)
    st.markdown("---")
    st.info("© **Wildlife Conservation Trust, Mumbai**\n\nDeveloped for MP Forest Department Elephant Monitoring.")

    # --- RISK PARAMETERS SLIDERS ---
    st.divider()
    st.subheader("🏘️ Risk Parameters")
    with st.expander("Configure Logic", expanded=True):
        p_dmg_rad = st.slider("Damage Radius (km)", 0.5, 5.0, 2.0, 0.5, key="slider_dmg", help="Alert village if damage occurs within this distance.")
        p_pres_rad = st.slider("Presence Radius (km)", 1.0, 10.0, 5.0, 0.5, key="slider_pres", help="Alert village if elephant present within this distance.")
        p_days = st.slider("Consecutive Days", 1, 7, 3, key="slider_days", help="Number of consecutive days required for presence alert.")

    # --- NEW: USER REGISTRY (Inside Sidebar) ---
    st.divider()
    st.header("👥 User Registry")
    uploaded_users = st.file_uploader("Upload Staff List (CSV)", type=['csv'], key="user_upload")

# Main Title (Outside Sidebar)
st.title("🐘 Elephant Sighting & Conflict Command Centre")

# ==========================================
# 6. DATA LOADING
# ==========================================

# Load GeoJSON (Map Boundaries)
geojson_features = load_map_data_as_geojson(".")

# Load Village Data
village_df = load_village_data("centroids.csv")

# Display Loading Status
if geojson_features:
    with st.expander(f"✅ Map Data Loaded ({len(geojson_features)} regions)"):
        st.write("System optimized for fast loading.")

if village_df is not None:
    st.sidebar.success(f"✅ Village Data Loaded ({len(village_df)} villages)")
else:
    st.sidebar.warning("⚠️ 'centroids.csv' not found. Village analysis disabled.")

# ==========================================
# 7. MAIN LOGIC (File Upload -> Filter -> Viz)
# ==========================================

uploaded_csv = st.file_uploader("Upload Sightings Report (CSV)", type=['csv'])

if uploaded_csv is not None:
    # --- A. DATA PROCESSING ---
    try:
        df_raw = pd.read_csv(uploaded_csv)
    except UnicodeDecodeError:
        uploaded_csv.seek(0)
        df_raw = pd.read_csv(uploaded_csv, encoding='ISO-8859-1')
        
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y', errors='coerce')
    
    # Process Hour
    try:
        df_raw['Hour'] = pd.to_datetime(df_raw['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    except:
        df_raw['Hour'] = 0 
        
    df_raw = df_raw.dropna(subset=['Date'])
    
    # Numeric Cleanup
    cols = ['Total Count', 'Male Count', 'Female Count', 'Calf Count', 'Crop Damage', 'House Damage', 'Injury']
    for c in cols:
        if c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').fillna(0)

    # Text Normalization
    for c in ['Range', 'Beat', 'Division']:
        if c in df_raw.columns:
            df_raw[c] = df_raw[c].astype(str).str.title().fillna('Unknown')
    
    # Fix Hierarchy for Sunburst
    df_raw['Division'] = df_raw['Division'].fillna('Unknown')
    df_raw['Range'] = df_raw['Range'].fillna('Unknown')
    df_raw['Beat'] = df_raw['Beat'].fillna('Unknown')

    # --- B. ADVANCED CALCULATIONS ---
    # Severity Score: (Presence=1, Crop=2.5, House=5, Injury=20)
    df_raw['Severity Score'] = (
        1 + 
        (df_raw['Crop Damage'] > 0).astype(int) * 2.5 +
        (df_raw['House Damage'] > 0).astype(int) * 5 +
        (df_raw['Injury'] > 0).astype(int) * 20
    )
    
    # Day/Night
    df_raw['Is_Night'] = df_raw['Hour'].apply(lambda x: 1 if (x >= 18 or x <= 6) else 0)
    
    # Village Distance (Updated for 2km Radius)
    if village_df is not None:
        df_raw = calculate_affected_villages(df_raw, village_df, radius_km=2.0)
        # Mark as 'Near Village' if the list is not "None"
        df_raw['Near Village'] = df_raw['Affected Villages'] != "None"

    # --- C. FILTERS (Global Scope) ---
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = df_raw['Date'].min(), df_raw['Date'].max()
        start, end = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        # Apply Time Filter
        df = df_raw[(df_raw['Date'].dt.date >= start) & (df_raw['Date'].dt.date <= end)]
        
        # Calculate Trend Previous Period
        delta_days = (end - start).days + 1
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=delta_days - 1)
        df_prev = df_raw[(df_raw['Date'].dt.date >= prev_start) & (df_raw['Date'].dt.date <= prev_end)]

        # Attribute Filters
        divisions = ['All'] + sorted(list(df['Division'].unique()))
        sel_div = st.selectbox("Filter Division", divisions)
        if sel_div != 'All': 
            df = df[df['Division'] == sel_div]
            df_prev = df_prev[df_prev['Division'] == sel_div]
        
        ranges_in_data = sorted(list(df['Range'].unique()))
        ranges = ['All'] + ranges_in_data
        sel_range = st.selectbox("Filter Range", ranges)
        if sel_range != 'All': 
            df = df[df['Range'] == sel_range]
            df_prev = df_prev[df_prev['Range'] == sel_range]
            
        st.divider()
        st.header("Map Settings")
        map_mode = st.radio("Visualization Mode:", ["Pins", "Heatmap"], horizontal=True)

    # --- D. OPERATIONAL METRICS (Row 1) ---
    n_sightings = len(df)
    n_cumulative = int(df['Total Count'].sum())
    n_direct = len(df[df['Sighting Type'] == 'Direct'])
    n_conflicts = ((df['Crop Damage']>0) | (df['House Damage']>0) | (df['Injury']>0)).sum()
    n_males = int(df['Male Count'].sum())
    n_calves = int(df['Calf Count'].sum())

    st.markdown("### 📈 Operational Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    # UPDATED: Replaced use_container_width=True with width="stretch"
    if c1.button(f"📝 Entries\n\n# {n_sightings}", width="stretch"): st.session_state.map_filter = 'All'
    if c2.button(f"🐾 Cumulative\n\n# {n_cumulative}", width="stretch"): st.session_state.map_filter = 'All'
    if c3.button(f"👁️ Direct\n\n# {n_direct}", width="stretch"): st.session_state.map_filter = 'Direct'
    if c4.button(f"🔥 Conflicts\n\n# {n_conflicts}", width="stretch"): st.session_state.map_filter = 'Conflict'
    if c5.button(f"♂️ Males\n\n# {n_males}", width="stretch"): st.session_state.map_filter = 'Males'
    if c6.button(f"👶 Calves\n\n# {n_calves}", width="stretch"): st.session_state.map_filter = 'Calves'
   
   # --- E. STRATEGIC INDICATORS (Row 2 - Reactive) ---
    st.markdown("### 🛡️ Strategic Indicators (Click to Visualize)")
    k1, k2, k3, k4 = st.columns(4)
    
    # 1. Severity
    curr_sev = df['Severity Score'].sum()
    prev_sev = df_prev['Severity Score'].sum()
    delta_sev = curr_sev - prev_sev
    if k1.button(f"🚨 Severity Score\n\n{curr_sev:.1f} ({'+' if delta_sev>0 else ''}{delta_sev:.1f})", width="stretch"):
        st.session_state.map_filter = 'Severity_View'
    
    # 2. HEC Ratio
    curr_hec = (n_conflicts / n_sightings * 100) if n_sightings > 0 else 0
    prev_conf = ((df_prev['Crop Damage']>0)|(df_prev['House Damage']>0)|(df_prev['Injury']>0)).sum()
    prev_sight = len(df_prev)
    prev_hec = (prev_conf / prev_sight * 100) if prev_sight > 0 else 0
    delta_hec = curr_hec - prev_hec
    if k2.button(f"⚠️ HEC Ratio\n\n{curr_hec:.1f}% ({'+' if delta_hec>0 else ''}{delta_hec:.1f}%)", width="stretch"):
        st.session_state.map_filter = 'Conflict'
        
    # 3. Night Activity
    curr_night = (df['Is_Night'].sum() / n_sightings * 100) if n_sightings > 0 else 0
    prev_night = (df_prev['Is_Night'].sum() / prev_sight * 100) if prev_sight > 0 else 0
    delta_night = curr_night - prev_night
    if k3.button(f"🌙 Night Activity\n\n{curr_night:.1f}% ({'+' if delta_night>0 else ''}{delta_night:.1f}%)", width="stretch"):
        st.session_state.map_filter = 'Night_View'
        
    # 4. Hotspot
    if not df.empty:
        beat_stats = df.groupby('Beat')['Severity Score'].sum().reset_index().sort_values('Severity Score', ascending=False)
        if not beat_stats.empty and beat_stats.iloc[0]['Severity Score'] > 0:
            top_beat = beat_stats.iloc[0]['Beat']
            st.session_state.hotspot_beat = top_beat
        else:
            top_beat = "None"
            st.session_state.hotspot_beat = None
    else:
        top_beat = "None"
    
    if k4.button(f"🔁 Hotspot: {top_beat}\n\n(Click to Focus)", width="stretch"):
        st.session_state.map_filter = 'Hotspot_View'
   
    # Explanation Panel
    explanation = ""
    if st.session_state.map_filter == 'Severity_View': explanation = "🔴 **Viewing Severity:** Markers sized by Severity Score."
    elif st.session_state.map_filter == 'Night_View': explanation = "🟣 **Viewing Night Activity:** Sightings between 6PM and 6AM."
    elif st.session_state.map_filter == 'Hotspot_View': explanation = f"🎯 **Viewing Hotspot:** Focusing on **{top_beat}** (Highest Severity)."
    elif st.session_state.map_filter == 'Conflict': explanation = "🔥 **Viewing Conflicts:** Crop, House, or Injury incidents."
    else: explanation = "📍 **Viewing All Data:** Showing all records."
    st.info(explanation)

# --- F. MAP VISUALIZATION (Reactive, Hybrid & Interactive) ---
    
    # 1. Initialize Selection State
    if 'selected_village' not in st.session_state:
        st.session_state.selected_village = None

    # 2. PREPARE BASE MAP DATA
    map_df = df.copy()
    
    if st.session_state.map_filter == 'Conflict':
        map_df = map_df[(map_df['Crop Damage']>0)|(map_df['House Damage']>0)|(map_df['Injury']>0)]
    elif st.session_state.map_filter == 'Night_View':
        map_df = map_df[map_df['Is_Night'] == 1]
    elif st.session_state.map_filter == 'Hotspot_View' and st.session_state.hotspot_beat:
        map_df = map_df[map_df['Beat'] == st.session_state.hotspot_beat]
    elif st.session_state.map_filter == 'Direct':
        map_df = map_df[map_df['Sighting Type'] == 'Direct']
    elif st.session_state.map_filter == 'Males':
        map_df = map_df[map_df['Male Count'] > 0]
    elif st.session_state.map_filter == 'Calves':
        map_df = map_df[map_df['Calf Count'] > 0]

    # 3. Calculate Risk Villages
    risk_villages = []
    if village_df is not None:
        risk_villages = identify_risk_villages(map_df, village_df, p_dmg_rad, p_pres_rad, p_days)

    # 4. Handle Selection & Zoom Logic
    displayed_map_df = map_df.copy()
    
    # Default View
    map_center = [23.5, 80.5]
    map_zoom = 9
    
    if st.session_state.selected_village and village_df is not None:
        sel_v = next((v for v in risk_villages if v['Village'] == st.session_state.selected_village), None)
        if sel_v:
            map_center = [sel_v['Lat'], sel_v['Lon']]
            map_zoom = 13
            search_rad = max(p_dmg_rad, p_pres_rad)
            v_lons = sel_v['Lon']
            v_lats = sel_v['Lat']
            s_lons = displayed_map_df['Longitude'].values
            s_lats = displayed_map_df['Latitude'].values
            if len(s_lons) > 0:
                dists = haversine_np(v_lons, v_lats, s_lons, s_lats)
                displayed_map_df = displayed_map_df[dists <= search_rad]

    # 5. Create Layout
    c_map, c_list = st.columns([3, 1])
    
    # --- RIGHT COLUMN: LIST ---
    with c_list:
        st.subheader("🚨 Affected Villages")
        if st.button("🌍 Show Full Map", type="secondary", use_container_width=True):
            st.session_state.selected_village = None
            st.rerun()
        st.markdown("---")
        
        if not risk_villages:
            st.info("No villages at risk.")
        else:
            for i, v in enumerate(risk_villages):
                is_active = (v['Village'] == st.session_state.selected_village)
                btn_type = "primary" if is_active else "secondary"
                label = f"{'📍' if is_active else ''} {v['Village']}"
                # Unique key prevents duplicate error
                if st.button(f"{label}\n\n({v['Reason']})", key=f"btn_{v['Village']}_{i}", type=btn_type, use_container_width=True):
                    st.session_state.selected_village = v['Village']
                    st.rerun()

    # --- LEFT COLUMN: MAP ---
    with c_map:
        # 1. Initialize Map
        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")

        # 2. Add Google Hybrid (Optional Toggle)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Hybrid',
            overlay=False,
            control=True
        ).add_to(m)

        # 3. GeoJSON Boundaries
        if geojson_features:
            folium.GeoJson(
                data={"type": "FeatureCollection", "features": geojson_features},
                style_function=lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.1},
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Region:'])
            ).add_to(m)

        # 4. Markers
        if map_mode == "Heatmap":
            heat_data = [[r['Latitude'], r['Longitude'], max(r['Severity Score'], 1)] for _, r in displayed_map_df.iterrows()]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
        else:
            for _, row in displayed_map_df.iterrows():
                color = 'green'
                radius = 4
                if st.session_state.map_filter == 'Severity_View':
                    s = row['Severity Score']
                    if s >= 20: color='darkred'; radius=15
                    elif s >= 5: color='red'; radius=10
                    elif s >= 2.5: color='orange'; radius=7
                    else: color='blue'; radius=4
                elif st.session_state.map_filter == 'Night_View':
                    color='purple'; radius=6
                else:
                    if row['Injury']>0: color='darkred'
                    elif row['House Damage']>0: color='red'
                    elif row['Crop Damage']>0: color='orange'
                    elif row['Sighting Type'] == 'Direct': color='blue'
                
                dmg_str = []
                if row['Crop Damage'] > 0: dmg_str.append('Crop')
                if row['House Damage'] > 0: dmg_str.append('House')
                if row['Injury'] > 0: dmg_str.append('Injury')
                dmg_txt = ", ".join(dmg_str) if dmg_str else "None"

                tooltip = f"""
                <div style='font-family:sans-serif; font-size:12px;'>
                    <b>📅 {row['Date'].date()}</b> | 🕒 {row['Time'] if pd.notnull(row['Time']) else '???'}<br>
                    <b>📍 Location:</b> {row['Beat']}<br>
                    <b>🐘 Herd:</b> M:{int(row['Male Count'])} | F:{int(row['Female Count'])} | C:{int(row['Calf Count'])}<br>
                    <b>⚠️ Damage:</b> {dmg_txt}
                </div>
                """
                
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    tooltip=tooltip
                ).add_to(m)

        # 5. Risk Rings
        if village_df is not None:
            villages_to_show = [v for v in risk_villages if v['Village'] == st.session_state.selected_village] if st.session_state.selected_village else risk_villages
            for v in villages_to_show:
                folium.Circle(
                    location=[v['Lat'], v['Lon']],
                    radius=v['Radius'],
                    color=v['Color'],
                    weight=3 if st.session_state.selected_village else 2,
                    fill=True, fill_opacity=0.15,
                    tooltip=f"<b>{v['Village']}</b><br>{v['Reason']}"
                ).add_to(m)
                folium.Marker(
                    location=[v['Lat'], v['Lon']],
                    icon=folium.Icon(color="red" if v['Color']=='red' else "orange", icon="home", prefix="fa"),
                    tooltip=f"<b>{v['Village']}</b>"
                ).add_to(m)
        
        # 6. Add Layer Control
        folium.LayerControl().add_to(m)

        # 7. RENDER COMMAND (Crucial: Must be indented inside 'with c_map:', NOT inside 'if')
        st_data = st_folium(m, width=None, height=600, use_container_width=True, returned_objects=[])
   # --- G. ANALYTICS CHARTS ---
    st.divider()
    r1c1, r1c2 = st.columns(2)
    
    with r1c1:
        st.subheader("Hierarchy Drill-Down")
        if not df.empty:
            sb_df = df.copy()
            sb_val = 'Total Count'
            sb_title = "Sighting Distribution"
            
            # Dynamic Context for Sunburst
            if st.session_state.map_filter == 'Conflict': 
                sb_df = sb_df[(sb_df['Crop Damage']>0)|(sb_df['House Damage']>0)|(sb_df['Injury']>0)]
                sb_df['Incidents'] = 1; sb_val = 'Incidents'
                sb_title = "Conflict Hierarchy"
            elif st.session_state.map_filter == 'Direct': 
                sb_df = sb_df[sb_df['Sighting Type'] == 'Direct']
                sb_title = "Direct Sightings"
            elif st.session_state.map_filter == 'Males': 
                sb_df = sb_df[sb_df['Male Count'] > 0]
                sb_val = 'Male Count'
                sb_title = "Male Population"
                
            if not sb_df.empty:
                fig_sun = px.sunburst(sb_df, path=['Division', 'Range', 'Beat'], values=sb_val, title=sb_title)
                # UPDATED: Replaced use_container_width with width="stretch"
                st.plotly_chart(fig_sun, width="stretch")
            else:
                fig_sun = None
                st.info("No data available for this view.")
        else:
            st.info("No data available.")

    with r1c2:
        st.subheader("Hourly Activity (0-24h)")
        if not df.empty:
            h_counts = df['Hour'].value_counts().reindex(range(24), fill_value=0).reset_index()
            fig_hourly = px.bar(h_counts, x='Hour', y='count', title="Activity Peaks", 
                                color='count', color_continuous_scale='Viridis')
            # UPDATED: Replaced use_container_width with width="stretch"
            st.plotly_chart(fig_hourly, width="stretch")
        else: fig_hourly = None

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trend Analysis")
        daily = df.groupby('Date').size().reset_index(name='Count')
        fig_trend = px.line(daily, x='Date', y='Count', markers=True)
        # UPDATED: Replaced use_container_width with width="stretch"
        st.plotly_chart(fig_trend, width="stretch")
    with c2:
        st.subheader("Demographics")
        demog = df[['Male Count', 'Female Count', 'Calf Count']].sum().reset_index()
        demog.columns = ['Type', 'Count']
        fig_demog = px.pie(demog, values='Count', names='Type', hole=0.4)
        # UPDATED: Replaced use_container_width with width="stretch"
        st.plotly_chart(fig_demog, width="stretch")

    # Conflict Breakdown
    c4 = st.columns(1)[0]
    with c4:
        st.subheader("Conflict Type Breakdown")
        damage_sums = df[['Crop Damage', 'House Damage', 'Injury']].apply(lambda x: (x > 0).sum()).reset_index()
        damage_sums.columns = ['Damage Type', 'Incidents']
        damage_sums = damage_sums[damage_sums['Incidents'] > 0]
        if not damage_sums.empty:
            fig_damage = px.pie(damage_sums, values='Incidents', names='Damage Type', 
                                color='Damage Type', 
                                color_discrete_map={'Crop Damage':'orange', 'House Damage':'red', 'Injury':'darkred'})
            # UPDATED: Replaced use_container_width with width="stretch"
            st.plotly_chart(fig_damage, width="stretch")
        else:
            fig_damage = None
            st.info("No conflicts reported.")

   # [Insert this block after the Range Growth Animation in Section G]

    st.divider()
    st.subheader("🏆 Division Comparison: Cumulative Reports")
    
    if not df.empty:
        # 1. Data Preparation
        # Group by Date and Division to get daily counts (Rows = Reports)
        div_daily = df.groupby(['Date', 'Division']).size().reset_index(name='Daily')
        
        # 2. Pivot for Continuity
        # Create a matrix (Date x Division) to ensure we handle days with 0 reports
        div_pivot = div_daily.pivot(index='Date', columns='Division', values='Daily').fillna(0)
        
        # 3. Reindex Date Range
        # Fill in missing dates to make the animation smooth
        all_dates = pd.date_range(start=div_pivot.index.min(), end=div_pivot.index.max())
        div_pivot = div_pivot.reindex(all_dates, fill_value=0)
        
        # 4. Calculate Cumulative Sum
        div_cum = div_pivot.cumsum()
        
        # 5. Melt back to Long Format for Plotly
        div_long = div_cum.stack().reset_index()
        div_long.columns = ['Date', 'Division', 'Cumulative Reports'] # <--- Updated Label
        div_long['Date_Str'] = div_long['Date'].dt.strftime('%Y-%m-%d')
        
        # 6. Generate Animated Bar Chart
        fig_div_anim = px.bar(
            div_long, 
            x="Division", 
            y="Cumulative Reports", 
            color="Division", 
            animation_frame="Date_Str", 
            range_y=[0, div_long['Cumulative Reports'].max() * 1.1], # Lock Y-axis
            title="Cumulative Reports by Division", # <--- Updated Title
            hover_data=['Cumulative Reports']
        )
        
        # 7. Animation Settings
        fig_div_anim.update_layout(
            xaxis_title="Forest Division",
            yaxis_title="Total Reports (Cumulative)", # <--- Updated Axis Title
            showlegend=False
        )
        # Set animation speed
        fig_div_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
        
        st.plotly_chart(fig_div_anim, width="stretch")
    # [Insert this block after the Conflict Breakdown chart in Section G]

    st.divider()
    st.subheader("⏳ Temporal Dynamics: Range Growth Animation")
    
    if not df.empty:
        # 1. Division Filter
        # Create list of divisions, ensuring 'All' is not included if strictly filtering
        div_options = sorted(list(df['Division'].unique()))
        selected_div_anim = st.selectbox("Select Division to Visualize:", div_options, key="anim_div_filter")
        
        # 2. Filter Data
        subset = df[df['Division'] == selected_div_anim].copy()
        
        if not subset.empty:
            # 3. Data Preparation for Smooth Animation
            # We need a continuous timeline so bars don't jump.
            
            # A. Group by Day and Range to get daily counts
            daily_counts = subset.groupby(['Date', 'Range']).size().reset_index(name='Daily')
            
            # B. Pivot to create a matrix (Date x Range)
            # This ensures we have a column for every Range and index for every Date
            pivoted = daily_counts.pivot(index='Date', columns='Range', values='Daily').fillna(0)
            
            # C. Reindex to fill in missing days (optional, makes animation smoother)
            full_date_range = pd.date_range(start=pivoted.index.min(), end=pivoted.index.max())
            pivoted = pivoted.reindex(full_date_range, fill_value=0)
            
            # D. Calculate Cumulative Sum
            cum_pivoted = pivoted.cumsum()
            
            # E. Melt back to "Long" format for Plotly
            cum_long = cum_pivoted.stack().reset_index()
            cum_long.columns = ['Date', 'Range', 'Cumulative Entries']
            
            # F. Create String Date for Animation Slider
            cum_long['Date_Str'] = cum_long['Date'].dt.strftime('%Y-%m-%d')
            
            # 4. Generate Animated Bar Chart
            fig_anim = px.bar(
                cum_long, 
                x="Range", 
                y="Cumulative Entries", 
                color="Range", 
                animation_frame="Date_Str", 
                range_y=[0, cum_long['Cumulative Entries'].max() * 1.1], # Lock Y-axis
                title=f"Cumulative Entry Growth by Range ({selected_div_anim})",
                hover_data=['Cumulative Entries']
            )
            
            # 5. Animation Settings
            fig_anim.update_layout(
                xaxis_title="Forest Range",
                yaxis_title="Total Entries (Cumulative)",
                showlegend=False
            )
            # Adjust speed (Frame duration in ms)
            fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 100
            
            st.plotly_chart(fig_anim, width="stretch")
        else:
            st.warning(f"No data available for Division: {selected_div_anim}")
    # --- H. DATA TABLES ---
    st.divider()
    t1, t2 = st.tabs(["⚠️ Damage Report", "🏆 Leaderboards"])
    with t1:
        dmg = df.groupby(['Division', 'Range']).agg({'Crop Damage': lambda x: (x>0).sum(), 'House Damage': lambda x: (x>0).sum(), 'Injury': lambda x: (x>0).sum()}).reset_index()
        dmg = dmg[(dmg['Crop Damage']>0)|(dmg['House Damage']>0)|(dmg['Injury']>0)]
        dmg.columns = ['Division', 'Range', '🌾 Crop', '🏠 House', '🚑 Injury']
        # UPDATED
        st.dataframe(dmg.style.background_gradient(cmap="Reds"), width="stretch")
    with t2:
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Top Beats (Activity)**")
            if 'Beat' in df.columns:
                # UPDATED
                st.dataframe(df['Beat'].value_counts().reset_index(name='Entries').head(10), width="stretch", hide_index=True)
        with l2:
            st.markdown("**Top Reporters**")
            if 'Created By' in df.columns:
                # UPDATED
                st.dataframe(df['Created By'].value_counts().reset_index(name='Entries').head(10), width="stretch", hide_index=True)

    # --- I. REPORT GENERATION ---
    st.divider()
    st.subheader("📄 Report Generation")
    
    if st.button("🖨️ Generate Full Report"):
        # Pass charts only if they exist
        sun_chart = fig_sun if 'fig_sun' in locals() and fig_sun else None
        dam_chart = fig_damage if 'fig_damage' in locals() and fig_damage else None
        
        html_report = generate_full_html_report(df, m, fig_trend, fig_demog, fig_hourly, dam_chart, sun_chart, start, end)
        st.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name="Elephant_Monitoring_Report.html",
            mime="text/html"
        )

    # ==========================================
# J. USER REGISTRY ANALYTICS (New Section)
# ==========================================

if uploaded_users is not None:
    st.markdown("---")
    st.title("👥 Gaj Rakshak Staff Analytics")
    
    # 1. Load & Clean Data
    try:
        u_df = pd.read_csv(uploaded_users)
    except UnicodeDecodeError:
        uploaded_users.seek(0)
        u_df = pd.read_csv(uploaded_users, encoding='ISO-8859-1')
        
    # Standardize Columns
    u_df.columns = [c.strip() for c in u_df.columns]
    
    if {'Division', 'Post'}.issubset(u_df.columns):
        # Clean Data
        u_df['Division'] = u_df['Division'].astype(str).str.title().str.strip()
        u_df['Post'] = u_df['Post'].astype(str).str.title().str.strip()
        if 'Range' in u_df.columns:
            u_df['Range'] = u_df['Range'].astype(str).str.title().str.strip()

        # 2. Metrics
        total_users = len(u_df)
        unique_divs = u_df['Division'].nunique()
        unique_posts = u_df['Post'].nunique()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Registered Staff", total_users)
        c2.metric("Active Divisions", unique_divs)
        c3.metric("Designation Categories", unique_posts)
        
        # 3. Visualizations
        st.subheader("📊 Staff Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Chart A: Users by Division (Colored by Post)
            st.markdown("**Staff Strength by Division**")
            fig_users_div = px.histogram(
                u_df, 
                x="Division", 
                color="Post", 
                title="Staff Count by Division & Designation",
                text_auto=True,
                barmode='stack'
            )
            fig_users_div.update_layout(yaxis_title="Count")
            st.plotly_chart(fig_users_div, width="stretch")
            
        with col2:
            # Chart B: Designation Breakdown (Pie)
            st.markdown("**Designation Hierarchy**")
            post_counts = u_df['Post'].value_counts().reset_index()
            post_counts.columns = ['Post', 'Count']
            fig_users_pie = px.pie(
                post_counts, 
                values='Count', 
                names='Post', 
                hole=0.4,
                title="Overall Designation Split"
            )
            st.plotly_chart(fig_users_pie, width="stretch")

        # 4. Sunburst Drill-Down (Division -> Range -> Post)
        if 'Range' in u_df.columns:
            st.subheader("🔍 Hierarchy Drill-Down")
            st.markdown("Visualizing **Division ➔ Range ➔ Post** distribution.")
            
            # Group for clean tree structure
            tree_df = u_df.fillna('Unknown').groupby(['Division', 'Range', 'Post']).size().reset_index(name='Count')
            
            fig_tree = px.sunburst(
                tree_df,
                path=['Division', 'Range', 'Post'],
                values='Count',
                height=700,
                title="Staff Deployment Hierarchy"
            )
            st.plotly_chart(fig_tree, width="stretch")
            
        # 5. Data Table
        with st.expander("📄 View Full Staff List"):
            st.dataframe(u_df, use_container_width=True)
            
    else:
        st.error("CSV must contain 'Division' and 'Post' columns.")

else:
    st.info("👆 Upload CSV to begin.")

























