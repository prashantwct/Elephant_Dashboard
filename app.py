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

# Main Title
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
    
    # Village Distance
    if village_df is not None:
        df_raw = calculate_nearest_village(df_raw, village_df)
        df_raw['Near Village'] = df_raw['Distance to Village (km)'] < 0.5 

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
    if c1.button(f"📝 Entries\n\n# {n_sightings}", use_container_width=True): st.session_state.map_filter = 'All'
    if c2.button(f"🐾 Cumulative\n\n# {n_cumulative}", use_container_width=True): st.session_state.map_filter = 'All'
    if c3.button(f"👁️ Direct\n\n# {n_direct}", use_container_width=True): st.session_state.map_filter = 'Direct'
    if c4.button(f"🔥 Conflicts\n\n# {n_conflicts}", use_container_width=True): st.session_state.map_filter = 'Conflict'
    if c5.button(f"♂️ Males\n\n# {n_males}", use_container_width=True): st.session_state.map_filter = 'Males'
    if c6.button(f"👶 Calves\n\n# {n_calves}", use_container_width=True): st.session_state.map_filter = 'Calves'

    # --- E. STRATEGIC INDICATORS (Row 2 - Reactive) ---
    st.markdown("### 🛡️ Strategic Indicators (Click to Visualize)")
    k1, k2, k3, k4 = st.columns(4)
    
    # 1. Severity
    curr_sev = df['Severity Score'].sum()
    prev_sev = df_prev['Severity Score'].sum()
    delta_sev = curr_sev - prev_sev
    if k1.button(f"🚨 Severity Score\n\n{curr_sev:.1f} ({'+' if delta_sev>0 else ''}{delta_sev:.1f})", use_container_width=True):
        st.session_state.map_filter = 'Severity_View'
    
    # 2. HEC Ratio
    curr_hec = (n_conflicts / n_sightings * 100) if n_sightings > 0 else 0
    prev_conf = ((df_prev['Crop Damage']>0)|(df_prev['House Damage']>0)|(df_prev['Injury']>0)).sum()
    prev_sight = len(df_prev)
    prev_hec = (prev_conf / prev_sight * 100) if prev_sight > 0 else 0
    delta_hec = curr_hec - prev_hec
    if k2.button(f"⚠️ HEC Ratio\n\n{curr_hec:.1f}% ({'+' if delta_hec>0 else ''}{delta_hec:.1f}%)", use_container_width=True):
        st.session_state.map_filter = 'Conflict'
        
    # 3. Night Activity
    curr_night = (df['Is_Night'].sum() / n_sightings * 100) if n_sightings > 0 else 0
    prev_night = (df_prev['Is_Night'].sum() / prev_sight * 100) if prev_sight > 0 else 0
    delta_night = curr_night - prev_night
    if k3.button(f"🌙 Night Activity\n\n{curr_night:.1f}% ({'+' if delta_night>0 else ''}{delta_night:.1f}%)", use_container_width=True):
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
    
    if k4.button(f"🔁 Hotspot: {top_beat}\n\n(Click to Focus)", use_container_width=True):
        st.session_state.map_filter = 'Hotspot_View'

    # Explanation Panel
    explanation = ""
    if st.session_state.map_filter == 'Severity_View': explanation = "🔴 **Viewing Severity:** Markers sized by Severity Score."
    elif st.session_state.map_filter == 'Night_View': explanation = "🟣 **Viewing Night Activity:** Sightings between 6PM and 6AM."
    elif st.session_state.map_filter == 'Hotspot_View': explanation = f"🎯 **Viewing Hotspot:** Focusing on **{top_beat}** (Highest Severity)."
    elif st.session_state.map_filter == 'Conflict': explanation = "🔥 **Viewing Conflicts:** Crop, House, or Injury incidents."
    else: explanation = "📍 **Viewing All Data:** Showing all records."
    st.info(explanation)

    # --- F. MAP VISUALIZATION ---
    c_map, c_legend = st.columns([3, 1])
    
    with c_map:
        # Prepare Map Data
        map_df = df.copy()
        
        # Apply Reactive Filters
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

        # Map Center
        if not map_df.empty:
            center = [map_df['Latitude'].mean(), map_df['Longitude'].mean()]
            zoom = 11 if st.session_state.map_filter == 'Hotspot_View' else 9
        else:
            center = [23.5, 80.5]
            zoom = 9

        m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

        # 1. Boundaries
        if geojson_features:
            folium.GeoJson(
                data={"type": "FeatureCollection", "features": geojson_features},
                style_function=lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.1},
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Region:'])
            ).add_to(m)

        # 2. Markers
        if map_mode == "Heatmap":
            heat_data = [[r['Latitude'], r['Longitude'], max(r['Severity Score'], 1)] for _, r in map_df.iterrows()]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
        else:
            for _, row in map_df.iterrows():
                # Styling Logic
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
                    # Default Context Colors
                    if row['Injury']>0: color='darkred'
                    elif row['House Damage']>0: color='red'
                    elif row['Crop Damage']>0: color='orange'
                    elif row['Sighting Type'] == 'Direct': color='blue'
                    
                # Tooltip
                v_text = f"<br>🏠 {row['Nearest Village']} ({row['Distance to Village (km)']:.1f}km)" if 'Nearest Village' in row and pd.notnull(row['Nearest Village']) else ""
                tooltip = f"<b>{row['Date'].date()}</b><br>Loc: {row['Beat']}<br>Score: {row['Severity Score']:.1f}{v_text}"
                
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    tooltip=tooltip
                ).add_to(m)

        st_folium(m, width="100%", height=500, returned_objects=[])

    with c_legend:
        st.subheader("🏠 Affected Villages")
        if 'Nearest Village' in df.columns:
            # Filter affected villages based on CURRENT map view (not just global)
            affected_df = map_df[map_df['Near Village'] == True]
            if not affected_df.empty:
                v_counts = affected_df['Nearest Village'].value_counts().head(10).reset_index()
                v_counts.columns = ['Village', 'Incidents']
                st.dataframe(v_counts, use_container_width=True, hide_index=True)
            else:
                st.write("No villages within 500m of sightings in current view.")
        else:
            st.info("Upload 'centroids.csv' to see village data.")

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
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                fig_sun = None
                st.info("No data available for this view.")

    with r1c2:
        st.subheader("Hourly Activity (0-24h)")
        if not df.empty:
            h_counts = df['Hour'].value_counts().reindex(range(24), fill_value=0).reset_index()
            fig_hourly = px.bar(h_counts, x='Hour', y='count', title="Activity Peaks", 
                                color='count', color_continuous_scale='Viridis')
            st.plotly_chart(fig_hourly, use_container_width=True)
        else: fig_hourly = None

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Trend Analysis")
        daily = df.groupby('Date').size().reset_index(name='Count')
        fig_trend = px.line(daily, x='Date', y='Count', markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    with c2:
        st.subheader("Demographics")
        demog = df[['Male Count', 'Female Count', 'Calf Count']].sum().reset_index()
        demog.columns = ['Type', 'Count']
        fig_demog = px.pie(demog, values='Count', names='Type', hole=0.4)
        st.plotly_chart(fig_demog, use_container_width=True)

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
            st.plotly_chart(fig_damage, use_container_width=True)
        else:
            fig_damage = None
            st.info("No conflicts reported.")

    # --- H. DATA TABLES ---
    st.divider()
    t1, t2 = st.tabs(["⚠️ Damage Report", "🏆 Leaderboards"])
    with t1:
        dmg = df.groupby(['Division', 'Range']).agg({'Crop Damage': lambda x: (x>0).sum(), 'House Damage': lambda x: (x>0).sum(), 'Injury': lambda x: (x>0).sum()}).reset_index()
        dmg = dmg[(dmg['Crop Damage']>0)|(dmg['House Damage']>0)|(dmg['Injury']>0)]
        dmg.columns = ['Division', 'Range', '🌾 Crop', '🏠 House', '🚑 Injury']
        st.dataframe(dmg.style.background_gradient(cmap="Reds"), use_container_width=True)
    with t2:
        l1, l2 = st.columns(2)
        with l1:
            st.markdown("**Top Beats (Activity)**")
            if 'Beat' in df.columns:
                st.dataframe(df['Beat'].value_counts().reset_index(name='Entries').head(10), use_container_width=True, hide_index=True)
        with l2:
            st.markdown("**Top Reporters**")
            if 'Created By' in df.columns:
                st.dataframe(df['Created By'].value_counts().reset_index(name='Entries').head(10), use_container_width=True, hide_index=True)

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

else:
    st.info("👆 Upload CSV to begin.")

