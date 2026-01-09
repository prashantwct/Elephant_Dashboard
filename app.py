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

# --- CONFIGURATION ---
LOGO_WCT = "https://www.wildlifeconservationtrust.org/wp-content/uploads/2016/09/wct-logo.png"
LOGO_MP = "https://upload.wikimedia.org/wikipedia/commons/2/23/Emblem_of_Madhya_Pradesh.svg"

# --- 1. UTILITY FUNCTIONS ---

@st.cache_data
def load_village_data(filepath="centroids.csv"):
    if os.path.exists(filepath):
        try:
            try: v_df = pd.read_csv(filepath)
            except UnicodeDecodeError: v_df = pd.read_csv(filepath, encoding='ISO-8859-1')
            
            v_df.columns = [c.strip() for c in v_df.columns]
            rename_map = {'Lat': 'Latitude', 'Lon': 'Longitude', 'Long': 'Longitude', 'Name': 'Village'}
            v_df = v_df.rename(columns=rename_map)
            return v_df
        except Exception: return None
    return None

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6367 * c

def calculate_nearest_village(sightings_df, villages_df):
    if villages_df is None or sightings_df.empty: return sightings_df
    
    # Extract arrays
    v_lons = villages_df['Longitude'].values
    v_lats = villages_df['Latitude'].values
    v_names = villages_df['Village'].values
    
    def get_nearest(row):
        dists = haversine_np(v_lons, v_lats, row['Longitude'], row['Latitude'])
        min_idx = np.argmin(dists)
        return pd.Series([v_names[min_idx], dists[min_idx]])

    sightings_df[['Nearest Village', 'Distance to Village (km)']] = sightings_df.apply(get_nearest, axis=1)
    return sightings_df

# --- 2. KML PARSER ---

def parse_kml_coordinates(placemark, ns):
    polygons = []
    for polygon in placemark.findall('.//kml:Polygon', ns):
        outer_boundary = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
        if outer_boundary is not None and outer_boundary.text:
            coord_str = outer_boundary.text.strip()
            poly_points = []
            for point in coord_str.split():
                try:
                    p = point.split(',')
                    poly_points.append([float(p[1]), float(p[0])]) # Lat, Lon
                except: continue
            if poly_points: polygons.append(poly_points)
    return polygons

def get_feature_name(placemark, ns):
    name_tag = placemark.find('kml:name', ns)
    if name_tag is not None and name_tag.text:
        return name_tag.text.strip()
    return "Unknown"

@st.cache_data
def load_map_data_as_geojson(folder_path="."):
    features = []
    files = glob.glob(os.path.join(folder_path, "*.kmz")) + glob.glob(os.path.join(folder_path, "*.kml"))
    
    for file_path in files:
        try:
            content = None
            if file_path.endswith('.kmz'):
                with zipfile.ZipFile(file_path, 'r') as z:
                    kml_inside = [f for f in z.namelist() if f.endswith('.kml')]
                    if kml_inside:
                        with z.open(kml_inside[0]) as f: content = f.read()
            else:
                with open(file_path, 'rb') as f: content = f.read()

            if content:
                tree = ET.fromstring(content)
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                for placemark in tree.findall('.//kml:Placemark', ns):
                    name = get_feature_name(placemark, ns)
                    polys = parse_kml_coordinates(placemark, ns)
                    for poly_coords in polys:
                        features.append({
                            "type": "Feature",
                            "properties": {"name": name.title()},
                            "geometry": {"type": "Polygon", "coordinates": [[ [p[1], p[0]] for p in poly_coords ]]} 
                        })
        except: pass
    return features

# --- 3. REPORT GENERATOR ---

def generate_full_html_report(df, map_object, fig_trend, fig_demog, fig_hourly, fig_damage, start_date, end_date):
    total_sightings = len(df)
    cumulative_count = int(df['Total Count'].sum())
    conflicts = df[(df['Crop Damage']>0) | (df['House Damage']>0) | (df['Injury']>0)]
    conflict_count = len(conflicts)
    severity_score = int(df['Severity Score'].sum())
    
    narrative = f"""
    <h2>🐘 Elephant Monitoring Report</h2>
    <p><b>Period:</b> {start_date} to {end_date}</p>
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
        <h4>Executive Summary</h4>
        <p>This report covers <b>{total_sightings}</b> sighting entries recording <b>{cumulative_count}</b> cumulative elephant signs.</p>
        <p><b>Risk Assessment:</b> The calculated Conflict Severity Score is <b>{severity_score}</b> based on {conflict_count} incidents.</p>
    </div>
    <br>
    """
    
    map_html = map_object.get_root().render()
    map_iframe = f"""<iframe srcdoc="{map_html.replace('"', '&quot;')}" width="100%" height="500px" style="border:none;"></iframe>"""
    
    chart_trend = fig_trend.to_html(full_html=False, include_plotlyjs='cdn')
    chart_demog = fig_demog.to_html(full_html=False, include_plotlyjs='cdn')
    chart_damage = fig_damage.to_html(full_html=False, include_plotlyjs='cdn')
    chart_hourly = fig_hourly.to_html(full_html=False, include_plotlyjs='cdn') if fig_hourly else "<p>No data.</p>"
    
    dmg_table = df.groupby(['Division', 'Range']).agg({'Crop Damage': lambda x: (x>0).sum(), 'House Damage': lambda x: (x>0).sum(), 'Injury': lambda x: (x>0).sum()}).reset_index()
    dmg_table = dmg_table[(dmg_table['Crop Damage']>0)|(dmg_table['House Damage']>0)|(dmg_table['Injury']>0)]
    table_html = dmg_table.to_html(classes='table table-striped', index=False)

    full_html = f"""
    <html>
    <head>
        <title>Elephant Report</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
        <style>body{{padding: 20px; font-family: sans-serif;}} h3{{margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px;}}</style>
    </head>
    <body>
        <div class="container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <img src="{LOGO_MP}" width="80" alt="MP Forest">
                <img src="{LOGO_WCT}" width="150" alt="WCT Mumbai">
            </div>
            {narrative}
            <h3>📍 Spatial Distribution Map</h3>
            {map_iframe}
            <h3>📊 Analysis</h3>
            <div class="row"><div class="col-md-6">{chart_trend}</div><div class="col-md-6">{chart_demog}</div></div>
            <div class="row"><div class="col-md-6">{chart_damage}</div><div class="col-md-6">{chart_hourly}</div></div>
            <h3>⚠️ Detailed Damage Data</h3>
            {table_html}
            <hr><p style="text-align:center; font-size:12px; color:gray;">IP Property of Wildlife Conservation Trust, Mumbai</p>
        </div>
    </body>
    </html>
    """
    return full_html

# --- 4. MAIN APP ---

st.set_page_config(page_title="Elephant Dashboard", layout="wide")

# Sidebar
with st.sidebar:
    c1, c2 = st.columns([1, 1])
    c1.markdown(f'<img src="{LOGO_WCT}" width="100%">', unsafe_allow_html=True)
    c2.markdown(f'<img src="{LOGO_MP}" width="80%">', unsafe_allow_html=True)
    st.markdown("---")
    st.info("© **Wildlife Conservation Trust, Mumbai**")

st.title("🐘 Elephant Sighting & Conflict Command Centre")

if 'map_filter' not in st.session_state: st.session_state.map_filter = 'All'
if 'hotspot_beat' not in st.session_state: st.session_state.hotspot_beat = None

# A. LOAD DATA
geojson_features = load_map_data_as_geojson(".")
village_df = load_village_data("centroids.csv")

if geojson_features:
    with st.expander(f"✅ Map Data Loaded ({len(geojson_features)} regions)"): st.write("Ready")
if village_df is not None: 
    st.sidebar.success(f"✅ Villages Loaded")
else:
    st.sidebar.warning("⚠️ 'centroids.csv' not found. Village analysis disabled.")

# B. FILE UPLOADER
uploaded_csv = st.file_uploader("Upload Sightings Report (CSV)", type=['csv'])

if uploaded_csv is not None:
    # 1. Load & Clean
    try: df_raw = pd.read_csv(uploaded_csv)
    except: 
        uploaded_csv.seek(0)
        df_raw = pd.read_csv(uploaded_csv, encoding='ISO-8859-1')
        
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y', errors='coerce')
    try: df_raw['Hour'] = pd.to_datetime(df_raw['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    except: df_raw['Hour'] = 0 
    df_raw = df_raw.dropna(subset=['Date'])
    
    cols = ['Total Count', 'Male Count', 'Female Count', 'Calf Count', 'Crop Damage', 'House Damage', 'Injury']
    for c in cols: 
        if c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').fillna(0)

    for c in ['Range', 'Beat', 'Division']:
        if c in df_raw.columns: df_raw[c] = df_raw[c].astype(str).str.title().fillna('Unknown')

    # 2. Calculations (Raw)
    # Severity: Presence(1) + Crop(2.5) + House(5) + Injury(20)
    df_raw['Severity Score'] = (
        1 + (df_raw['Crop Damage']>0)*2.5 + (df_raw['House Damage']>0)*5 + (df_raw['Injury']>0)*20
    )
    df_raw['Is_Night'] = df_raw['Hour'].apply(lambda x: 1 if (x >= 18 or x <= 6) else 0)
    
    if village_df is not None:
        df_raw = calculate_nearest_village(df_raw, village_df)
        df_raw['Near Village'] = df_raw['Distance to Village (km)'] < 0.5 

    # --- FILTERS ---
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = df_raw['Date'].min(), df_raw['Date'].max()
        start, end = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        # Apply Time Filter Globally
        df = df_raw[(df_raw['Date'].dt.date >= start) & (df_raw['Date'].dt.date <= end)]
        
        # Trend Data (Relative to selected duration)
        delta_days = (end - start).days + 1
        df_prev = df_raw[(df_raw['Date'].dt.date >= (start - timedelta(days=delta_days))) & (df_raw['Date'].dt.date < start)]

        divisions = ['All'] + sorted(list(df['Division'].unique()))
        sel_div = st.selectbox("Filter Division", divisions)
        if sel_div != 'All': 
            df = df[df['Division'] == sel_div]
            df_prev = df_prev[df_prev['Division'] == sel_div] # Apply same filters to trend data for fair comparison
        
        map_mode = st.radio("Visualization Mode:", ["Pins", "Heatmap"], horizontal=True)

    # --- METRICS CALCULATOR ---
    n_sightings = len(df)
    n_conflicts = ((df['Crop Damage']>0)|(df['House Damage']>0)|(df['Injury']>0)).sum()
    
    # 1. Severity
    curr_sev = df['Severity Score'].sum()
    prev_sev = df_prev['Severity Score'].sum()
    delta_sev = curr_sev - prev_sev
    
    # 2. HEC Ratio
    curr_hec = (n_conflicts / n_sightings * 100) if n_sightings > 0 else 0
    prev_hec = (((df_prev['Crop Damage']>0)|(df_prev['House Damage']>0)|(df_prev['Injury']>0)).sum() / len(df_prev) * 100) if len(df_prev) > 0 else 0
    
    # 3. Night
    curr_night = (df['Is_Night'].sum() / n_sightings * 100) if n_sightings > 0 else 0
    prev_night = (df_prev['Is_Night'].sum() / len(df_prev) * 100) if len(df_prev) > 0 else 0
    
    # 4. Hotspot
    if not df.empty:
        beat_stats = df.groupby('Beat')['Severity Score'].sum().reset_index().sort_values('Severity Score', ascending=False)
        top_beat = beat_stats.iloc[0]['Beat'] if not beat_stats.empty and beat_stats.iloc[0]['Severity Score'] > 0 else "None"
        st.session_state.hotspot_beat = top_beat
    else: top_beat = "None"

    # --- UI: OPERATIONAL BUTTONS ---
    st.markdown("### 📈 Operational Overview")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    if c1.button(f"📝 Entries\n\n# {len(df)}", use_container_width=True): st.session_state.map_filter = 'All'
    if c2.button(f"🐾 Cumulative\n\n# {int(df['Total Count'].sum())}", use_container_width=True): st.session_state.map_filter = 'All'
    if c3.button(f"👁️ Direct\n\n# {len(df[df['Sighting Type']=='Direct'])}", use_container_width=True): st.session_state.map_filter = 'Direct'
    if c4.button(f"🔥 Conflicts\n\n# {n_conflicts}", use_container_width=True): st.session_state.map_filter = 'Conflict'
    if c5.button(f"♂️ Males\n\n# {int(df['Male Count'].sum())}", use_container_width=True): st.session_state.map_filter = 'Males'
    if c6.button(f"👶 Calves\n\n# {int(df['Calf Count'].sum())}", use_container_width=True): st.session_state.map_filter = 'Calves'

    # --- UI: REACTIVE STRATEGIC INDICATORS ---
    st.markdown("### 🛡️ Strategic Indicators (Click to Visualize)")
    k1, k2, k3, k4 = st.columns(4)
    
    if k1.button(f"🚨 Severity Score\n\n{curr_sev:.1f} ({'+' if delta_sev>0 else ''}{delta_sev:.1f})", use_container_width=True):
        st.session_state.map_filter = 'Severity_View'
        
    if k2.button(f"⚠️ HEC Ratio\n\n{curr_hec:.1f}% ({'+' if (curr_hec-prev_hec)>0 else ''}{curr_hec-prev_hec:.1f}%)", use_container_width=True):
        st.session_state.map_filter = 'Conflict'
        
    if k3.button(f"🌙 Night Activity\n\n{curr_night:.1f}% ({'+' if (curr_night-prev_night)>0 else ''}{curr_night-prev_night:.1f}%)", use_container_width=True):
        st.session_state.map_filter = 'Night_View'
        
    if k4.button(f"🔁 Hotspot: {top_beat}\n\n(Click to Focus)", use_container_width=True):
        st.session_state.map_filter = 'Hotspot_View'

    # --- DYNAMIC EXPLANATION PANEL ---
    explanation_text = ""
    if st.session_state.map_filter == 'Severity_View':
        explanation_text = "🔴 **Viewing Severity:** Markers are sized by their Severity Score (Injury=Big, Presence=Small)."
    elif st.session_state.map_filter == 'Night_View':
        explanation_text = "🟣 **Viewing Night Activity:** Showing only sightings recorded between 6:00 PM and 6:00 AM."
    elif st.session_state.map_filter == 'Hotspot_View':
        explanation_text = f"🎯 **Viewing Hotspot:** Focused on **{top_beat}** which has the highest cumulative Severity Score in this period."
    elif st.session_state.map_filter == 'Conflict':
        explanation_text = "🔥 **Viewing Conflicts:** Showing only incidents involving Crop Damage, House Damage, or Injury."
    else:
        explanation_text = "📍 **Viewing All Data:** Showing all recorded presence signs and sightings."
    
    st.info(explanation_text)

    # --- MAP LOGIC ---
    c_map, c_legend = st.columns([3, 1])
    with c_map:
        # Filter Logic for Map
        map_df = df.copy()
        zoom_loc = [df['Latitude'].mean(), df['Longitude'].mean()] if not df.empty else [23.5, 80.5]
        zoom_start = 9

        if st.session_state.map_filter == 'Severity_View': pass
        elif st.session_state.map_filter == 'Conflict':
            map_df = map_df[(map_df['Crop Damage']>0)|(map_df['House Damage']>0)|(map_df['Injury']>0)]
        elif st.session_state.map_filter == 'Night_View':
            map_df = map_df[map_df['Is_Night'] == 1]
        elif st.session_state.map_filter == 'Hotspot_View' and st.session_state.hotspot_beat != "None":
            map_df = map_df[map_df['Beat'] == st.session_state.hotspot_beat]
            if not map_df.empty:
                zoom_loc = [map_df.iloc[0]['Latitude'], map_df.iloc[0]['Longitude']]
                zoom_start = 11
        elif st.session_state.map_filter == 'Direct': map_df = map_df[map_df['Sighting Type'] == 'Direct']
        elif st.session_state.map_filter == 'Males': map_df = map_df[map_df['Male Count'] > 0]
        elif st.session_state.map_filter == 'Calves': map_df = map_df[map_df['Calf Count'] > 0]

        # Render Map
        m = folium.Map(location=zoom_loc, zoom_start=zoom_start, tiles="OpenStreetMap")

        if geojson_features:
            folium.GeoJson(
                data={"type": "FeatureCollection", "features": geojson_features},
                style_function=lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.1}
            ).add_to(m)

        if map_mode == "Heatmap":
            heat_data = map_df[['Latitude', 'Longitude', 'Severity Score']].values.tolist()
            heat_data = [[r[0], r[1], max(r[2], 1)] for r in heat_data]
            HeatMap(heat_data, radius=15, blur=10).add_to(m)
        else:
            for _, row in map_df.iterrows():
                # Color Logic
                if st.session_state.map_filter == 'Severity_View':
                    score = row['Severity Score']
                    color = 'darkred' if score >= 20 else ('red' if score >= 5 else ('orange' if score >= 2.5 else 'blue'))
                    radius = 15 if score >= 20 else (10 if score >= 5 else (7 if score >= 2.5 else 4))
                elif st.session_state.map_filter == 'Night_View':
                    color = 'purple'; radius = 6
                else:
                    color = 'green'
                    if row['Sighting Type'] == 'Direct': color = 'blue'
                    if row['Injury'] > 0: color = 'darkred'
                    elif row['House Damage'] > 0: color = 'red'
                    elif row['Crop Damage'] > 0: color = 'orange'
                    radius = 6

                # Tooltip Construction
                v_info = f"<br>🏠 {row['Nearest Village']} ({row['Distance to Village (km)']:.1f}km)" if 'Nearest Village' in row and pd.notnull(row['Nearest Village']) else ""
                tooltip = f"<b>{row['Date'].date()}</b><br>{row['Beat']}<br>Score: {row['Severity Score']:.1f}{v_info}"
                
                folium.CircleMarker(
                    [row['Latitude'], row['Longitude']],
                    radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
                    tooltip=tooltip
                ).add_to(m)

        st_folium(m, width="100%", height=500, returned_objects=[])

    with c_legend:
        st.subheader("Affected Villages")
        if 'Nearest Village' in df.columns:
            # Filter affected villages based on CURRENT map view (not just global)
            affected_df = map_df[map_df['Near Village'] == True]
            if not affected_df.empty:
                v_counts = affected_df['Nearest Village'].value_counts().head(10)
                st.dataframe(v_counts, use_container_width=True, column_config={"count": "Incidents"})
            else:
                st.write("No villages within 500m of sightings in current view.")
        else:
            st.info("Upload 'centroids.csv' to see village data.")

    # --- CHARTS ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hourly Activity")
        if not df.empty:
            h_counts = df['Hour'].value_counts().reindex(range(24), fill_value=0).reset_index()
            st.plotly_chart(px.bar(h_counts, x='Hour', y='count', title="Activity (0-24h)"), use_container_width=True)
    with c2:
        st.subheader("Trend")
        daily = df.groupby('Date').size().reset_index(name='Count')
        st.plotly_chart(px.line(daily, x='Date', y='Count', markers=True), use_container_width=True)

    st.divider()
    if st.button("🖨️ Generate Full Report"):
        html_report = generate_full_html_report(df, m, fig_trend, fig_demog, fig_hourly, fig_damage, start, end)
        st.download_button("📥 Download HTML Report", html_report, "Elephant_Monitoring_Report.html", "text/html")

else:
    st.info("👆 Upload CSV to begin.")
