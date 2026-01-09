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
    """Loads village centroid data if available."""
    if os.path.exists(filepath):
        try:
            v_df = pd.read_csv(filepath)
            required_cols = {'Village', 'Latitude', 'Longitude'}
            # Case-insensitive column check
            v_df.columns = [c.strip() for c in v_df.columns]
            # Normalize common variations
            rename_map = {'Lat': 'Latitude', 'Lon': 'Longitude', 'Long': 'Longitude', 'Name': 'Village'}
            v_df = v_df.rename(columns=rename_map)
            
            if required_cols.issubset(v_df.columns):
                return v_df
        except Exception:
            return None
    return None

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using NumPy
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def calculate_nearest_village(sightings_df, villages_df):
    """Finds the nearest village and distance for each sighting."""
    if villages_df is None or sightings_df.empty:
        return sightings_df

    # Cross join is too heavy for large datasets, using a simpler apply approach
    # (For very large datasets, Scipy cKDTree is better, but keeping dependencies minimal here)
    
    def get_nearest(row):
        dists = haversine_np(
            villages_df['Longitude'].values, 
            villages_df['Latitude'].values, 
            row['Longitude'], 
            row['Latitude']
        )
        min_idx = np.argmin(dists)
        return pd.Series([villages_df.iloc[min_idx]['Village'], dists[min_idx]])

    sightings_df[['Nearest Village', 'Distance to Village (km)']] = sightings_df.apply(get_nearest, axis=1)
    return sightings_df

# --- 2. KML PARSER ---

def parse_kml_coordinates(placemark, ns):
    """Extracts coordinates for GeoJSON (Lon, Lat)."""
    polygons = []
    for polygon in placemark.findall('.//kml:Polygon', ns):
        outer_boundary = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
        if outer_boundary is not None and outer_boundary.text:
            coord_str = outer_boundary.text.strip()
            poly_points = []
            for point in coord_str.split():
                try:
                    p = point.split(',')
                    lon, lat = float(p[0]), float(p[1])
                    poly_points.append([lon, lat])
                except (ValueError, IndexError):
                    continue
            if poly_points:
                polygons.append(poly_points)
    return polygons

def extract_name_from_html(html_content):
    try:
        clean_html = html_content.replace('\n', ' ').replace('\r', '')
        patterns = [
            r'<td>\s*(?:RANGE_NM|Range Name|RANGE)\s*</td>\s*<td>\s*(.*?)\s*</td>',
            r'<td>\s*(?:BEAT_NM|Beat Name|BEAT)\s*</td>\s*<td>\s*(.*?)\s*</td>',
            r'<td>\s*(?:NAME|Name_1)\s*</td>\s*<td>\s*(.*?)\s*</td>'
        ]
        for pat in patterns:
            match = re.search(pat, clean_html, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if val and val.upper() not in ["BTR", "0", "1", "NULL"]:
                    return val
        cells = re.findall(r'<td>(.*?)</td>', clean_html, re.IGNORECASE)
        for cell in cells:
            text = cell.strip()
            if text.upper() in ['FID', 'RANGE_NM', 'BEAT_NM', 'DIV_NM_MAP', 'BTR', 'SATNA', 'SINGRAULI', '0', '1', 'REMARKS']:
                continue
            if len(text) > 2 and not text.isdigit():
                return text
    except Exception:
        pass
    return None

def get_feature_name(placemark, ns):
    desc = placemark.find('kml:description', ns)
    if desc is not None and desc.text:
        name = extract_name_from_html(desc.text)
        if name: return name
    ext = placemark.find('kml:ExtendedData', ns)
    if ext is not None:
        for sd in ext.findall('.//kml:SimpleData', ns):
            if sd.get('name', '').lower() in ['range_nm', 'beat_nm', 'name_1', 'range', 'beat']:
                return sd.text
    name_tag = placemark.find('kml:name', ns)
    if name_tag is not None and name_tag.text:
        val = name_tag.text.strip()
        if val.upper() not in ["BTR", "SATNA", "SINGRAULI", "UMARIA", "SOUTH SHAHDOL", "NORTH SHAHDOL", "0", "1", "NONE"]:
            return val
    return "Unknown Area"

@st.cache_data
def load_map_data_as_geojson(folder_path="."):
    features = []
    all_beats = set()
    files = glob.glob(os.path.join(folder_path, "*.kmz")) + glob.glob(os.path.join(folder_path, "*.kml"))
    
    for file_path in files:
        try:
            content = None
            filename = os.path.basename(file_path)
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
                    clean_name = name.title() if name else "Unknown"
                    if clean_name != "Unknown": all_beats.add(clean_name)
                    polys = parse_kml_coordinates(placemark, ns)
                    for poly_coords in polys:
                        feature = {
                            "type": "Feature",
                            "properties": {"name": clean_name, "source": filename},
                            "geometry": {"type": "Polygon", "coordinates": [poly_coords]}
                        }
                        features.append(feature)
        except Exception:
            pass # Silent fail for map load
    return features, list(all_beats)

# --- 3. REPORT GENERATOR ---

def generate_full_html_report(df, map_object, fig_trend, fig_demog, fig_hourly, fig_damage, fig_sunburst, start_date, end_date):
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
    
    # Method section
    methodology = """
    <div style="margin-top: 20px; border: 1px solid #ddd; padding: 15px; background-color: #fafafa;">
        <h5>ℹ️ Methodology & Definitions</h5>
        <ul>
            <li><b>Conflict Severity Score:</b> Weighted Index. <i>Score = (Presence × 1) + (Crop × 2.5) + (House × 5) + (Injury × 20)</i>.</li>
            <li><b>HEC Ratio:</b> Percentage of entries that resulted in conflict events.</li>
        </ul>
    </div>
    <br>
    """
    
    map_html = map_object.get_root().render()
    map_iframe = f"""<iframe srcdoc="{map_html.replace('"', '&quot;')}" width="100%" height="500px" style="border:none;"></iframe>"""
    
    chart_trend = fig_trend.to_html(full_html=False, include_plotlyjs='cdn')
    chart_demog = fig_demog.to_html(full_html=False, include_plotlyjs='cdn')
    chart_damage = fig_damage.to_html(full_html=False, include_plotlyjs='cdn')
    chart_sunburst = fig_sunburst.to_html(full_html=False, include_plotlyjs='cdn') if fig_sunburst else ""
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
                <img src="{LOGO_MP}" width="80">
                <img src="{LOGO_WCT}" width="150">
            </div>
            {narrative}
            {methodology}
            <h3>📍 Spatial Distribution Map</h3>
            {map_iframe}
            <h3>📊 Analysis</h3>
            <div class="row"><div class="col-md-6">{chart_trend}</div><div class="col-md-6">{chart_demog}</div></div>
            <div class="row"><div class="col-md-6">{chart_damage}</div><div class="col-md-6">{chart_hourly}</div></div>
            <div class="row"><div class="col-md-12">{chart_sunburst}</div></div>
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

# SIDEBAR
with st.sidebar:
    c1, c2 = st.columns([1, 1])
    c1.markdown(f'<img src="{LOGO_WCT}" width="100%">', unsafe_allow_html=True)
    c2.markdown(f'<img src="{LOGO_MP}" width="80%">', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Intellectual Property**")
    st.info("© **Wildlife Conservation Trust, Mumbai**\n\nDeveloped for MP Forest Department.")

st.title("🐘 Elephant Sighting & Conflict Command Centre")

if 'map_filter' not in st.session_state: st.session_state.map_filter = 'All'

# A. LOAD DATA
geojson_features, all_loaded_beats = load_map_data_as_geojson(".")
village_df = load_village_data("centroids.csv") # Try loading village data

if geojson_features:
    with st.expander(f"✅ Map Data Loaded ({len(geojson_features)} regions)"):
        st.write("System optimized for fast loading.")
        
if village_df is not None:
    st.sidebar.success(f"✅ Village Data Loaded ({len(village_df)} villages)")
else:
    st.sidebar.warning("⚠️ 'centroids.csv' not found. Village analysis disabled.")

# B. FILE UPLOADER
uploaded_csv = st.file_uploader("Upload Sightings Report (CSV)", type=['csv'])

if uploaded_csv is not None:
    # 1. Load Raw Data
    df_raw = pd.read_csv(uploaded_csv)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y', errors='coerce')
    try: df_raw['Hour'] = pd.to_datetime(df_raw['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    except: df_raw['Hour'] = 0 
    df_raw = df_raw.dropna(subset=['Date'])
    
    cols = ['Total Count', 'Male Count', 'Female Count', 'Calf Count', 'Crop Damage', 'House Damage', 'Injury']
    for c in cols:
        if c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').fillna(0)

    if 'Range' in df_raw.columns: df_raw['Range'] = df_raw['Range'].astype(str).str.title().fillna('Unknown')
    if 'Beat' in df_raw.columns: df_raw['Beat'] = df_raw['Beat'].astype(str).str.title().fillna('Unknown')
    if 'Division' in df_raw.columns: df_raw['Division'] = df_raw['Division'].astype(str).str.title().fillna('Unknown')

    # 2. Advanced Calculations
    df_raw['Severity Score'] = (
        1 + (df_raw['Crop Damage']>0).astype(int)*2.5 + 
        (df_raw['House Damage']>0).astype(int)*5 + 
        (df_raw['Injury']>0).astype(int)*20
    )
    df_raw['DayOfWeek'] = df_raw['Date'].dt.day_name()
    df_raw['Is_Night'] = df_raw['Hour'].apply(lambda x: 1 if (x >= 18 or x <= 6) else 0)
    
    # Calculate Village Proximity if data exists
    if village_df is not None:
        df_raw = calculate_nearest_village(df_raw, village_df)
        df_raw['Near Village'] = df_raw['Distance to Village (km)'] < 0.5 # 500m threshold

    # --- FILTERS ---
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = df_raw['Date'].min(), df_raw['Date'].max()
        start, end = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        df = df_raw[(df_raw['Date'].dt.date >= start) & (df_raw['Date'].dt.date <= end)]
        
        # Trend Calc
        delta_days = (end - start).days + 1
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=delta_days - 1)
        df_prev = df_raw[(df_raw['Date'].dt.date >= prev_start) & (df_raw['Date'].dt.date <= prev_end)]

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
        map_mode = st.radio("Visualization Mode:", ["Pins", "Heatmap"], horizontal=True)

    # --- KPI METRICS ---
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

    # --- STRATEGIC INDICATORS ---
    st.markdown("### 🛡️ Strategic Indicators")
    cols_count = 5 if village_df is not None else 4
    k_cols = st.columns(cols_count)
    
    curr_sev = df['Severity Score'].sum()
    prev_sev = df_prev['Severity Score'].sum()
    k_cols[0].metric("🚨 Severity Score", f"{curr_sev:.1f}", delta=f"{curr_sev - prev_sev:.1f}", delta_color="inverse")
    
    curr_hec = (n_conflicts / n_sightings * 100) if n_sightings > 0 else 0
    prev_hec = (((df_prev['Crop Damage']>0)|(df_prev['House Damage']>0)|(df_prev['Injury']>0)).sum() / len(df_prev) * 100) if len(df_prev) > 0 else 0
    k_cols[1].metric("⚠️ HEC Ratio", f"{curr_hec:.1f}%", delta=f"{curr_hec - prev_hec:.1f}%", delta_color="inverse")
    
    curr_night = (df['Is_Night'].sum() / n_sightings * 100) if n_sightings > 0 else 0
    prev_night = (df_prev['Is_Night'].sum() / len(df_prev) * 100) if len(df_prev) > 0 else 0
    k_cols[2].metric("🌙 Night Activity", f"{curr_night:.1f}%", delta=f"{curr_night - prev_night:.1f}%", delta_color="inverse")
    
    # Hotspot Logic
    if not df.empty:
        beat_stats = df.groupby(['Beat', 'Range', 'Division'])['Severity Score'].sum().reset_index()
        beat_stats = beat_stats.sort_values(by='Severity Score', ascending=False)
        if not beat_stats.empty and beat_stats.iloc[0]['Severity Score'] > 0:
            top_beat = beat_stats.iloc[0]
            hotspot_info = f"Score: {top_beat['Severity Score']:.1f}\n({start} to {end}, {top_beat['Range']}, {top_beat['Division']})"
            k_cols[3].metric("🔁 Top Hotspot", top_beat['Beat'], hotspot_info)
        else:
            k_cols[3].metric("🔁 Top Hotspot", "None", "No Conflicts")
    else:
        k_cols[3].metric("🔁 Top Hotspot", "No Data", "")

    # Village Risk Metric
    if village_df is not None:
        high_risk_count = df['Near Village'].sum() if 'Near Village' in df.columns else 0
        k_cols[4].metric("🏠 Proximity Risk", f"{high_risk_count}", "Sightings <500m from Village", delta_color="inverse")

    st.info(f"📍 **Map View:** {st.session_state.map_filter}")

    # --- MAP ---
    st.divider()
    c_map, c_legend = st.columns([3, 1])
    
    with c_map:
        if not df.empty:
            center_lat, center_lon = df['Latitude'].mean(), df['Longitude'].mean()
            zoom = 9
        else:
            center_lat, center_lon, zoom = 23.5, 80.5, 7

        m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="OpenStreetMap")

        if geojson_features:
            if sel_range != 'All':
                filtered_features = [f for f in geojson_features if sel_range.lower() in f['properties']['name'].lower()]
                style_function = lambda x: {'fillColor': '#ffaf00', 'color': '#ffaf00', 'weight': 2, 'fillOpacity': 0.2}
            else:
                filtered_features = geojson_features
                style_function = lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.1}
            
            if filtered_features:
                folium.GeoJson(
                    data={"type": "FeatureCollection", "features": filtered_features},
                    style_function=style_function,
                    tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Region:'])
                ).add_to(m)

        map_df = df.copy()
        if st.session_state.map_filter == 'Conflict':
            map_df = map_df[(map_df['Crop Damage']>0) | (map_df['House Damage']>0) | (map_df['Injury']>0)]
        elif st.session_state.map_filter == 'Direct':
            map_df = map_df[map_df['Sighting Type'] == 'Direct']
        elif st.session_state.map_filter == 'Males':
            map_df = map_df[map_df['Male Count'] > 0]
        elif st.session_state.map_filter == 'Calves':
            map_df = map_df[map_df['Calf Count'] > 0]

        if map_mode == "Heatmap":
            heat_data = map_df[['Latitude', 'Longitude', 'Severity Score']].values.tolist()
            heat_data = [[row[0], row[1], max(row[2], 1)] for row in heat_data]
            HeatMap(heat_data, radius=15, blur=10, max_zoom=12).add_to(m)
        else:
            for _, row in map_df.iterrows():
                icon = 'eye'; color = 'green'; opacity = 1.0
                if row['Sighting Type'] == 'Direct': color = 'blue'
                if row['Injury'] > 0: icon = 'medkit'; color = 'darkred'; opacity = 1.0
                elif row['House Damage'] > 0: icon = 'home'; color = 'red'; opacity = 1.0
                elif row['Crop Damage'] > 0: icon = 'leaf'; color = 'orange'; opacity = 1.0
                
                # Dynamic tooltip with village info if available
                village_info = f"<br><b>🏠 Near:</b> {row['Nearest Village']} ({row['Distance to Village (km)']:.1f} km)" if 'Nearest Village' in row and pd.notnull(row['Nearest Village']) else ""
                
                tooltip_html = f"""
                <div style='font-size:12px; width:200px'>
                    <b>📅 {row['Date'].date()}</b> | {row['Time']}<br>
                    <b>👤 {row['Created By']}</b><br>
                    <hr style='margin:5px 0'>
                    <b>🐘 Signs:</b> {int(row['Total Count'])} (M:{int(row['Male Count'])} C:{int(row['Calf Count'])})<br>
                    <b>📍 {row['Beat']}</b>
                    {village_info}
                </div>
                """
                folium.Marker(
                    [row['Latitude'], row['Longitude']],
                    icon=folium.Icon(color=color, icon=icon, prefix='fa'),
                    tooltip=tooltip_html,
                    opacity=opacity
                ).add_to(m)

        st_folium(m, width="100%", height=500, returned_objects=[])

    with c_legend:
        st.subheader("Patrol Gaps")
        active_beats = {b.strip().title() for b in df['Beat'].unique()}
        all_beats_clean = {b.strip().title() for b in all_loaded_beats}
        missing_beats = list(all_beats_clean - active_beats)
        coverage_pct = len(active_beats) / len(all_beats_clean) * 100 if all_beats_clean else 0
        st.metric("Patrol Coverage", f"{coverage_pct:.1f}%")
        if missing_beats:
            with st.expander("🔻 Beats with Zero Reports"):
                st.write(", ".join(sorted(missing_beats)))

    # --- CHARTS ---
    st.divider()
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("Hierarchy Drill-Down")
        if not df.empty:
            sb_df = df.copy()
            sb_val = 'Total Count'
            if st.session_state.map_filter == 'Conflict': 
                sb_df = sb_df[(sb_df['Crop Damage']>0)|(sb_df['House Damage']>0)|(sb_df['Injury']>0)]
                sb_df['Incidents'] = 1; sb_val = 'Incidents'
            elif st.session_state.map_filter == 'Direct': sb_df = sb_df[sb_df['Sighting Type'] == 'Direct']
            elif st.session_state.map_filter == 'Males': sb_df = sb_df[sb_df['Male Count'] > 0]; sb_val = 'Male Count'
            elif st.session_state.map_filter == 'Calves': sb_df = sb_df[sb_df['Calf Count'] > 0]; sb_val = 'Calf Count'
            
            if not sb_df.empty:
                fig_sun = px.sunburst(sb_df, path=['Division', 'Range', 'Beat'], values=sb_val)
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.info("No data for current view.")
                fig_sun = None

    with r1c2:
        st.subheader("Activity by Hour (0-24h)")
        if not df.empty:
            hourly_counts = df['Hour'].value_counts().reindex(list(range(24)), fill_value=0).reset_index()
            hourly_counts.columns = ['Hour', 'Sightings']
            fig_hourly = px.bar(hourly_counts, x='Hour', y='Sightings', 
                                title="Hourly Intensity", color='Sightings', color_continuous_scale='Viridis')
            st.plotly_chart(fig_hourly, use_container_width=True)

    # --- REPORT GEN ---
    st.divider()
    if st.button("🖨️ Generate Full Report"):
        sb_chart = fig_sun if 'fig_sun' in locals() and fig_sun else None
        html_report = generate_full_html_report(df, m, fig_trend, fig_demog, fig_hourly, fig_damage, sb_chart, start, end)
        st.download_button("📥 Download HTML Report", html_report, "Elephant_Monitoring_Report.html", "text/html")
else:
    st.info("👆 Upload CSV to begin.")
