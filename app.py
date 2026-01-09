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

# --- 1. DATA LOADING & PROCESSING ---

@st.cache_data
def load_villages(csv_path="centroids.csv"):
    """Loads village centroids from CSV."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    try:
        df_v = pd.read_csv(csv_path)
        # Ensure lat/lon are numeric
        df_v['Latitude'] = pd.to_numeric(df_v['Latitude'], errors='coerce')
        df_v['Longitude'] = pd.to_numeric(df_v['Longitude'], errors='coerce')
        return df_v.dropna(subset=['Latitude', 'Longitude'])
    except Exception as e:
        st.error(f"Error loading villages: {e}")
        return pd.DataFrame()

def identify_affected_villages(sightings_df, villages_df, radius_km=5):
    """
    Identifies villages affected by:
    1. Crop Raiding (Crop Damage > 0 within 5km)
    2. Persistent Presence (Sightings within 5km for 3 consecutive days)
    """
    if sightings_df.empty or villages_df.empty:
        return pd.DataFrame()

    # Vectorized Haversine Distance (Sightings x Villages)
    # Convert to radians
    lat1 = np.radians(sightings_df['Latitude'].values)[:, np.newaxis] # (N, 1)
    lon1 = np.radians(sightings_df['Longitude'].values)[:, np.newaxis]
    lat2 = np.radians(villages_df['Latitude'].values)[np.newaxis, :] # (1, M)
    lon2 = np.radians(villages_df['Longitude'].values)[np.newaxis, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist_matrix = 6371 * c # Distance in km (N x M)

    # Boolean mask for proximity
    is_near = dist_matrix <= radius_km
    
    affected_data = []
    
    # Iterate over villages that have at least one nearby sighting
    # Sum over axis 0 (sightings) checks if any sighting is near village j
    nearby_counts = np.sum(is_near, axis=0)
    candidate_village_indices = np.where(nearby_counts > 0)[0]

    for v_idx in candidate_village_indices:
        # Get indices of sightings near this village
        s_indices = np.where(is_near[:, v_idx])[0]
        subset = sightings_df.iloc[s_indices]
        
        # Check Criteria 1: Crop Raiding
        has_crop_raid = (subset['Crop Damage'] > 0).any()
        
        # Check Criteria 2: 3 Consecutive Days
        dates = sorted(subset['Date'].dt.date.unique())
        has_3_consecutive = False
        if len(dates) >= 3:
            # Check for streak
            dates_series = pd.Series(dates)
            diffs = dates_series.diff().dt.days
            # Check for patterns like [NaN, 1, 1] which sums to 2 in a rolling window of 2 non-NaNs
            # Easier: Iterate
            streak = 0
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days == 1:
                    streak += 1
                    if streak >= 2: # 2 gaps = 3 days
                        has_3_consecutive = True
                        break
                else:
                    streak = 0
        
        if has_crop_raid or has_3_consecutive:
            reason = []
            if has_crop_raid: reason.append("Crop Raiding")
            if has_3_consecutive: reason.append("Persistent Presence (3+ Days)")
            
            affected_data.append({
                'Name': villages_df.iloc[v_idx].get('Name', f"Village {v_idx}"),
                'Latitude': villages_df.iloc[v_idx]['Latitude'],
                'Longitude': villages_df.iloc[v_idx]['Longitude'],
                'Reason': ", ".join(reason),
                'Risk_Level': 'High' if has_3_consecutive else 'Moderate'
            })
            
    return pd.DataFrame(affected_data)

# --- 2. EXISTING PARSERS ---

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
            filename = os.path.basename(file_path)
            content = None
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
                        feature = {"type": "Feature", "properties": {"name": clean_name, "source": filename}, "geometry": {"type": "Polygon", "coordinates": [poly_coords]}}
                        features.append(feature)
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
    return features, list(all_beats)

def generate_full_html_report(df, map_object, fig_trend, fig_demog, fig_hourly, fig_damage, fig_sunburst, start_date, end_date):
    """Generates HTML Report."""
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
    methodology = """
    <div style="margin-top: 20px; border: 1px solid #ddd; padding: 15px; background-color: #fafafa;">
        <h5>ℹ️ Methodology & Definitions</h5>
        <ul>
            <li><b>Affected Villages:</b> Villages with crop raiding (<5km) OR elephant presence for 3 consecutive days.</li>
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
    <head><title>Elephant Report</title><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    <style>body{{padding: 20px; font-family: sans-serif;}} h3{{margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 10px;}}</style></head>
    <body>
        <div class="container">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
                <img src="{LOGO_MP}" width="80" alt="MP Forest"><img src="{LOGO_WCT}" width="150" alt="WCT Mumbai">
            </div>
            {narrative}{methodology}
            <h3>📍 Spatial Distribution Map</h3>{map_iframe}
            <h3>📊 Analysis</h3><div class="row"><div class="col-md-6">{chart_trend}</div><div class="col-md-6">{chart_demog}</div></div>
            <div class="row"><div class="col-md-6">{chart_damage}</div><div class="col-md-6">{chart_hourly}</div></div>
            <div class="row"><div class="col-md-12">{chart_sunburst}</div></div>
            <h3>⚠️ Detailed Damage Data</h3>{table_html}
            <hr><p style="text-align:center; font-size:12px; color:gray;">IP Property of Wildlife Conservation Trust, Mumbai</p>
        </div>
    </body></html>
    """
    return full_html

# --- 3. MAIN APP ---

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
if 'map_style' not in st.session_state: st.session_state.map_style = 'Pins'

# A. LOAD DATA
geojson_features, all_loaded_beats = load_map_data_as_geojson(".")
villages_df = load_villages() # Load villages

if geojson_features:
    with st.expander(f"✅ Map Data Loaded ({len(geojson_features)} regions)"):
        st.write("System optimized for fast loading.")

# B. FILE UPLOADER
uploaded_csv = st.file_uploader("Upload Sightings Report (CSV)", type=['csv'])

if uploaded_csv is not None:
    df_raw = pd.read_csv(uploaded_csv)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='%d-%m-%Y', errors='coerce')
    try: df_raw['Hour'] = pd.to_datetime(df_raw['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    except: df_raw['Hour'] = 0 
    df_raw = df_raw.dropna(subset=['Date'])
    
    cols = ['Total Count', 'Male Count', 'Female Count', 'Calf Count', 'Crop Damage', 'House Damage', 'Injury']
    for c in cols:
        if c in df_raw.columns: df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce').fillna(0)

    for col in ['Range', 'Beat', 'Division']:
        if col in df_raw.columns: df_raw[col] = df_raw[col].astype(str).str.title().fillna('Unknown')

    # Calculations
    df_raw['Severity Score'] = (
        1 + (df_raw['Crop Damage']>0)*2.5 + (df_raw['House Damage']>0)*5 + (df_raw['Injury']>0)*20
    )
    df_raw['DayOfWeek'] = df_raw['Date'].dt.day_name()
    df_raw['Is_Night'] = df_raw['Hour'].apply(lambda x: 1 if (x >= 18 or x <= 6) else 0)

    # --- FILTERS ---
    with st.sidebar:
        st.header("Filters")
        min_date, max_date = df_raw['Date'].min(), df_raw['Date'].max()
        start, end = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
        
        df = df_raw[(df_raw['Date'].dt.date >= start) & (df_raw['Date'].dt.date <= end)]
        
        # Trends
        delta_days = (end - start).days + 1
        prev_start = start - timedelta(days=delta_days)
        df_prev = df_raw[(df_raw['Date'].dt.date >= prev_start) & (df_raw['Date'].dt.date < start)]

        divisions = ['All'] + sorted(list(df['Division'].unique()))
        sel_div = st.selectbox("Filter Division", divisions)
        if sel_div != 'All': 
            df = df[df['Division'] == sel_div]
            df_prev = df_prev[df_prev['Division'] == sel_div]
        
        ranges = ['All'] + sorted(list(df['Range'].unique()))
        sel_range = st.selectbox("Filter Range", ranges)
        if sel_range != 'All': 
            df = df[df['Range'] == sel_range]
            df_prev = df_prev[df_prev['Range'] == sel_range]
        
        st.divider()
        st.header("Map Settings")
        map_mode = st.radio("Visualization Mode:", ["Pins", "Heatmap"], horizontal=True)

    # --- KPI LOGIC ---
    n_sightings = len(df)
    n_cumulative = int(df['Total Count'].sum())
    n_direct = len(df[df['Sighting Type'] == 'Direct'])
    n_conflicts = ((df['Crop Damage']>0) | (df['House Damage']>0) | (df['Injury']>0)).sum()
    n_males = int(df['Male Count'].sum())
    n_calves = int(df['Calf Count'].sum())
    
    # Calculate Affected Villages
    affected_df = identify_affected_villages(df, villages_df)
    n_affected_villages = len(affected_df)

    st.markdown("### 📈 Operational Overview")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    if c1.button(f"📝 Entries\n\n# {n_sightings}", use_container_width=True): st.session_state.map_filter = 'All'
    if c2.button(f"🐾 Cumulative\n\n# {n_cumulative}", use_container_width=True): st.session_state.map_filter = 'All'
    if c3.button(f"👁️ Direct\n\n# {n_direct}", use_container_width=True): st.session_state.map_filter = 'Direct'
    if c4.button(f"🔥 Conflicts\n\n# {n_conflicts}", use_container_width=True): st.session_state.map_filter = 'Conflict'
    if c5.button(f"♂️ Males\n\n# {n_males}", use_container_width=True): st.session_state.map_filter = 'Males'
    if c6.button(f"👶 Calves\n\n# {n_calves}", use_container_width=True): st.session_state.map_filter = 'Calves'
    if c7.button(f"🏘️ Affected\n\n# {n_affected_villages}", use_container_width=True): st.session_state.map_filter = 'Affected'

    st.markdown("### 🛡️ Strategic Indicators (vs. Previous Period)")
    k1, k2, k3, k4 = st.columns(4)
    
    curr_sev = df['Severity Score'].sum()
    prev_sev = df_prev['Severity Score'].sum()
    k1.metric("🚨 Conflict Severity Score", f"{curr_sev:.1f}", delta=f"{curr_sev - prev_sev:.1f}", delta_color="inverse")
    
    curr_hec = (n_conflicts / n_sightings * 100) if n_sightings > 0 else 0
    prev_conf = ((df_prev['Crop Damage']>0) | (df_prev['House Damage']>0) | (df_prev['Injury']>0)).sum()
    prev_sight = len(df_prev)
    prev_hec = (prev_conf / prev_sight * 100) if prev_sight > 0 else 0
    k2.metric("⚠️ HEC Ratio", f"{curr_hec:.1f}%", delta=f"{curr_hec - prev_hec:.1f}%", delta_color="inverse")
    
    curr_night = (df['Is_Night'].sum() / n_sightings * 100) if n_sightings > 0 else 0
    prev_night = (df_prev['Is_Night'].sum() / prev_sight * 100) if prev_sight > 0 else 0
    k3.metric("🌙 Nocturnal Activity", f"{curr_night:.1f}%", delta=f"{curr_night - prev_night:.1f}%", delta_color="inverse")
    
    if not df.empty:
        beat_stats = df.groupby(['Beat', 'Range', 'Division'])['Severity Score'].sum().reset_index().sort_values(by='Severity Score', ascending=False)
        if not beat_stats.empty and beat_stats.iloc[0]['Severity Score'] > 0:
            top = beat_stats.iloc[0]
            k4.metric("🔁 Top Conflict Hotspot", top['Beat'], f"Score: {top['Severity Score']:.1f}\n({start}~{end}, {top['Range']}, {top['Division']})")
        else:
            k4.metric("🔁 Top Conflict Hotspot", "None", "No Conflicts")
    else:
        k4.metric("🔁 Top Conflict Hotspot", "No Data", "")

    st.info(f"📍 **Map View:** {st.session_state.map_filter} | **Mode:** {map_mode}")

    # --- MAP ---
    st.divider()
    c_map, c_legend = st.columns([3, 1])
    
    with c_map:
        center_lat = df['Latitude'].mean() if not df.empty else 23.5
        center_lon = df['Longitude'].mean() if not df.empty else 80.5
        m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="OpenStreetMap")

        if geojson_features:
            if sel_range != 'All':
                filtered_features = [f for f in geojson_features if sel_range.lower() in f['properties']['name'].lower()]
                style = lambda x: {'fillColor': '#ffaf00', 'color': '#ffaf00', 'weight': 2, 'fillOpacity': 0.2}
            else:
                filtered_features = geojson_features
                style = lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.1}
            folium.GeoJson(data={"type": "FeatureCollection", "features": filtered_features}, style_function=style, tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Region:'])).add_to(m)

        # MARKERS
        map_df = df.copy()
        
        # Handle "Affected Villages" view specially
        if st.session_state.map_filter == 'Affected':
            if not affected_df.empty:
                for _, row in affected_df.iterrows():
                    color = 'red' if row['Risk_Level'] == 'High' else 'orange'
                    icon = 'fire' if row['Risk_Level'] == 'High' else 'home'
                    tooltip = f"<b>{row['Name']}</b><br>Reason: {row['Reason']}"
                    folium.Marker([row['Latitude'], row['Longitude']], icon=folium.Icon(color=color, icon=icon, prefix='fa'), tooltip=tooltip).add_to(m)
            else:
                st.warning("No villages meet the 'Affected' criteria (Crop Raid or 3-day presence) in this period.")
        
        else:
            # Standard Views
            if st.session_state.map_filter == 'Conflict':
                map_df = map_df[(map_df['Crop Damage']>0) | (map_df['House Damage']>0) | (map_df['Injury']>0)]
            elif st.session_state.map_filter == 'Direct':
                map_df = map_df[map_df['Sighting Type'] == 'Direct']
            elif st.session_state.map_filter == 'Males':
                map_df = map_df[map_df['Male Count'] > 0]
            elif st.session_state.map_filter == 'Calves':
                map_df = map_df[map_df['Calf Count'] > 0]

            if map_mode == "Heatmap":
                heat_data = [[row['Latitude'], row['Longitude'], max(row['Severity Score'], 1)] for _, row in map_df.iterrows()]
                HeatMap(heat_data, radius=15, blur=10, max_zoom=12).add_to(m)
            else:
                for _, row in map_df.iterrows():
                    icon, color, opacity = 'eye', 'green', 1.0
                    if row['Sighting Type'] == 'Direct': color = 'blue'
                    if row['Injury'] > 0: icon, color = 'medkit', 'darkred'
                    elif row['House Damage'] > 0: icon, color = 'home', 'red'
                    elif row['Crop Damage'] > 0: icon, color = 'leaf', 'orange'
                    if (row['Crop Damage']>0 or row['House Damage']>0 or row['Injury']>0): opacity = 0.5
                    
                    tooltip = f"<div style='width:150px'><b>{row['Date'].date()}</b> | {row['Time']}<br>Count: {int(row['Total Count'])}<br>Loc: {row['Beat']}</div>"
                    folium.Marker([row['Latitude'], row['Longitude']], icon=folium.Icon(color=color, icon=icon, prefix='fa'), tooltip=tooltip, opacity=opacity).add_to(m)

        st_folium(m, width="100%", height=500, returned_objects=[])

    with c_legend:
        st.subheader("Patrol Gaps")
        active = set(df['Beat'].unique())
        active_clean = {b.strip().title() for b in active}
        all_clean = {b.strip().title() for b in all_loaded_beats}
        missing = list(all_clean - active_clean)
        pct = len(active_clean)/len(all_clean)*100 if all_clean else 0
        st.metric("Patrol Coverage", f"{pct:.1f}%")
        if missing:
            with st.expander("🔻 Zero Reports"): st.write(", ".join(sorted(missing)))

    # --- CHARTS ---
    st.divider()
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("Hierarchy Drill-Down")
        if not df.empty:
            sb_df = df.copy()
            val, title = 'Total Count', "Overall Distribution"
            if st.session_state.map_filter == 'Conflict':
                sb_df = sb_df[(sb_df['Crop Damage']>0)|(sb_df['House Damage']>0)|(sb_df['Injury']>0)]
                sb_df['Incidents'] = 1; val, title = 'Incidents', "Conflict Incidents"
            elif st.session_state.map_filter == 'Males':
                sb_df = sb_df[sb_df['Male Count']>0]; val, title = 'Male Count', "Male Distribution"
            elif st.session_state.map_filter == 'Calves':
                sb_df = sb_df[sb_df['Calf Count']>0]; val, title = 'Calf Count', "Calf Distribution"
            
            if not sb_df.empty:
                fig_sun = px.sunburst(sb_df, path=['Division', 'Range', 'Beat'], values=val, title=title)
                st.plotly_chart(fig_sun, use_container_width=True)
            else: st.info("No data for selection."); fig_sun = None
    with r1c2:
        st.subheader("Hourly Activity (0-24h)")
        if not df.empty:
            h_counts = df['Hour'].value_counts().reindex(range(24), fill_value=0).reset_index()
            h_counts.columns = ['Hour', 'Sightings']
            fig_hourly = px.bar(h_counts, x='Hour', y='Sightings', title="Activity Intensity", color='Sightings', color_continuous_scale='Viridis')
            st.plotly_chart(fig_hourly, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.plotly_chart(px.line(df.groupby('Date').size().reset_index(name='Count'), x='Date', y='Count', title="Daily Trends", markers=True), use_container_width=True)
    with c2: st.plotly_chart(px.pie(df[['Male Count', 'Female Count', 'Calf Count']].sum().reset_index(), values=0, names='index', title="Demographics"), use_container_width=True)
    with c3:
        d_sums = df[['Crop Damage', 'House Damage', 'Injury']].apply(lambda x: (x>0).sum()).reset_index()
        d_sums.columns = ['Type', 'Incidents']
        d_sums = d_sums[d_sums['Incidents']>0]
        if not d_sums.empty: st.plotly_chart(px.pie(d_sums, values='Incidents', names='Type', title="Conflict Types", color='Type', color_discrete_map={'Crop Damage':'orange', 'House Damage':'red', 'Injury':'darkred'}), use_container_width=True)
        else: st.info("No conflicts.")

    # --- REPORT ---
    st.divider()
    if st.button("🖨️ Generate Full Report"):
        html = generate_full_html_report(df, m, px.line(df.groupby('Date').size().reset_index(name='Count'), x='Date', y='Count'), px.pie(df[['Male Count', 'Female Count', 'Calf Count']].sum().reset_index(), values=0, names='index'), fig_hourly, px.pie(d_sums, values='Incidents', names='Type') if not d_sums.empty else None, fig_sun if 'fig_sun' in locals() and fig_sun else None, start, end)
        st.download_button("📥 Download Report", html, "Elephant_Report.html", "text/html")

else:
    st.info("👆 Upload CSV to begin.")
