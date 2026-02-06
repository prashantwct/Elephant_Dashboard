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

# 1. IMPORT MODULAR LOGIC
from config import *
from data_validation import validate_and_clean_data, calculate_derived_fields
from spatial_analytics import (
    calculate_affected_villages_vectorized, 
    identify_risk_villages
)

# ==========================================
# CONFIGURATION & ASSETS
# ==========================================
st.set_page_config(
    page_title=APP_TITLE, 
    layout=LAYOUT, 
    page_icon=PAGE_ICON
)

# ==========================================
# UTILITY FUNCTIONS (Map Layers)
# ==========================================

@st.cache_data
def load_village_data(filepath="centroids.csv"):
    if os.path.exists(filepath):
        try:
            v_df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            v_df = pd.read_csv(filepath, encoding='ISO-8859-1')
        
        v_df.columns = [c.strip() for c in v_df.columns]
        rename_map = {
            'Lat': 'Latitude', 'lat': 'Latitude',
            'Lon': 'Longitude', 'Long': 'Longitude', 'lon': 'Longitude',
            'Name': 'Village', 'village': 'Village', 'VILLAGE': 'Village'
        }
        v_df = v_df.rename(columns=rename_map)
        if {'Latitude', 'Longitude', 'Village'}.issubset(v_df.columns):
            return v_df
    return None

def parse_kml_coordinates(placemark, ns):
    polygons = []
    for polygon in placemark.findall('.//kml:Polygon', ns):
        outer_boundary = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
        if outer_boundary is not None and outer_boundary.text:
            coord_str = outer_boundary.text.strip()
            poly_points = [[float(p.split(',')[0]), float(p.split(',')[1])] 
                           for p in coord_str.split() if len(p.split(',')) >= 2]
            if poly_points: polygons.append(poly_points)
    return polygons

def get_feature_name(placemark, ns):
    name_tag = placemark.find('kml:name', ns)
    if name_tag is not None and name_tag.text:
        return name_tag.text.strip().title()
    return "Unknown Area"

@st.cache_data
def load_map_data_as_geojson(folder_path="."):
    features = []
    files = glob.glob(os.path.join(folder_path, "*.kmz")) + glob.glob(os.path.join(folder_path, "*.kml"))
    
    for file_path in files:
        try:
            if file_path.endswith('.kmz'):
                with zipfile.ZipFile(file_path, 'r') as z:
                    kml_file = [f for f in z.namelist() if f.endswith('.kml')][0]
                    content = z.read(kml_file)
            else:
                with open(file_path, 'rb') as f: content = f.read()

            tree = ET.fromstring(content)
            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            for placemark in tree.findall('.//kml:Placemark', ns):
                name = get_feature_name(placemark, ns)
                polys = parse_kml_coordinates(placemark, ns)
                for poly_coords in polys:
                    features.append({
                        "type": "Feature",
                        "properties": {"name": name},
                        "geometry": {"type": "Polygon", "coordinates": [poly_coords]}
                    })
        except Exception: continue
    return features

# ==========================================
# SIDEBAR & PARAMETERS
# ==========================================
with st.sidebar:
    c1, c2 = st.columns([1, 1])
    c1.markdown(f'<img src="{LOGO_WCT}" width="100%">', unsafe_allow_html=True)
    c2.markdown(f'<img src="{LOGO_MP}" width="80%">', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("🏘️ Risk Parameters")
    with st.expander("Configure Logic", expanded=True):
        p_dmg_rad = st.slider("Damage Radius (km)", MIN_RADIUS_KM, MAX_DAMAGE_RADIUS_KM, DEFAULT_DAMAGE_RADIUS_KM)
        p_pres_rad = st.slider("Presence Radius (km)", MIN_RADIUS_KM, MAX_PRESENCE_RADIUS_KM, DEFAULT_PRESENCE_RADIUS_KM)
        p_days = st.slider("Consecutive Days", 1, MAX_CONSECUTIVE_DAYS, DEFAULT_CONSECUTIVE_DAYS)

# ==========================================
# MAIN DATA PIPELINE
# ==========================================
st.title("🐘 Elephant Sighting & Conflict Command Centre")
uploaded_csv = st.file_uploader("Upload Sightings Report (CSV)", type=['csv'])
village_df = load_village_data("centroids.csv")
geojson_features = load_map_data_as_geojson(".")

if uploaded_csv is not None:
    # 2. VALIDATE & ENRICH DATA
    df_raw = pd.read_csv(uploaded_csv)
    df_raw = validate_and_clean_data(df_raw, show_warnings=True)
    df_raw = calculate_derived_fields(df_raw)
    
    # 3. VECTORIZED SPATIAL ANALYTICS
    if village_df is not None:
        df_raw = calculate_affected_villages_vectorized(
            df_raw, village_df, radius_km=AFFECTED_VILLAGE_RADIUS_KM
        )

    # Global Scope Filters
    with st.sidebar:
        st.header("Filters")
        start, end = st.date_input("Date Range", [df_raw['Date'].min(), df_raw['Date'].max()])
        df = df_raw[(df_raw['Date'].dt.date >= start) & (df_raw['Date'].dt.date <= end)].copy()
        
        sel_div = st.selectbox("Filter Division", ['All'] + sorted(list(df['Division'].unique())))
        if sel_div != 'All': df = df[df['Division'] == sel_div]

    # ==========================================
    # VISUALIZATION TABS
    # ==========================================
    tab_map, tab_charts = st.tabs(["🗺️ Live Map", "📊 Analytics"])

    with tab_map:
        m = folium.Map(location=DEFAULT_MAP_CENTER, zoom_start=DEFAULT_MAP_ZOOM)
        
        # Add Boundaries
        if geojson_features:
            folium.GeoJson(
                data={"type": "FeatureCollection", "features": geojson_features},
                style_function=lambda x: {'fillColor': '#3388ff', 'color': '#3388ff', 'weight': 1, 'fillOpacity': 0.1},
                tooltip=folium.GeoJsonTooltip(fields=['name'], aliases=['Region:'])
            ).add_to(m)

        # Vectorized Risk Village Detection
        risk_villages = identify_risk_villages(df, village_df, p_dmg_rad, p_pres_rad, p_days)
        
        # Plot Risk Villages
        for v in risk_villages:
            folium.Circle(
                location=[v['Lat'], v['Lon']], radius=v['Radius'], color=v['Color'],
                fill=True, fill_opacity=0.15, tooltip=f"<b>{v['Village']}</b>: {v['Reason']}"
            ).add_to(m)

        # Plot Sightings
        for _, row in df.iterrows():
            folium.CircleMarker(
                [row['Latitude'], row['Longitude']],
                radius=5, color='blue', fill=True,
                tooltip=f"Date: {row['Date'].date()}<br>Beat: {row['Beat']}"
            ).add_to(m)

        st_folium(m, width=None, height=600, use_container_width=True)

    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            fig_trend = px.line(df.groupby('Date').size().reset_index(name='Count'), x='Date', y='Count', title="Sighting Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
        with c2:
            demog = df[['Male Count', 'Female Count', 'Calf Count', 'Unknown Count']].sum().reset_index()
            demog.columns = ['Type', 'Count']
            st.plotly_chart(px.pie(demog, values='Count', names='Type', hole=0.4, title="Population Split"), use_container_width=True)

else:
    st.info("Please upload a sightings CSV to begin.")
