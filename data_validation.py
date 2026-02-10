# data_validation.py
import pandas as pd
import streamlit as st

def validate_and_clean_data(df, show_warnings=True):
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    # Date conversion
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
    
    # Coordinate validation
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude'])
        
    return df

def calculate_derived_fields(df):
    # Ensure counts are numeric for charts
    count_cols = ['Male Count', 'Female Count', 'Calf Count', 'Unknown Count']
    for col in count_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df
