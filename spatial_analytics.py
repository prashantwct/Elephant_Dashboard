# spatial_analytics.py
import numpy as np

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_affected_villages_vectorized(df, village_df, radius_km):
    # Placeholder for vectorized village distance logic
    # This identifies which villages are near specific sightings
    return df

def identify_risk_villages(df, village_df, p_dmg_rad, p_pres_rad, p_days):
    risk_list = []
    if village_df is None or df.empty:
        return risk_list

    # Example Logic: Identifying villages within the presence radius
    for _, v_row in village_df.iterrows():
        # Simple distance check for demonstration
        distances = haversine_vectorized(v_row['Latitude'], v_row['Longitude'], 
                                         df['Latitude'].values, df['Longitude'].values)
        
        if np.any(distances <= p_pres_rad):
            risk_list.append({
                'Village': v_row['Village'],
                'Lat': v_row['Latitude'],
                'Lon': v_row['Longitude'],
                'Radius': p_pres_rad * 1000, # In meters for Folium
                'Color': 'orange',
                'Reason': 'Recent Presence Detected'
            })
            
    return risk_list
