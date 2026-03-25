# config.py — single source of truth for all app constants

# ── App Settings ──────────────────────────────────────────────
APP_TITLE = "Elephant Sighting & Conflict Command Centre"
LAYOUT = "wide"
PAGE_ICON = "🐘"

# ── Logos ─────────────────────────────────────────────────────
LOGO_WCT = "https://www.wildlifeconservationtrust.org/wp-content/uploads/2016/09/wct-logo.png"
LOGO_MP  = "https://upload.wikimedia.org/wikipedia/commons/2/23/Emblem_of_Madhya_Pradesh.svg"

# ── Map Defaults ──────────────────────────────────────────────
DEFAULT_MAP_CENTER = [23.5, 80.5]   # Bandhavgarh region fallback
DEFAULT_MAP_ZOOM   = 9

# ── Risk Parameter Slider Bounds & Defaults ───────────────────
DAMAGE_RADIUS_MIN_KM     = 0.5
DAMAGE_RADIUS_MAX_KM     = 5.0
DAMAGE_RADIUS_DEFAULT_KM = 2.0
DAMAGE_RADIUS_STEP_KM    = 0.5

PRESENCE_RADIUS_MIN_KM     = 1.0
PRESENCE_RADIUS_MAX_KM     = 10.0
PRESENCE_RADIUS_DEFAULT_KM = 5.0
PRESENCE_RADIUS_STEP_KM    = 0.5

CONSECUTIVE_DAYS_MIN     = 1
CONSECUTIVE_DAYS_MAX     = 7
CONSECUTIVE_DAYS_DEFAULT = 3

# ── Spatial Analysis ──────────────────────────────────────────
AFFECTED_VILLAGE_RADIUS_KM = 2.0   # radius used in calculate_affected_villages
EARTH_RADIUS_KM            = 6371  # standard WGS-84 mean radius

# ── Severity Score Weights (single definition used everywhere) ─
SEVERITY_PRESENCE_WEIGHT = 1.0
SEVERITY_CROP_WEIGHT     = 2.5
SEVERITY_HOUSE_WEIGHT    = 5.0
SEVERITY_INJURY_WEIGHT   = 20.0

# ── Daytime Refuge Detection ──────────────────────────────────
REFUGE_DAY_START_HOUR = 9
REFUGE_DAY_END_HOUR   = 16
REFUGE_MAX_SEVERITY   = 1     # only low-conflict sightings qualify

# ── Spatial Refuge Inference from Conflict Proximity ─────────
# How far from a conflict site to search for likely refuge zones
REFUGE_CONFLICT_SEARCH_RADIUS_KM   = 8.0
# Grid cell size for the inference heatmap (smaller = finer, slower)
REFUGE_SPATIAL_GRID_RESOLUTION_KM  = 0.5

# ── KML/KMZ Parsing ──────────────────────────────────────────
KML_IGNORED_NAMES = {"BTR", "SATNA", "0", "1", "NONE", "UNKNOWN"}

# ── Herd Classification ──────────────────────────────────────
HERD_SPATIAL_GAP_KM      = 2.0   # max km between records to stay in same herd
HERD_TEMPORAL_GAP_HOURS  = 12    # max hours between records to stay in same herd
HERD_MIN_SIZE            = 1     # minimum Total Count to include a record

HERD_RISK_COLORS = {
    "Critical": "#E24B4A",
    "High":     "#EF9F27",
    "Moderate": "#378ADD",
    "Low":      "#639922",
}
HERD_COMPOSITION_COLORS = {
    "Bull Group":    "#1f77b4",
    "Nursery Herd":  "#e377c2",
    "Mixed Herd":    "#2ca02c",
    "Unclassified":  "#7f7f7f",
}

# ── Staff Analytics ───────────────────────────────────────────
MIN_STAFF_PER_DIVISION = 10   # divisions below this are flagged, not silently excluded

# ── Date Parsing ─────────────────────────────────────────────
DATE_FORMATS = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]

# ── Duplicate Detection Keys ──────────────────────────────────
DUPLICATE_SUBSET = ["Date", "Beat", "Created By", "Total Count", "Latitude", "Longitude"]
