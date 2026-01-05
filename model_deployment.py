# ============================================================================
# VOYAGE DELAY PREDICTION - STREAMLINED DASHBOARD
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="⚓ Voyage Delay Prediction",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean and minimal
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .big-metric {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-delayed {
        background: linear-gradient(135deg, #ff5252 0%, #f48fb1 100%);
        color: white;
    }
    .status-ontime {
        background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
        color: white;
    }
    .risk-card {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1e1e1e;
    }
    .risk-low {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
    }
    .risk-moderate {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
    }
    .risk-high {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
    }
    .ai-recommendation {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #5e72e4;
        margin: 15px 0;
        color: #1e1e1e;
    }
    .ai-recommendation h4 {
        color: #1e1e1e;
    }
    .insight-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        margin: 10px 0;
        color: #1e1e1e;
    }
    .insight-card h4 {
        color: #1e1e1e;
    }
    .insight-card p {
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURE GEMINI AI (FREE API)
# ============================================================================

# Initialize Gemini AI
# From: https://makersuite.google.com/app/apikey

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        AI_AVAILABLE = True
    except Exception as e:
        print(f"Error initializing Gemini AI: {e}")
        AI_AVAILABLE = False
        client = None
else:
    AI_AVAILABLE = False
    client = None

# ============================================================================
# MAJOR PORTS DATABASE
# ============================================================================

MAJOR_PORTS = {
    "Singapore": {"lat": 1.29, "lon": 103.85, "region": "Southeast Asia"},
    "Rotterdam": {"lat": 51.92, "lon": 4.48, "region": "Europe"},
    "Shanghai": {"lat": 31.23, "lon": 121.47, "region": "East Asia"},
    "Hong Kong": {"lat": 22.28, "lon": 114.17, "region": "East Asia"},
    "Los Angeles": {"lat": 33.74, "lon": -118.27, "region": "North America"},
    "New York": {"lat": 40.69, "lon": -74.04, "region": "North America"},
    "Hamburg": {"lat": 53.55, "lon": 9.99, "region": "Europe"},
    "Dubai": {"lat": 25.27, "lon": 55.29, "region": "Middle East"},
    "Tokyo": {"lat": 35.44, "lon": 139.64, "region": "East Asia"},
    "Sydney": {"lat": -33.87, "lon": 151.21, "region": "Oceania"},
    "Santos": {"lat": -23.96, "lon": -46.33, "region": "South America"},
    "Mumbai": {"lat": 19.07, "lon": 72.88, "region": "South Asia"},
    "Colombo": {"lat": 6.93, "lon": 79.85, "region": "South Asia"},
    "Suez": {"lat": 29.97, "lon": 32.55, "region": "Middle East"},
    "Panama City": {"lat": 8.97, "lon": -79.53, "region": "Central America"},
}

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource
def load_models():
    """Load trained models and preprocessing objects"""
    with open('model_rf_classification_tuned.pkl', 'rb') as f:
        clf_model = pickle.load(f)
    with open('model_rfr_regression_tuned.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('preprocessing_params.pkl', 'rb') as f:
        preprocessing_params = pickle.load(f)
    return clf_model, reg_model, scaler, preprocessing_params

@st.cache_data
def load_2024_data():
    """Load 2024 weather and news data"""
    weather_2024 = pd.read_csv('era5_weather_2024_sampled.csv', parse_dates=['time'])
    gdelt_2024 = pd.read_csv('gdelt_events_clean_2024.csv', parse_dates=['Date'])
    
    # Preprocessing
    weather_2024 = weather_2024[
        weather_2024['latitude'].between(-90, 90) &
        weather_2024['longitude'].between(-180, 180)
    ]
    weather_2024['temp_celsius'] = weather_2024['t2m'] - 273.15
    weather_2024['pressure_hpa'] = weather_2024['sp'] / 100
    weather_2024['year_month'] = weather_2024['time'].dt.to_period('M')
    weather_2024['lat_bin'] = (weather_2024['latitude'] // 1).astype(int)
    weather_2024['lon_bin'] = (weather_2024['longitude'] // 1).astype(int)
    
    gdelt_2024['Actor1CountryCode'] = gdelt_2024['Actor1CountryCode'].fillna('UNKNOWN')
    gdelt_2024 = gdelt_2024[
        gdelt_2024['GoldsteinScale'].between(-10, 10) &
        gdelt_2024['ActionGeo_Lat'].between(-90, 90) &
        gdelt_2024['ActionGeo_Long'].between(-180, 180)
    ]
    gdelt_2024['year_month'] = gdelt_2024['Date'].dt.to_period('M')
    gdelt_2024['lat_bin'] = (gdelt_2024['ActionGeo_Lat'] // 1).astype(int)
    gdelt_2024['lon_bin'] = (gdelt_2024['ActionGeo_Long'] // 1).astype(int)
    
    return weather_2024, gdelt_2024

# Load everything
try:
    clf_model, reg_model, scaler, preprocessing_params = load_models()
    weather_2024, gdelt_2024 = load_2024_data()
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def calculate_route_distance(start_lat, start_lon, end_lat, end_lon):
    """Calculate distance using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    lat1, lon1, lat2, lon2 = map(radians, [start_lat, start_lon, end_lat, end_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def estimate_voyage_duration(start_lat, start_lon, end_lat, end_lon):
    """Estimate voyage duration"""
    distance_km = calculate_route_distance(start_lat, start_lon, end_lat, end_lon)
    avg_speed_kmh = 37  # ~20 knots
    hours = distance_km / avg_speed_kmh
    return hours, hours / 24

def extract_weather_features(start_time, end_time, start_lat, start_lon, end_lat, end_lon):
    """Extract weather features along route"""
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    
    BUFFER = 5
    min_lat = min(start_lat, end_lat) - BUFFER
    max_lat = max(start_lat, end_lat) + BUFFER
    min_lon = min(start_lon, end_lon) - BUFFER
    max_lon = max(start_lon, end_lon) + BUFFER
    
    voyage_period = start_time.to_period('M')
    period_range = [voyage_period]
    if start_time.to_period('M') != end_time.to_period('M'):
        period_range.append(end_time.to_period('M'))
    
    weather_filtered = weather_2024[weather_2024['year_month'].isin(period_range)]
    
    lat_bins = range(int(min_lat) - 1, int(max_lat) + 2)
    lon_bins = range(int(min_lon) - 1, int(max_lon) + 2)
    
    if len(weather_filtered) > 0:
        weather_filtered = weather_filtered[
            weather_filtered['lat_bin'].isin(lat_bins) &
            weather_filtered['lon_bin'].isin(lon_bins)
        ]
        
        if len(weather_filtered) > 0:
            mask = (
                (weather_filtered['time'] >= start_time) & 
                (weather_filtered['time'] <= end_time) &
                (weather_filtered['latitude'].between(min_lat, max_lat)) &
                (weather_filtered['longitude'].between(min_lon, max_lon))
            )
            voyage_weather = weather_filtered[mask]
        else:
            voyage_weather = pd.DataFrame()
    else:
        voyage_weather = pd.DataFrame()
    
    if len(voyage_weather) > 0:
        return {
            'avg_wind_speed': voyage_weather['wind_speed'].mean(),
            'max_wind_speed': voyage_weather['wind_speed'].max(),
            'min_wind_speed': voyage_weather['wind_speed'].min(),
            'avg_temp_celsius': voyage_weather['temp_celsius'].mean(),
            'max_temp_celsius': voyage_weather['temp_celsius'].max(),
            'min_temp_celsius': voyage_weather['temp_celsius'].min(),
            'avg_pressure_hpa': voyage_weather['pressure_hpa'].mean(),
            'total_precipitation': voyage_weather['tp'].sum(),
            'weather_records': len(voyage_weather)
        }
    else:
        return {
            'avg_wind_speed': 0, 'max_wind_speed': 0, 'min_wind_speed': 0,
            'avg_temp_celsius': 0, 'max_temp_celsius': 0, 'min_temp_celsius': 0,
            'avg_pressure_hpa': 0, 'total_precipitation': 0, 'weather_records': 0
        }

def extract_news_features(start_time, end_time, start_lat, start_lon, end_lat, end_lon):
    """Extract geopolitical features"""
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)
    
    BUFFER = 10
    min_lat = min(start_lat, end_lat) - BUFFER
    max_lat = max(start_lat, end_lat) + BUFFER
    min_lon = min(start_lon, end_lon) - BUFFER
    max_lon = max(start_lon, end_lon) + BUFFER
    
    voyage_period = start_time.to_period('M')
    period_range = [voyage_period]
    if start_time.to_period('M') != end_time.to_period('M'):
        period_range.append(end_time.to_period('M'))
    
    gdelt_filtered = gdelt_2024[gdelt_2024['year_month'].isin(period_range)]
    
    lat_bins = range(int(min_lat) - 1, int(max_lat) + 2)
    lon_bins = range(int(min_lon) - 1, int(max_lon) + 2)
    
    if len(gdelt_filtered) > 0:
        gdelt_filtered = gdelt_filtered[
            gdelt_filtered['lat_bin'].isin(lat_bins) &
            gdelt_filtered['lon_bin'].isin(lon_bins)
        ]
        
        if len(gdelt_filtered) > 0:
            mask = (
                (gdelt_filtered['Date'] >= start_time) & 
                (gdelt_filtered['Date'] <= end_time) &
                (gdelt_filtered['ActionGeo_Lat'].between(min_lat, max_lat)) &
                (gdelt_filtered['ActionGeo_Long'].between(min_lon, max_lon))
            )
            voyage_news = gdelt_filtered[mask]
        else:
            voyage_news = pd.DataFrame()
    else:
        voyage_news = pd.DataFrame()
    
    if len(voyage_news) > 0:
        return {
            'num_events': len(voyage_news),
            'avg_goldstein': voyage_news['GoldsteinScale'].mean(),
            'min_goldstein': voyage_news['GoldsteinScale'].min(),
            'max_goldstein': voyage_news['GoldsteinScale'].max(),
            'avg_tone': voyage_news['AvgTone'].mean(),
            'total_mentions': voyage_news['NumMentions'].sum(),
            'total_sources': voyage_news['NumSources'].sum(),
            'negative_events': (voyage_news['GoldsteinScale'] < 0).sum(),
            'positive_events': (voyage_news['GoldsteinScale'] > 0).sum()
        }
    else:
        return {
            'num_events': 0, 'avg_goldstein': 0, 'min_goldstein': 0, 
            'max_goldstein': 0, 'avg_tone': 0, 'total_mentions': 0, 
            'total_sources': 0, 'negative_events': 0, 'positive_events': 0
        }

def prepare_features_for_prediction(voyage_info):
    """Prepare feature vector for model prediction"""
    
    weather_features = extract_weather_features(
        voyage_info['start_time'],
        voyage_info['estimated_end_time'],
        voyage_info['start_lat'],
        voyage_info['start_lon'],
        voyage_info['end_lat'],
        voyage_info['end_lon']
    )
    
    news_features = extract_news_features(
        voyage_info['start_time'],
        voyage_info['estimated_end_time'],
        voyage_info['start_lat'],
        voyage_info['start_lon'],
        voyage_info['end_lat'],
        voyage_info['end_lon']
    )
    
    start_lat_region = int(voyage_info['start_lat'] // 10)
    start_lon_region = int(voyage_info['start_lon'] // 10)
    end_lat_region = int(voyage_info['end_lat'] // 10)
    end_lon_region = int(voyage_info['end_lon'] // 10)
    
    start_time = pd.Timestamp(voyage_info['start_time'])
    start_month = start_time.month
    start_dow = start_time.dayofweek
    start_hour = start_time.hour
    start_quarter = start_time.quarter
    start_year = start_time.year
    
    high_wind_flag = int(weather_features['max_wind_speed'] > 15)
    heavy_precip_flag = int(weather_features['total_precipitation'] > 0.01)
    has_negative_news = int(news_features['negative_events'] > 0)
    negative_news_ratio = news_features['negative_events'] / (news_features['num_events'] + 1)
    
    features = {
        'start_lat_region': start_lat_region,
        'start_lon_region': start_lon_region,
        'end_lat_region': end_lat_region,
        'end_lon_region': end_lon_region,
        'avg_wind_speed': weather_features['avg_wind_speed'],
        'max_wind_speed': weather_features['max_wind_speed'],
        'min_wind_speed': weather_features['min_wind_speed'],
        'avg_temp_celsius': weather_features['avg_temp_celsius'],
        'max_temp_celsius': weather_features['max_temp_celsius'],
        'min_temp_celsius': weather_features['min_temp_celsius'],
        'avg_pressure_hpa': weather_features['avg_pressure_hpa'],
        'total_precipitation': weather_features['total_precipitation'],
        'weather_records': weather_features['weather_records'],
        'num_events': news_features['num_events'],
        'avg_goldstein': news_features['avg_goldstein'],
        'min_goldstein': news_features['min_goldstein'],
        'max_goldstein': news_features['max_goldstein'],
        'avg_tone': news_features['avg_tone'],
        'total_mentions': news_features['total_mentions'],
        'total_sources': news_features['total_sources'],
        'negative_events': news_features['negative_events'],
        'positive_events': news_features['positive_events'],
        'StartMonth': start_month,
        'StartDayOfWeek': start_dow,
        'StartHour': start_hour,
        'StartQuarter': start_quarter,
        'StartYear': start_year,
        'HighWindFlag': high_wind_flag,
        'HeavyPrecipitationFlag': heavy_precip_flag,
        'HasNegativeNews': has_negative_news,
        'NegativeNewsRatio': negative_news_ratio,
    }
    
    return features, weather_features, news_features

# ============================================================================
# RISK SCORING & ANALYSIS
# ============================================================================

def calculate_risk_score(weather_features, news_features, regression_pred):
    """Calculate simplified risk score"""
    
    # Weather risk (0-5)
    weather_risk = 0
    if weather_features['max_wind_speed'] > 20:
        weather_risk += 3
    elif weather_features['max_wind_speed'] > 15:
        weather_risk += 2
    elif weather_features['max_wind_speed'] > 10:
        weather_risk += 1
    
    if weather_features['total_precipitation'] > 0.01:
        weather_risk += 1
    
    if weather_features['max_temp_celsius'] > 35 or weather_features['min_temp_celsius'] < 0:
        weather_risk += 1
    
    weather_risk = min(5, weather_risk)
    
    # Geopolitical risk (0-5)
    geo_risk = 0
    if news_features['avg_goldstein'] < -5:
        geo_risk += 4
    elif news_features['avg_goldstein'] < -2:
        geo_risk += 2
    elif news_features['avg_goldstein'] < 0:
        geo_risk += 1
    
    if news_features['negative_events'] > 10:
        geo_risk += 1
    
    geo_risk = min(5, geo_risk)
    
    # Total risk (0-10) with adjustments
    base_risk = weather_risk + geo_risk
    
    if regression_pred < -24:  # Early arrival
        total_risk = max(0, base_risk - 2)
    elif regression_pred < 0:
        total_risk = max(0, base_risk - 1)
    else:
        total_risk = base_risk
    
    return {
        'weather_risk': round(weather_risk, 1),
        'geo_risk': round(geo_risk, 1),
        'total_risk': round(total_risk, 1),
        'base_risk': round(base_risk, 1)
    }

# ============================================================================
# AI RECOMMENDATION SYSTEM (GEMINI)
# ============================================================================

def generate_ai_mitigation_weather(voyage_info, weather_features, risk_scores, delay_prediction):
    """Generate AI-powered weather mitigation recommendations"""
    
    if not AI_AVAILABLE:
        return "⚠️ AI recommendations unavailable. Please configure GEMINI_API_KEY variable."
    
    prompt = f"""
You are a maritime operations expert providing concise mitigation recommendations.

VOYAGE DETAILS:
- Route: {voyage_info['start_port']} → {voyage_info['end_port']}
- Distance: {voyage_info['distance_km']:.0f} km
- Duration: {voyage_info['estimated_duration_hours']/24:.1f} days
- Departure: {voyage_info['start_time'].strftime('%Y-%m-%d %H:%M')}

PREDICTION:
- Delay Status: {"DELAYED" if delay_prediction['is_delayed'] else "ON-TIME"}
- Expected Delay: {delay_prediction['delay_hours']:.1f} hours

WEATHER CONDITIONS:
- Max Wind Speed: {weather_features['max_wind_speed']:.1f} m/s
- Avg Temperature: {weather_features['avg_temp_celsius']:.1f}°C
- Total Precipitation: {weather_features['total_precipitation']:.4f} m
- Weather Risk Score: {risk_scores['weather_risk']}/5

TASK:
Provide 3-4 SPECIFIC, ACTIONABLE mitigation strategies for weather-related risks.
Each recommendation should be:
1. Practical and implementable
2. Specific to the conditions above
3. Brief (1-2 sentences max)

Format as numbered list. Keep response under 150 words total.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI service temporarily unavailable: {str(e)}"

def generate_ai_mitigation_geopolitical(voyage_info, news_features, risk_scores, delay_prediction):
    """Generate AI-powered geopolitical mitigation recommendations"""
    
    if not AI_AVAILABLE:
        return "⚠️ AI recommendations unavailable. Please configure GEMINI_API_KEY variable."
    
    prompt = f"""
You are a maritime security expert providing concise mitigation recommendations.

VOYAGE DETAILS:
- Route: {voyage_info['start_port']} → {voyage_info['end_port']}
- Distance: {voyage_info['distance_km']:.0f} km
- Duration: {voyage_info['estimated_duration_hours']/24:.1f} days

PREDICTION:
- Delay Status: {"DELAYED" if delay_prediction['is_delayed'] else "ON-TIME"}
- Expected Delay: {delay_prediction['delay_hours']:.1f} hours

GEOPOLITICAL CONDITIONS:
- Total Events: {news_features['num_events']}
- Negative Events: {news_features['negative_events']}
- Avg Goldstein Score: {news_features['avg_goldstein']:.2f} (scale: -10 to +10)
- Avg Tone: {news_features['avg_tone']:.2f}
- Geopolitical Risk Score: {risk_scores['geo_risk']}/5

TASK:
Provide 3-4 SPECIFIC, ACTIONABLE mitigation strategies for geopolitical risks.
Each recommendation should be:
1. Security-focused and practical
2. Specific to the risk level above
3. Brief (1-2 sentences max)

Format as numbered list. Keep response under 150 words total.
"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI service temporarily unavailable: {str(e)}"

# ============================================================================
# VISUALIZATION FUNCTIONS (SIMPLIFIED)
# ============================================================================

def create_risk_gauge(risk_score, title):
    """Create clean risk gauge"""
    if risk_score < 3:
        color = "#4caf50"
        status = "LOW"
    elif risk_score < 5:
        color = "#ff9800"
        status = "MODERATE"
    else:
        color = "#f44336"
        status = "HIGH"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        number={'suffix': "/10", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 10], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': '#e8f5e9'},
                {'range': [3, 5], 'color': '#fff3e0'},
                {'range': [5, 10], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def create_weather_chart(weather_features):
    """Create simplified weather overview"""
    categories = ['Wind Speed', 'Temperature', 'Precipitation']
    
    # Normalize to 0-100 scale for comparison
    wind_normalized = min(100, (weather_features['max_wind_speed'] / 30) * 100)
    temp_normalized = min(100, ((weather_features['avg_temp_celsius'] + 10) / 55) * 100)
    precip_normalized = min(100, (weather_features['total_precipitation'] / 0.05) * 100)
    
    values = [wind_normalized, temp_normalized, precip_normalized]
    
    colors = []
    for v in values:
        if v > 70:
            colors.append('#f44336')
        elif v > 40:
            colors.append('#ff9800')
        else:
            colors.append('#4caf50')
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{weather_features['max_wind_speed']:.1f} m/s",
                  f"{weather_features['avg_temp_celsius']:.1f}°C",
                  f"{weather_features['total_precipitation']:.4f} m"],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Weather Conditions Overview",
        yaxis_title="Severity Index",
        yaxis_range=[0, 120],
        height=350,
        showlegend=False
    )
    
    return fig

def create_geopolitical_chart(news_features):
    """Create simplified geopolitical overview"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Positive Events',
        x=['Events'],
        y=[news_features['positive_events']],
        marker_color='#4caf50',
        text=[news_features['positive_events']],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Negative Events',
        x=['Events'],
        y=[news_features['negative_events']],
        marker_color='#f44336',
        text=[news_features['negative_events']],
        textposition='inside'
    ))
    
    fig.update_layout(
        title="Geopolitical Event Balance",
        yaxis_title="Number of Events",
        barmode='stack',
        height=350,
        showlegend=True
    )
    
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state['prediction_made'] = False
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    
    # Header
    st.title("⚓ Voyage Delay Prediction System")
    
    # ========================================================================
    # SIDEBAR - USER INPUT
    # ========================================================================
    
    with st.sidebar:
        st.header("📋 Voyage Input")
        
        # Route Selection
        st.subheader("📍 Route")
        route_method = st.radio(
            "Input Method:",
            ["Major Ports", "Custom Coordinates"],
            label_visibility="collapsed"
        )
        
        if route_method == "Major Ports":
            start_port = st.selectbox("Departure Port", list(MAJOR_PORTS.keys()), key="start")
            end_port = st.selectbox("Destination Port", list(MAJOR_PORTS.keys()), key="end")
            
            start_lat = MAJOR_PORTS[start_port]["lat"]
            start_lon = MAJOR_PORTS[start_port]["lon"]
            end_lat = MAJOR_PORTS[end_port]["lat"]
            end_lon = MAJOR_PORTS[end_port]["lon"]
            
        else:
            st.markdown("**Departure**")
            col1, col2 = st.columns(2)
            start_lat = col1.number_input("Lat", -90.0, 90.0, 1.29, key="slat")
            start_lon = col2.number_input("Lon", -180.0, 180.0, 103.85, key="slon")
            
            st.markdown("**Destination**")
            col1, col2 = st.columns(2)
            end_lat = col1.number_input("Lat", -90.0, 90.0, 51.92, key="elat")
            end_lon = col2.number_input("Lon", -180.0, 180.0, 4.48, key="elon")
            
            start_port = f"({start_lat:.1f}, {start_lon:.1f})"
            end_port = f"({end_lat:.1f}, {end_lon:.1f})"
        
        # Calculate route info
        est_hours, est_days = estimate_voyage_duration(start_lat, start_lon, end_lat, end_lon)
        distance_km = calculate_route_distance(start_lat, start_lon, end_lat, end_lon)
        
        st.info(f"📏 {distance_km:,.0f} km\n⏱️ ~{est_days:.1f} days")
        
        # Timing
        st.subheader("🕐 Departure")
        start_date = st.date_input(
            "Date",
            value=datetime(2024, 6, 1),
            min_value=datetime(2024, 1, 1),
            max_value=datetime(2024, 12, 31)
        )
        start_time_input = st.time_input("Time", value=datetime.strptime("08:00", "%H:%M").time())
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🚀 Predict", use_container_width=True, type="primary")
        
        # Reset button
        if st.session_state.prediction_made:
            if st.button("🔄 New", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.results = None
                st.rerun()
        
        # AI Status
        st.markdown("---")
        if AI_AVAILABLE:
            st.success("🤖 AI Assistant: Active")
        else:
            st.warning("🤖 AI Assistant: Disabled\n\nSet GEMINI_API_KEY to enable AI recommendations.")
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    if predict_button or st.session_state.prediction_made:
        
        if predict_button:
            # Prepare voyage information
            start_datetime = datetime.combine(start_date, start_time_input)
            estimated_end_datetime = start_datetime + timedelta(hours=est_hours)
            
            voyage_info = {
                'start_port': start_port,
                'end_port': end_port,
                'start_lat': start_lat,
                'start_lon': start_lon,
                'end_lat': end_lat,
                'end_lon': end_lon,
                'start_time': start_datetime,
                'estimated_end_time': estimated_end_datetime,
                'distance_km': distance_km,
                'estimated_duration_hours': est_hours
            }
            
            with st.spinner("🔄 Analyzing voyage conditions..."):
                
                try:
                    # Extract features
                    features, weather_features, news_features = prepare_features_for_prediction(voyage_info)
                    
                    # Prepare for model
                    feature_list = [
                        features['start_lat_region'], features['start_lon_region'],
                        features['end_lat_region'], features['end_lon_region'],
                        features['avg_wind_speed'], features['max_wind_speed'], features['min_wind_speed'],
                        features['avg_temp_celsius'], features['max_temp_celsius'], features['min_temp_celsius'],
                        features['avg_pressure_hpa'], features['total_precipitation'], features['weather_records'],
                        features['num_events'], features['avg_goldstein'], features['min_goldstein'],
                        features['max_goldstein'], features['avg_tone'], features['total_mentions'],
                        features['total_sources'], features['negative_events'], features['positive_events'],
                        features['StartMonth'], features['StartDayOfWeek'], features['StartHour'],
                        features['StartQuarter'], features['StartYear'],
                        features['HighWindFlag'], features['HeavyPrecipitationFlag'],
                        features['HasNegativeNews'], features['NegativeNewsRatio']
                    ]
                    
                    X_input = np.array(feature_list).reshape(1, -1)
                    X_scaled = scaler.transform(X_input)
                    
                    # Model predictions
                    classification_pred = clf_model.predict(X_scaled)[0]
                    classification_proba = clf_model.predict_proba(X_scaled)[0][1]
                    regression_pred = reg_model.predict(X_scaled)[0]
                    
                    # Calculate risk scores
                    risk_scores = calculate_risk_score(weather_features, news_features, regression_pred)
                    
                    # Store results
                    st.session_state.results = {
                        'voyage_info': voyage_info,
                        'weather_features': weather_features,
                        'news_features': news_features,
                        'classification_pred': classification_pred,
                        'classification_proba': classification_proba,
                        'regression_pred': regression_pred,
                        'risk_scores': risk_scores
                    }
                    st.session_state.prediction_made = True
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.stop()
        
        # Retrieve results
        results = st.session_state.results
        voyage_info = results['voyage_info']
        weather_features = results['weather_features']
        news_features = results['news_features']
        classification_pred = results['classification_pred']
        classification_proba = results['classification_proba']
        regression_pred = results['regression_pred']
        risk_scores = results['risk_scores']
        
        # Prepare delay prediction dict for AI
        delay_prediction = {
            'is_delayed': classification_pred == 1,
            'confidence': classification_proba * 100,
            'delay_hours': regression_pred
        }
        
        # ====================================================================
        # MAIN CONTENT: 3 KEY CARDS
        # ====================================================================
        
        col1, col2, col3 = st.columns(3)
        
        # CARD 1: Status
        with col1:
            if classification_pred == 1:
                st.markdown(f"""
                    <div class="big-metric status-delayed">
                        <h2>🔴 DELAYED</h2>
                        <p style="font-size:24px; margin:10px 0;">{regression_pred:.1f} hours</p>
                        <p style="font-size:16px; opacity:0.9;">{classification_proba*100:.0f}% confidence</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                if regression_pred < 0:
                    st.markdown(f"""
                        <div class="big-metric status-ontime">
                            <h2>🟢 EARLY</h2>
                            <p style="font-size:24px; margin:10px 0;">{abs(regression_pred):.1f} hours</p>
                            <p style="font-size:16px; opacity:0.9;">{classification_proba*100:.0f}% confidence</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="big-metric status-ontime">
                            <h2>🟢 ON-TIME</h2>
                            <p style="font-size:24px; margin:10px 0;">±{abs(regression_pred):.1f} hours</p>
                            <p style="font-size:16px; opacity:0.9;">{classification_proba*100:.0f}% confidence</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # CARD 2: Weather Risk
        with col2:
            risk_class = "risk-low" if risk_scores['weather_risk'] < 2 else ("risk-moderate" if risk_scores['weather_risk'] < 4 else "risk-high")
            st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <h3>⛈️ Weather Risk</h3>
                    <h1 style="margin:10px 0; font-size:48px;">{risk_scores['weather_risk']}<span style="font-size:24px;">/5</span></h1>
                    <p>Max Wind: {weather_features['max_wind_speed']:.1f} m/s</p>
                </div>
            """, unsafe_allow_html=True)
        
        # CARD 3: Geopolitical Risk
        with col3:
            risk_class = "risk-low" if risk_scores['geo_risk'] < 2 else ("risk-moderate" if risk_scores['geo_risk'] < 4 else "risk-high")
            st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <h3>🌍 Geopolitical Risk</h3>
                    <h1 style="margin:10px 0; font-size:48px;">{risk_scores['geo_risk']}<span style="font-size:24px;">/5</span></h1>
                    <p>{news_features['negative_events']} negative events</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ====================================================================
        # TABS: Summary | Weather Insights | Geopolitical Insights
        # ====================================================================
        
        tab1, tab2, tab3 = st.tabs([
            "📊 Summary",
            "⛈️ Weather Insights",
            "🌍 Geopolitical Insights"
        ])
        
        # ===== TAB 1: SUMMARY =====
        with tab1:
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(
                    create_risk_gauge(risk_scores['total_risk'], "Overall Risk Score"),
                    width="stretch"
                )
            
            with col2:
                st.markdown("### 🎯 Key Findings")
                
                st.markdown(f"""
                <div class="insight-card">
                    <h4>📍 Route Information</h4>
                    <p><strong>{voyage_info['start_port']}</strong> → <strong>{voyage_info['end_port']}</strong></p>
                    <p>Distance: {voyage_info['distance_km']:,.0f} km | Duration: ~{voyage_info['estimated_duration_hours']/24:.1f} days</p>
                </div>
                """, unsafe_allow_html=True)
                
                if classification_pred == 1:
                    st.markdown(f"""
                    <div class="insight-card" style="border-left-color: #f44336;">
                        <h4>⚠️ Delay Expected</h4>
                        <p>Model predicts <strong>{regression_pred:.1f} hours</strong> delay with <strong>{classification_proba*100:.0f}%</strong> confidence.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if regression_pred < 0:
                        st.markdown(f"""
                        <div class="insight-card" style="border-left-color: #4caf50;">
                            <h4>✅ Early Arrival Expected</h4>
                            <p>Model predicts arrival <strong>{abs(regression_pred):.1f} hours early</strong> with <strong>{classification_proba*100:.0f}%</strong> confidence.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="insight-card" style="border-left-color: #4caf50;">
                            <h4>✅ On-Time Arrival Expected</h4>
                            <p>Model predicts on-time arrival (±{abs(regression_pred):.1f} hours) with <strong>{classification_proba*100:.0f}%</strong> confidence.</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            st.markdown("### 📋 Risk Breakdown")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Weather Risk",
                    f"{risk_scores['weather_risk']}/5",
                    delta="Severe" if risk_scores['weather_risk'] >= 4 else ("Moderate" if risk_scores['weather_risk'] >= 2 else "Low"),
                    delta_color="inverse" if risk_scores['weather_risk'] >=4 else ("off" if risk_scores['weather_risk'] >=2 else "normal")
                )
            
            with col2:
                st.metric(
                    "Geopolitical Risk",
                    f"{risk_scores['geo_risk']}/5",
                    delta="Severe" if risk_scores['geo_risk'] >= 4 else ("Moderate" if risk_scores['geo_risk'] >= 2 else "Low"),
                    delta_color="inverse" if risk_scores['geo_risk'] >=4 else ("off" if risk_scores['geo_risk'] >=2 else "normal")
                )
            
            with col3:
                st.metric(
                    "Total Risk",
                    f"{risk_scores['total_risk']}/10",
                    delta="Critical" if risk_scores['total_risk'] >= 7 else ("High" if risk_scores['total_risk'] >= 5 else ("Moderate" if risk_scores['total_risk'] >= 3 else "Low")),
                    delta_color="inverse" if risk_scores['total_risk'] >=5 else ("off" if risk_scores['total_risk'] >=3 else "normal")
                )
            
            # Decision recommendation
            st.markdown("---")
            st.markdown("### 💡 Recommendation")
            
            if risk_scores['total_risk'] >= 7:
                st.error("🚨 **CRITICAL RISK** - Strongly recommend postponing or rerouting this voyage.")
            elif risk_scores['total_risk'] >= 5:
                st.warning("⚠️ **HIGH RISK** - Proceed with extreme caution. Consider alternative routes or timing.")
            elif risk_scores['total_risk'] >= 3:
                st.info("ℹ️ **MODERATE RISK** - Voyage can proceed with enhanced monitoring and contingency plans.")
            else:
                st.success("✅ **LOW RISK** - Favorable conditions. Voyage can proceed as planned.")
        
        # ===== TAB 2: WEATHER INSIGHTS =====
        with tab2:
            
            st.markdown("### ⛈️ Weather Conditions")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(
                    create_weather_chart(weather_features),
                    width="stretch"
                )
            
            with col2:
                st.markdown("#### 📊 Weather Metrics")
                
                st.metric("Max Wind Speed", f"{weather_features['max_wind_speed']:.1f} m/s")
                st.metric("Avg Temperature", f"{weather_features['avg_temp_celsius']:.1f}°C")
                st.metric("Precipitation", f"{weather_features['total_precipitation']:.4f} m")
                
                if weather_features['max_wind_speed'] > 20:
                    st.error("⚠️ Severe wind conditions")
                elif weather_features['max_wind_speed'] > 15:
                    st.warning("⚠️ Moderate wind conditions")
                else:
                    st.success("✅ Favorable wind conditions")
            
            st.markdown("---")
            
            # AI Mitigation Recommendations
            st.markdown("### 🤖 AI-Powered Mitigation Strategies")
            
            with st.spinner("Generating AI recommendations..."):
                ai_recommendations = generate_ai_mitigation_weather(
                    voyage_info,
                    weather_features,
                    risk_scores,
                    delay_prediction
                )
            
            st.markdown(f"""
                <div class="ai-recommendation">
                    <h4>🤖 Weather Risk Mitigation</h4>
                    {ai_recommendations}
                </div>
            """, unsafe_allow_html=True)
        
        # ===== TAB 3: GEOPOLITICAL INSIGHTS =====
        with tab3:
            
            st.markdown("### 🌍 Geopolitical Conditions")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.plotly_chart(
                    create_geopolitical_chart(news_features),
                    width="stretch"
                )
            
            with col2:
                st.markdown("#### 📊 Event Metrics")
                
                st.metric("Total Events", news_features['num_events'])
                st.metric("Negative Events", news_features['negative_events'])
                st.metric("Goldstein Score", f"{news_features['avg_goldstein']:.2f}")
                
                if news_features['avg_goldstein'] < -5:
                    st.error("⚠️ Severe instability detected")
                elif news_features['avg_goldstein'] < -2:
                    st.warning("⚠️ Elevated tensions")
                elif news_features['avg_goldstein'] < 0:
                    st.info("ℹ️ Minor concerns")
                else:
                    st.success("✅ Stable environment")
            
            st.markdown("---")
            
            # AI Mitigation Recommendations
            st.markdown("### 🤖 AI-Powered Mitigation Strategies")
            
            with st.spinner("Generating AI recommendations..."):
                ai_recommendations = generate_ai_mitigation_geopolitical(
                    voyage_info,
                    news_features,
                    risk_scores,
                    delay_prediction
                )
            
            st.markdown(f"""
                <div class="ai-recommendation">
                    <h4>🤖 Geopolitical Risk Mitigation</h4>
                    {ai_recommendations}
                </div>
            """, unsafe_allow_html=True)
    
    else:
        # Welcome screen
        st.info("👈 Enter voyage details in the sidebar and click **Predict** to begin analysis.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Accuracy", "~94%")
        col2.metric("Training Data", "2020-2023")
        col3.metric("Analysis Year", "2024")
        
        st.markdown("---")
        
        st.markdown("""
        ### 🎯 How It Works
        
        1. **Enter Route** - Select ports or enter coordinates
        2. **Set Departure** - Choose date and time
        3. **Get Prediction** - Receive AI-powered analysis
        4. **Review Insights** - Weather and geopolitical risk assessments
        5. **Act on Recommendations** - AI-generated mitigation strategies
        
        ### 🤖 AI Assistant
        
        Our integrated Gemini AI provides context-aware, actionable mitigation recommendations based on:
        - Your specific route and timing
        - Current weather conditions
        - Geopolitical risk factors
        - Model predictions
        """)

if __name__ == "__main__":
    main()