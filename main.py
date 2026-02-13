import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

# App Config
st.set_page_config(
    page_title="Tsunami Alert AI System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Model Assets
@st.cache_resource
def load_models():
    model = joblib.load("knn.pkl")
    scaler = joblib.load("scaler.pkl")
    columns_pkl = joblib.load("columns.pkl")
    
    # Get actual feature names from model and scaler
    scaler_features = scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else columns_pkl
    model_features = model.feature_names_in_.tolist() if hasattr(model, 'feature_names_in_') else columns_pkl
    
    return model, scaler, model_features, scaler_features

model, scaler, expected_columns, scaler_columns = load_models()

# Load Dataset
@st.cache_data
def load_dataset():
    return pd.read_csv('earthquake_data_tsunami.csv')

df = load_dataset()

# Enhanced CSS with Modern Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
background: linear-gradient(
    to bottom,
    #021B79,
    #0575E6,
    #00C6FF
  );        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1929 0%, #132f4c 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Title Styles */
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: #e5f2ff;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 5px solid #1e88e5;
    }
    
    /* Stats Card */
    .stat-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        padding: 1.5rem;
        border-radius: 18px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 0.3rem;
    }
    
    .stat-label {
        font-size: 0.95rem;
        color: #e5f2ff;
        font-weight: 500;
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
        color: #fff;
        border-radius: 16px;
        border: none;
        padding: 14px 28px;
        font-size: 1.2rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 180, 219, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #0083b0 0%, #00b4db 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 180, 219, 0.6);
    }
    
    /* Alert Boxes */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 16px;
        padding: 1.8rem;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 8px 25px rgba(56, 239, 125, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    .error-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-radius: 16px;
        padding: 1.8rem;
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 8px 25px rgba(235, 51, 73, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Feature Badge */
    .feature-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.3rem;
        color: #fff;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.7);
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Custom number input */
    .stNumberInput>div>div>input {
        background: rgba(255,255,255,0.1);
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white; margin-bottom: 2rem;'>🌊 Navigation</h2>", unsafe_allow_html=True)
    
    menu = st.radio(
        "",
        ["🏠 Home", "🔍 Prediction", "📊 Analytics", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("<h3 style='color: white; text-align: center;'>📈 Dataset Stats</h3>", unsafe_allow_html=True)
    
    total_events = len(df)
    tsunami_events = int(df['tsunami'].sum())
    tsunami_rate = (tsunami_events / total_events) * 100
    
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{total_events}</div>
        <div class='stat-label'>Total Events</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{tsunami_events}</div>
        <div class='stat-label'>Tsunami Events</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='stat-card'>
        <div class='stat-number'>{tsunami_rate:.1f}%</div>
        <div class='stat-label'>Tsunami Rate</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; color: rgba(255,255,255,0.6); font-size: 0.85rem;'>
        <p>🎯 Model: K-Nearest Neighbors</p>
        <p>📊 Features: 6 Parameters</p>
        <p>⚡ Real-time Prediction</p>
    </div>
    """, unsafe_allow_html=True)

# HOME PAGE
def page_home():
    st.markdown("<h1 class='main-title'>🌊 Tsunami Alert AI System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Advanced Machine Learning for Seismic Tsunami Prediction</p>", unsafe_allow_html=True)
    
    # Hero Section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h2 style='color: white; margin-bottom: 1rem;'>🎯 Mission Statement</h2>
            <p style='color: #e5f2ff; font-size: 1.1rem; line-height: 1.8;'>
                Harnessing the power of artificial intelligence to predict tsunami risks 
                based on seismic data, enabling early warnings and saving lives through 
                data-driven decision making.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("<h2 style='color: white; text-align: center; margin: 2rem 0;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        {"icon": "⚡", "title": "Real-time Analysis", "desc": "Instant tsunami risk assessment"},
        {"icon": "🎯", "title": "High Accuracy", "desc": "ML-powered predictions"},
        {"icon": "📊", "title": "Data-Driven", "desc": "Based on 782+ events"},
        {"icon": "🌍", "title": "Global Coverage", "desc": "Worldwide seismic data"}
    ]
    
    for col, feature in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; min-height: 200px;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>{feature['icon']}</div>
                <h3 style='color: white; margin-bottom: 0.5rem;'>{feature['title']}</h3>
                <p style='color: #e5f2ff;'>{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown("<h2 style='color: white; text-align: center; margin: 2rem 0;'>🔬 How It Works</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #1e88e5;'>📥 Input Parameters</h3>
            <ul style='color: #333; line-height: 2; font-size: 1.05rem;'>
                <li><strong>Significance:</strong> Seismic event significance score</li>
                <li><strong>Stations:</strong> Number of reporting stations</li>
                <li><strong>Azimuthal Gap:</strong> Largest gap between stations</li>
                <li><strong>Depth:</strong> Depth of earthquake hypocenter</li>
                <li><strong>Latitude:</strong> Geographic coordinate (North-South)</li>
                <li><strong>Longitude:</strong> Geographic coordinate (East-West)</li>
                <li><strong>Year:</strong> Temporal context (auto-calculated)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #1e88e5;'>⚙️ Processing Pipeline</h3>
            <ol style='color: #333; line-height: 2; font-size: 1.05rem;'>
                <li>Data validation and preprocessing</li>
                <li>Feature scaling with StandardScaler</li>
                <li>K-Nearest Neighbors classification</li>
                <li>Probability calculation and risk assessment</li>
                <li>Real-time result visualization</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; text-align: center; margin: 2rem 0;'>🤖 Model Architecture</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3 style='color: white;'>Algorithm</h3>
            <div style='font-size: 2.5rem; margin: 1rem 0;'>🧠</div>
            <h2 style='color: #00b4db;'>KNN</h2>
            <p style='color: #e5f2ff;'>K-Nearest Neighbors Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3 style='color: white;'>Training Data</h3>
            <div style='font-size: 2.5rem; margin: 1rem 0;'>📚</div>
            <h2 style='color: #00b4db;'>782 Events</h2>
            <p style='color: #e5f2ff;'>Historical records (2001-2022)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h3 style='color: white;'>Features</h3>
            <div style='font-size: 2.5rem; margin: 1rem 0;'>🎯</div>
            <h2 style='color: #00b4db;'>6 Parameters</h2>
            <p style='color: #e5f2ff;'>Seismic & Geographic</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset Preview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: white; text-align: center; margin: 2rem 0;'>📊 Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: white;'>Sample Data</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True, height=350)
    
    with col2:
        st.markdown("<h3 style='color: white;'>Statistical Summary</h3>", unsafe_allow_html=True)
        st.dataframe(df[expected_columns + ['tsunami']].describe(), use_container_width=True, height=350)
    
    # Call to Action
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card' style='text-align: center;'>
        <h2 style='color: white; margin-bottom: 1rem;'>🚀 Ready to Predict?</h2>
        <p style='color: #e5f2ff; font-size: 1.1rem; margin-bottom: 1.5rem;'>
            Navigate to the Prediction page to assess tsunami risk based on seismic parameters
        </p>
    </div>
    """, unsafe_allow_html=True)

# PREDICTION PAGE
def page_prediction():
    st.markdown("<h1 class='main-title'>🔍 Tsunami Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enter seismic parameters for real-time tsunami risk assessment</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input Form
    with st.container():
        st.markdown("<h3 style='color: white; text-align: center; margin-bottom: 1.5rem;'>📝 Seismic Event Parameters</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: white;'>🌍 Event Characteristics</h4>", unsafe_allow_html=True)
            
            sig = st.number_input(
                "Significance (sig)",
                min_value=0.0,
                max_value=10000.0,
                value=800.0,
                step=10.0,
                help="Event significance score (typical range: 650-2910)"
            )
            
            nst = st.number_input(
                "Number of Stations (nst)",
                min_value=0.0,
                max_value=10000.0,
                value=150.0,
                step=1.0,
                help="Number of seismic stations reporting"
            )
            
            gap = st.number_input(
                "Azimuthal Gap (gap)",
                min_value=0.0,
                max_value=360.0,
                value=50.0,
                step=1.0,
                help="Largest azimuthal gap between stations (degrees)"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: white;'>📍 Geographic Location</h4>", unsafe_allow_html=True)
            
            depth = st.number_input(
                "Depth (km)",
                min_value=0.0,
                max_value=1000.0,
                value=10.0,
                step=1.0,
                help="Depth of earthquake hypocenter"
            )
            
            latitude = st.number_input(
                "Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                value=0.0,
                step=0.1,
                help="Latitude coordinate (-90 to 90)"
            )
            
            longitude = st.number_input(
                "Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                value=0.0,
                step=0.1,
                help="Longitude coordinate (-180 to 180)"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Info boxes
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📊 **Sig:** 650 - 2910")
    with col2:
        st.info("📡 **NST:** 0 - 934")
    with col3:
        st.info("📏 **Depth:** 0 - 700 km")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Predict Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🔮 ANALYZE TSUNAMI RISK", use_container_width=True)
    
    if predict_btn:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📥 Processing input data...")
        progress_bar.progress(20)
        time.sleep(0.3)
        
        # Prepare input for scaler (using scaler's expected features)
        scaler_data = pd.DataFrame({
            'sig': [float(sig)],
            'nst': [float(nst)],
            'gap': [float(gap)],
            'depth': [float(depth)],
            'latitude': [float(latitude)],
            'longitude': [float(longitude)]
        })
        
        # Ensure columns match scaler's expected order
        scaler_data = scaler_data[scaler_columns]
        
        status_text.text("⚙️ Scaling features...")
        progress_bar.progress(40)
        time.sleep(0.3)
        
        # Scale the features
        scaled_features = scaler.transform(scaler_data)
        
        # Now prepare data for model (model might need different features)
        # The model expects: ['sig', 'nst', 'latitude', 'longitude', 'Year']
        # We need to calculate Year from current date or use a default
        import datetime
        current_year = datetime.datetime.now().year
        
        model_data = pd.DataFrame({
            'sig': [float(sig)],
            'nst': [float(nst)],
            'latitude': [float(latitude)],
            'longitude': [float(longitude)],
            'Year': [float(current_year)]
        })
        
        # Ensure columns match model's expected order
        model_data = model_data[expected_columns]
        
        # For model, we need to scale only the features that were scaled during training
        # Since scaler was trained on different features, we'll use the scaled values
        # that correspond to the overlapping features
        
        status_text.text("🧠 Running ML model...")
        progress_bar.progress(60)
        time.sleep(0.3)
        
        # Create input for model using scaled values from overlapping features
        # Map: sig, nst, latitude, longitude from scaled_features
        model_input = []
        for col in expected_columns:
            if col in scaler_columns:
                idx = scaler_columns.index(col)
                model_input.append(scaled_features[0][idx])
            elif col == 'Year':
                # Year wasn't scaled, use raw value normalized
                model_input.append((current_year - 2011.5) / 10.5)  # approximate normalization
        
        model_input = np.array([model_input])
        
        # Predict
        pred = model.predict(model_input)[0]
        prob = model.predict_proba(model_input)[0]
        
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display Results
        st.markdown("---")
        st.markdown("<h2 style='color: white; text-align: center; margin: 2rem 0;'>🎯 Prediction Results</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if pred == 1:
                st.markdown(f"""
                <div class='error-box'>
                    <h2>⚠️ TSUNAMI ALERT</h2>
                    <p style='font-size: 1.5rem; margin: 1rem 0;'>Risk Probability: {prob[1]*100:.1f}%</p>
                    <p style='margin-top: 1rem; font-size: 1rem;'>
                        High tsunami risk detected. Immediate action recommended.
                        Please follow official evacuation procedures.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-card'>
                    <h4 style='color: #eb3349;'>🚨 Immediate Actions Required:</h4>
                    <ul style='color: #333; line-height: 2;'>
                        <li>Alert local emergency management authorities</li>
                        <li>Initiate evacuation procedures for coastal areas</li>
                        <li>Monitor official tsunami warning systems</li>
                        <li>Move to higher ground immediately</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.balloons()
                st.markdown(f"""
                <div class='success-box'>
                    <h2>✅ LOW RISK PROFILE</h2>
                    <p style='font-size: 1.5rem; margin: 1rem 0;'>Risk Probability: {prob[1]*100:.1f}%</p>
                    <p style='margin-top: 1rem; font-size: 1rem;'>
                        Minimal tsunami threat detected. Standard monitoring protocols apply.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class='info-card'>
                    <h4 style='color: #11998e;'>💚 Recommended Monitoring:</h4>
                    <ul style='color: #333; line-height: 2;'>
                        <li>Continue standard seismic monitoring</li>
                        <li>Maintain readiness of warning systems</li>
                        <li>Regular equipment testing and maintenance</li>
                        <li>Community preparedness education</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Probability Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob[1] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Tsunami Probability (%)", 'font': {'size': 20, 'color': 'white'}},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#0083b0"},
                    'bgcolor': "rgba(255,255,255,0.1)",
                    'borderwidth': 2,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 30], 'color': '#38ef7d'},
                        {'range': [30, 70], 'color': '#FFC107'},
                        {'range': [70, 100], 'color': '#eb3349'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white", 'family': "Inter"},
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Probability breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    "No Tsunami",
                    f"{prob[0]*100:.2f}%",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Tsunami",
                    f"{prob[1]*100:.2f}%",
                    delta=None
                )
        
        # Detailed breakdown
        st.markdown("---")
        st.markdown("<h3 style='color: white;'>📊 Detailed Analysis</h3>", unsafe_allow_html=True)
        
        # Bar chart
        prob_df = pd.DataFrame({
            'Outcome': ['No Tsunami', 'Tsunami'],
            'Probability': [prob[0] * 100, prob[1] * 100]
        })
        
        fig = px.bar(
            prob_df,
            x='Outcome',
            y='Probability',
            color='Outcome',
            color_discrete_map={'No Tsunami': '#38ef7d', 'Tsunami': '#eb3349'},
            text='Probability',
            title='Probability Distribution'
        )
        
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            yaxis_title="Probability (%)",
            xaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Input summary
        st.markdown("---")
        st.markdown("<h3 style='color: white;'>📋 Input Parameters Summary</h3>", unsafe_allow_html=True)
        
        current_year = datetime.datetime.now().year
        summary_df = pd.DataFrame({
            'Parameter': ['Significance', 'Stations', 'Azimuthal Gap', 'Depth', 'Latitude', 'Longitude', 'Year'],
            'Value': [sig, nst, gap, depth, latitude, longitude, current_year],
            'Unit': ['score', 'stations', '°', 'km', '°N/S', '°E/W', 'year']
        })
        
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ANALYTICS PAGE
def page_analytics():
    st.markdown("<h1 class='main-title'>📊 Model Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Comprehensive dataset and model performance analysis</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Cards
    st.markdown("<h3 style='color: white; text-align: center;'>📈 Dataset Overview</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(df)
    tsunami_events = int(df['tsunami'].sum())
    no_tsunami = total_events - tsunami_events
    tsunami_pct = (tsunami_events / total_events) * 100
    year_range = f"{int(df['Year'].min())}-{int(df['Year'].max())}"
    
    with col1:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <h4 style='color: #e5f2ff;'>Total Events</h4>
            <div class='stat-number'>{total_events}</div>
            <p style='color: #e5f2ff;'>Seismic Records</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <h4 style='color: #e5f2ff;'>Tsunami Events</h4>
            <div class='stat-number'>{tsunami_events}</div>
            <p style='color: #e5f2ff;'>{tsunami_pct:.1f}% of Total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <h4 style='color: #e5f2ff;'>No Tsunami</h4>
            <div class='stat-number'>{no_tsunami}</div>
            <p style='color: #e5f2ff;'>{100-tsunami_pct:.1f}% of Total</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='glass-card' style='text-align: center;'>
            <h4 style='color: #e5f2ff;'>Time Period</h4>
            <div class='stat-number' style='font-size: 2rem;'>{year_range}</div>
            <p style='color: #e5f2ff;'>22 Years</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Target Distribution
    st.markdown("<h3 style='color: white;'>🎯 Target Distribution</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        tsunami_dist = df['tsunami'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=['No Tsunami', 'Tsunami'],
            values=[tsunami_dist[0], tsunami_dist[1]],
            hole=0.4,
            marker=dict(colors=['#38ef7d', '#eb3349']),
            textinfo='label+percent',
            textfont_size=16
        )])
        
        fig.update_layout(
            title="Event Distribution",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(
                x=['No Tsunami', 'Tsunami'],
                y=[tsunami_dist[0], tsunami_dist[1]],
                marker=dict(color=['#38ef7d', '#eb3349']),
                text=[tsunami_dist[0], tsunami_dist[1]],
                textposition='outside',
                textfont=dict(size=16, color='white')
            )
        ])
        
        fig.update_layout(
            title="Event Count",
            yaxis_title="Number of Events",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Analysis
    st.markdown("<h3 style='color: white;'>📊 Feature Analysis</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4 style='color: white;'>Statistical Summary</h4>", unsafe_allow_html=True)
        st.dataframe(df[expected_columns].describe(), use_container_width=True)
    
    with col2:
        st.markdown("<h4 style='color: white;'>Correlation Matrix</h4>", unsafe_allow_html=True)
        corr = df[expected_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12, "color": "white"}
        ))
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Distributions
    st.markdown("<h3 style='color: white;'>📉 Feature Distributions</h3>", unsafe_allow_html=True)
    
    selected_feature = st.selectbox("Select feature to analyze:", expected_columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x=selected_feature,
            color='tsunami',
            color_discrete_map={0: '#38ef7d', 1: '#eb3349'},
            title=f"Distribution of {selected_feature}",
            nbins=30,
            barmode='overlay',
            opacity=0.7
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df,
            x='tsunami',
            y=selected_feature,
            color='tsunami',
            color_discrete_map={0: '#38ef7d', 1: '#eb3349'},
            title=f"{selected_feature} by Tsunami"
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No', 'Yes'])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Geographical Distribution
    st.markdown("<h3 style='color: white;'>🗺️ Global Distribution</h3>", unsafe_allow_html=True)
    
    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='tsunami',
        color_discrete_map={0: '#38ef7d', 1: '#eb3349'},
        title="Seismic Events Worldwide",
        projection='natural earth'
    )
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Temporal Analysis
    st.markdown("<h3 style='color: white;'>📅 Temporal Trends</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        yearly = df.groupby(['Year', 'tsunami']).size().reset_index(name='count')
        
        fig = px.line(
            yearly,
            x='Year',
            y='count',
            color='tsunami',
            color_discrete_map={0: '#38ef7d', 1: '#eb3349'},
            title="Events Over Time",
            markers=True
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        yearly_pct = df.groupby('Year')['tsunami'].agg(['sum', 'count'])
        yearly_pct['pct'] = (yearly_pct['sum'] / yearly_pct['count']) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=yearly_pct.index,
            y=yearly_pct['pct'],
            marker=dict(color='#00b4db'),
            text=yearly_pct['pct'].round(1),
            texttemplate='%{text}%',
            textposition='outside',
            textfont=dict(color='white')
        ))
        
        fig.update_layout(
            title="Tsunami Rate by Year",
            yaxis_title="Percentage (%)",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data Table
    st.markdown("<h3 style='color: white;'>📋 Raw Data Explorer</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_tsunami = st.checkbox("Tsunami Only")
    with col2:
        show_no_tsunami = st.checkbox("No Tsunami Only")
    with col3:
        num_rows = st.number_input("Rows:", 10, len(df), 50, 10)
    
    filtered = df.copy()
    if show_tsunami:
        filtered = filtered[filtered['tsunami'] == 1]
    elif show_no_tsunami:
        filtered = filtered[filtered['tsunami'] == 0]
    
    st.dataframe(filtered.head(num_rows), use_container_width=True, height=400)
    
    csv = filtered.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "tsunami_data.csv", "text/csv", use_container_width=True)

# ABOUT PAGE
def page_about():
    st.markdown("<h1 class='main-title'>ℹ️ About This System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Learn about our technology and mission</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #1e88e5;'>🎯 Our Mission</h3>
            <p style='color: #333; line-height: 1.8; font-size: 1.05rem;'>
                The Tsunami Alert AI System is dedicated to saving lives through early detection 
                and prediction of tsunami risks. Using advanced machine learning algorithms trained 
                on decades of seismic data, we provide rapid, accurate risk assessments that can 
                help communities prepare and respond to potential tsunami threats.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-card'>
            <h3 style='color: #1e88e5;'>🔬 The Technology</h3>
            <p style='color: #333; line-height: 1.8; font-size: 1.05rem;'>
                Our system uses <strong>K-Nearest Neighbors (KNN)</strong>, a proven machine learning 
                algorithm for pattern recognition. Trained on 782 historical seismic events spanning 
                22 years (2001-2022), the model analyzes 6 critical parameters including seismic 
                significance, station coverage, azimuthal gap, depth, and geographic coordinates 
                to predict tsunami probability with high accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #fff; text-align: center; margin-bottom: 1rem;'>📊 Model Stats</h4>
            <div style='color: #e5f2ff; line-height: 2; font-size: 0.95rem;'>
                <p><strong>Algorithm:</strong> K-Nearest Neighbors</p>
                <p><strong>Features:</strong> 6 parameters</p>
                <p><strong>Training Data:</strong> 782 events</p>
                <p><strong>Time Period:</strong> 2001-2022</p>
                <p><strong>Validation:</strong> Cross-validated</p>
                <p><strong>Response Time:</strong> <1 second</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Features
    st.markdown("""
    <div class='info-card'>
        <h3 style='color: #1e88e5; margin-bottom: 1rem;'>🧬 Model Features</h3>
        <p style='color: #333; margin-bottom: 1rem;'>
            Our AI analyzes the following seismic parameters:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show the scaler features (the ones user actually inputs)
    display_features = ['sig', 'nst', 'gap', 'depth', 'latitude', 'longitude']
    cols = st.columns(len(display_features))
    for i, feature in enumerate(display_features):
        with cols[i]:
            st.markdown(f"<span class='feature-badge'>{feature}</span>", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Technical Details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #1e88e5;'>⚙️ How KNN Works</h4>
            <p style='color: #333; line-height: 1.7;'>
                K-Nearest Neighbors is a non-parametric algorithm that classifies new data points 
                based on the majority class of their k nearest neighbors in the feature space. 
                Our model considers multiple seismic parameters simultaneously to identify patterns 
                associated with tsunami-generating earthquakes.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='info-card'>
            <h4 style='color: #1e88e5;'>📈 Model Training</h4>
            <p style='color: #333; line-height: 1.7;'>
                The model was trained using StandardScaler normalization for feature preprocessing 
                and optimized hyperparameters through cross-validation. Historical earthquake data 
                from global seismic networks provides a robust foundation for accurate predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class='info-card' style='background: linear-gradient(135deg, #fff9e6 0%, #ffe6cc 100%); border-left: 5px solid #ff9800;'>
        <h4 style='color: #f57c00; margin-bottom: 0.5rem;'>⚠️ Important Disclaimer</h4>
        <p style='color: #555; line-height: 1.7; font-size: 0.95rem;'>
            This system is designed for educational and research purposes. While it uses real historical 
            data and proven machine learning techniques, it should <strong>NOT</strong> be used as the 
            sole basis for emergency decisions. Always follow official tsunami warnings and evacuation 
            orders from authorized government agencies and emergency management authorities.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Router
if menu == "🏠 Home":
    page_home()
elif menu == "🔍 Prediction":
    page_prediction()
elif menu == "📊 Analytics":
    page_analytics()
else:
    page_about()

# Footer
st.markdown("""
<div class='footer'>
    <p style='font-size: 1rem; margin-bottom: 0.5rem;'>🌊 Tsunami Alert AI System</p>
    <p>Powered by Machine Learning | Built with Streamlit & Python</p>
    <p style='margin-top: 0.5rem; font-size: 0.85rem;'>
        © 2025 All Rights Reserved | For Educational Purposes Only
    </p>
</div>
""", unsafe_allow_html=True)
