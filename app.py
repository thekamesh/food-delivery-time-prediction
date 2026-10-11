
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path

# ============================================================
# Page
# ============================================================
st.set_page_config(
    page_title="SwiftDrop • Delivery ETA",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LIGHT THEME STYLING
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 5%, rgba(99,102,241,.08), transparent 28%),
        radial-gradient(circle at 90% 10%, rgba(20,184,166,.07), transparent 25%),
        #f6f8fc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ================= HERO ================= */

.hero {
    padding: 2.2rem 2.4rem;
    border: 1px solid #e5e7eb;
    border-radius: 26px;
    background: linear-gradient(135deg, #ffffff, #f8faff);
    box-shadow: 0 15px 45px rgba(15, 23, 42, .08);
    margin-bottom: 1.4rem;
    animation: fadeUp .65s ease-out;
}

.hero-badge {
    display: inline-block;
    padding: .38rem .72rem;
    border-radius: 999px;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    color: #4f46e5;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .04em;
    margin-bottom: .8rem;
}

.hero h1 {
    margin: 0;
    font-size: clamp(2rem, 5vw, 3.6rem);
    line-height: 1.03;
    letter-spacing: -.055em;
    color: #111827;
}

.hero p {
    color: #64748b;
    font-size: 1rem;
    margin: .9rem 0 0;
    max-width: 720px;
}

/* ================= CARDS ================= */

.card {
    padding: 1.25rem 1.35rem;
    border-radius: 20px;
    background: rgba(255,255,255,.95);
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 30px rgba(15,23,42,.06);
}

.section-title {
    color: #111827;
    font-size: 1.1rem;
    font-weight: 800;
    margin: .2rem 0 1rem;
}

/* ================= RESULT ================= */

.result-card {
    position: relative;
    overflow: hidden;
    padding: 1.8rem;
    border-radius: 24px;
    border: 1px solid #c7d2fe;
    background: linear-gradient(145deg, #ffffff, #f5f7ff);
    box-shadow: 0 18px 50px rgba(15,23,42,.10);
    animation: resultIn .8s cubic-bezier(.2,.8,.2,1);
}

.result-card:before {
    content:"";
    position:absolute;
    width:260px;
    height:260px;
    border-radius:50%;
    background: rgba(99,102,241,.08);
    filter: blur(30px);
    top:-150px;
    right:-90px;
}

.result-label {
    color:#64748b;
    font-size:.82rem;
    text-transform:uppercase;
    letter-spacing:.12em;
    font-weight:700;
}

.result-value {
    color:#111827;
    font-size:4.2rem;
    line-height:1;
    font-weight:800;
    letter-spacing:-.06em;
    margin:.45rem 0;
}

.result-unit {
    color:#4f46e5;
    font-size:1rem;
    font-weight:700;
}

/* ================= ROUTE ANIMATION ================= */

.route {
    display:flex;
    align-items:center;
    gap:.8rem;
    margin-top:1.2rem;
}

.dot {
    width:11px;
    height:11px;
    border-radius:50%;
    background:#6366f1;
    box-shadow:0 0 0 7px rgba(99,102,241,.10);
}

.route-line {
    flex:1;
    height:2px;
    background: linear-gradient(
        90deg,
        rgba(99,102,241,.7),
        rgba(20,184,166,.35)
    );
    position:relative;
    overflow:hidden;
}

.route-line:after {
    content:"";
    position:absolute;
    left:-30%;
    top:0;
    width:30%;
    height:100%;
    background:#6366f1;
    animation:travel 1.8s linear infinite;
}

/* ================= BUTTON ================= */

div.stButton > button {
    border-radius:14px;
    border:1px solid #c7d2fe;
    min-height:3rem;
    font-weight:800;
    background:#ffffff;
    color:#3730a3;
    transition:all .2s ease;
}

div.stButton > button:hover {
    transform:translateY(-2px);
    border-color:#818cf8;
    box-shadow:0 10px 28px rgba(99,102,241,.15);
}

/* ================= SIDEBAR ================= */

[data-testid="stSidebar"] {
    background:#ffffff;
    border-right:1px solid #e5e7eb;
}

[data-testid="stSidebar"] * {
    color:#1f2937;
}

/* ================= METRICS ================= */

[data-testid="stMetric"] {
    background:#ffffff;
    border:1px solid #e5e7eb;
    padding:.8rem;
    border-radius:14px;
    box-shadow:0 5px 18px rgba(15,23,42,.04);
}

/* ================= INPUTS ================= */

[data-baseweb="select"] > div {
    background-color:#ffffff;
    border-color:#d1d5db;
}

input {
    background-color:#ffffff !important;
}

/* ================= ANIMATIONS ================= */

@keyframes fadeUp {
    from {
        opacity:0;
        transform:translateY(18px);
    }

    to {
        opacity:1;
        transform:translateY(0);
    }
}

@keyframes resultIn {
    0% {
        opacity:0;
        transform:translateY(28px) scale(.97);
    }

    70% {
        opacity:1;
        transform:translateY(-3px) scale(1.005);
    }

    100% {
        opacity:1;
        transform:translateY(0) scale(1);
    }
}

@keyframes travel {
    from {
        left:-30%;
    }

    to {
        left:110%;
    }
}

.small-note {
    color:#64748b;
    font-size:.78rem;
    margin-top:.75rem;
}
</style>
""", unsafe_allow_html=True)
# ============================================================
# Load model + scaler
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(BASE_DIR / "standard_scaler.pkl")
    model = joblib.load(BASE_DIR / "xgb_best.pkl")
    return scaler, model

try:
    scaler, model = load_artifacts()
except Exception as e:
    st.error("Model files could not be loaded. Keep standard_scaler.pkl and xgb_best.pkl in the same folder as app.py.")
    st.exception(e)
    st.stop()

FEATURES = list(scaler.feature_names_in_)

# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI DELIVERY ETA • XGBOOST</div>
    <h1>Predict your delivery time.<br>Before the rider arrives.</h1>
    <p>
        Enter the order conditions below. The trained XGBoost model will estimate
        the delivery time in minutes using the same 21-feature preprocessing pipeline
        used during model training.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🚴 SwiftDrop")
    st.caption("Food Delivery Time Prediction")
    st.divider()

    st.markdown("### Model")
    st.write("**XGBoost Regressor**")
    st.write("21 scaled features")
    st.write("Target: `Time_taken(min)`")

    st.divider()
    st.markdown("### Input guide")
    st.caption("Use the values that describe the order at the moment it is placed.")

    if st.button("↺ Reset inputs", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ============================================================
# Inputs
# ============================================================
st.markdown('<div class="section-title">📦 Order details</div>', unsafe_allow_html=True)

left, right = st.columns(2, gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    age = st.slider("👤 Delivery person age", 18, 50, 30)
    rating = st.slider("⭐ Delivery person rating", 1.0, 5.0, 4.7, 0.1)
    distance = st.slider("📍 Delivery distance (km)", 0.5, 40.0, 8.0, 0.5)
    order_hour = st.slider("🕒 Order hour", 0, 23, 18)
    vehicle_condition = st.slider("🛵 Vehicle condition", 0, 3, 1)
    multiple_deliveries = st.selectbox(
        "📦 Multiple deliveries",
        [0, 1, 2, 3],
        index=0,
        help="Number of deliveries being handled in the same trip."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    weather = st.selectbox(
        "🌦️ Weather",
        ["Sunny", "Windy", "Fog", "Sandstorms", "Stormy", "Cloudy"]
    )
    traffic = st.selectbox(
        "🚦 Road traffic density",
        ["Low", "Medium", "High", "Jam"],
        index=1
    )
    order_type = st.selectbox(
        "🍱 Type of order",
        ["Buffet", "Drinks", "Meal", "Snack"]
    )
    vehicle = st.selectbox(
        "🏍️ Type of vehicle",
        ["Bicycle", "Electric scooter", "Motorcycle", "Scooter"]
    )
    festival = st.selectbox("🎉 Festival", ["No", "Yes"])
    city = st.selectbox(
        "🏙️ City",
        ["Metropolitian", "Urban", "Semi-Urban"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Prediction helper
# ============================================================
def build_features():
    row = pd.DataFrame(0, index=[0], columns=FEATURES)

    row.loc[0, "Delivery_person_Age"] = age
    row.loc[0, "Delivery_person_Ratings"] = rating
    row.loc[0, "distance_km"] = distance
    row.loc[0, "order_hour"] = order_hour
    row.loc[0, "Road_traffic_density"] = {
        "Low": 0, "Medium": 1, "High": 2, "Jam": 3
    }[traffic]
    row.loc[0, "Vehicle_condition"] = vehicle_condition
    row.loc[0, "multiple_deliveries"] = multiple_deliveries

    # drop_first=True was used during training.
    # Therefore the first category in each original categorical feature is the baseline.
    if order_type != "Buffet":
        row.loc[0, f"Type_of_order_{order_type}"] = 1

    if festival == "Yes":
        row.loc[0, "Festival_Yes"] = 1

    vehicle_map = {
        "Bicycle": None,
        "Electric scooter": "Type_of_vehicle_electric_scooter",
        "Motorcycle": "Type_of_vehicle_motorcycle",
        "Scooter": "Type_of_vehicle_scooter",
    }
    if vehicle_map[vehicle]:
        row.loc[0, vehicle_map[vehicle]] = 1

    if city != "Metropolitian":
        row.loc[0, f"City_{city}"] = 1

    if weather != "Cloudy":
        row.loc[0, f"Weatherconditions_{weather}"] = 1

    return row

# ============================================================
# Predict
# ============================================================
st.markdown("###")

if st.button("⚡ Predict delivery time", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing route, traffic and order conditions..."):
            time.sleep(0.65)

        input_df = build_features()
        scaled_input = scaler.transform(input_df)
        prediction = float(model.predict(scaled_input)[0])
        prediction = max(1.0, prediction)

        st.session_state["prediction"] = prediction

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated delivery time</div>
            <div class="result-value">{prediction:.0f}<span class="result-unit"> min</span></div>
            <div style="color:#94a3b8;">
                Based on the order conditions you entered.
            </div>
            <div class="route">
                <div class="dot"></div>
                <div class="route-line"></div>
                <div style="font-size:1.3rem;">🍔</div>
                <div class="dot" style="background:#2dd4bf;"></div>
            </div>
            <div class="small-note">
                This is a model estimate, not a guaranteed delivery time.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

        st.markdown("###")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Distance", f"{distance:.1f} km")
        with c2:
            st.metric("Traffic", traffic)
        with c3:
            st.metric("Weather", weather)

    except Exception as e:
        st.error("Prediction failed. Check that the input feature names and model/scaler files match.")
        st.exception(e)

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div style="text-align:center; color:#475569; padding:2rem 0 .5rem; font-size:.78rem;">
    Built with Streamlit • StandardScaler • XGBoost
</div>
""", unsafe_allow_html=True)
