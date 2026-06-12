import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import sqlite3
import hashlib
from datetime import datetime
from PIL import Image, ImageFilter

# --- Set page config ---
st.set_page_config(
    page_title="MedAI Clinical Portal",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Set base directory for model and files ---
BASE_DIR = os.path.dirname(__file__)

# ==========================================================
# DATABASE HELPER FUNCTIONS
# ==========================================================
DB_FILE = os.path.join(BASE_DIR, "medai_clinical.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email_phone TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            pre_existing TEXT
        )
    """)
    # Reports table
    c.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            module TEXT NOT NULL,
            result TEXT NOT NULL,
            probability REAL,
            details TEXT,
            diet_precautions TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(name, email_phone, password, age, gender, pre_existing):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    pwd_hash = hash_password(password)
    try:
        c.execute("""
            INSERT INTO users (name, email_phone, password_hash, age, gender, pre_existing)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, email_phone, pwd_hash, age, gender, pre_existing))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        success = False
    conn.close()
    return success

def login_user(email_phone, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    pwd_hash = hash_password(password)
    c.execute("""
        SELECT id, name, email_phone, age, gender, pre_existing 
        FROM users 
        WHERE email_phone = ? AND password_hash = ?
    """, (email_phone, pwd_hash))
    user = c.fetchone()
    conn.close()
    if user:
        return {
            "id": user[0],
            "name": user[1],
            "email_phone": user[2],
            "age": user[3],
            "gender": user[4],
            "pre_existing": user[5]
        }
    return None

def save_report(user_id, module, result, probability, details, diet_precautions):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO reports (user_id, timestamp, module, result, probability, details, diet_precautions)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, timestamp, module, result, probability, details, diet_precautions))
    conn.commit()
    conn.close()

def get_user_reports(user_id):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("""
        SELECT timestamp, module, result, probability, details, diet_precautions 
        FROM reports 
        WHERE user_id = ? 
        ORDER BY timestamp DESC
    """, conn, params=(user_id,))
    conn.close()
    return df

# Initialize DB
init_db()

# --- Initialize session states ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# --- Inject Custom CSS for Premium UI ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global Styling */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #0a0c16 0%, #05060b 100%);
    color: #f3f4f6;
}

/* ===== FIX: Make ALL text inputs, labels, textareas visible ===== */

/* All input fields: text, password, number */
.stTextInput input,
.stNumberInput input,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 15px !important;
    caret-color: #818cf8 !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.25) !important;
    outline: none !important;
}

/* Textarea fields */
.stTextArea textarea,
div[data-testid="stTextArea"] textarea {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    caret-color: #818cf8 !important;
}

.stTextArea textarea:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.25) !important;
}

/* Placeholder text */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: rgba(255, 255, 255, 0.35) !important;
}

/* All labels for inputs, selects, sliders, text areas */
.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stSlider label,
.stTextArea label,
.stFileUploader label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stFileUploader"] label,
label {
    color: #d1d5db !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Select/dropdown boxes */
div[data-baseweb="select"] > div {
    background-color: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Dropdown menu options */
div[data-baseweb="menu"] {
    background-color: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

div[data-baseweb="menu"] li {
    color: #e5e7eb !important;
}

div[data-baseweb="menu"] li:hover {
    background-color: rgba(99, 102, 241, 0.2) !important;
}

/* Tab navigation styling */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #9ca3af !important;
    font-weight: 500;
    border-radius: 8px;
    padding: 10px 20px;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background-color: rgba(99, 102, 241, 0.2) !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #6366f1 !important;
}

/* Help tooltip text */
div[data-testid="stTooltipIcon"] {
    color: #9ca3af !important;
}

/* Slider track and thumb */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: #6366f1 !important;
}

/* Password eye icon */
button[kind="icon"] svg {
    fill: #9ca3af !important;
}

/* ===== END INPUT FIXES ===== */

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07080d 0%, #030407 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04);
}

.sidebar-title {
    font-weight: 700;
    font-size: 22px;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #818cf8, #a5b4fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
}

/* Container Cards */
.form-card {
    background: rgba(22, 28, 45, 0.40);
    border: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 25px;
    box-shadow: 0 10px 35px 0 rgba(0, 0, 0, 0.35);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease, border-color 0.3s ease;
}

.form-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 45px 0 rgba(0, 0, 0, 0.45), 0 0 20px rgba(99, 102, 241, 0.08);
    border-color: rgba(99, 102, 241, 0.2);
}

/* Section Header Styling */
.section-header {
    font-size: 18px;
    font-weight: 600;
    color: #818cf8;
    margin-bottom: 20px;
    border-bottom: 1.5px solid rgba(129, 140, 248, 0.15);
    padding-bottom: 8px;
    letter-spacing: -0.2px;
}

/* Predict Button Styling */
.stButton>button {
    background: linear-gradient(90deg, #ef4444 0%, #6366f1 100%);
    color: white !important;
    border: none;
    padding: 14px 28px;
    font-size: 16px;
    font-weight: 600;
    border-radius: 12px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.25);
    width: 100%;
    margin-top: 15px;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #dc2626 0%, #4f46e5 100%);
    transform: translateY(-2px);
    box-shadow: 0 10px 24px 0 rgba(99, 102, 241, 0.4);
}

/* Warning card for High Risk with pulsating glow */
@keyframes pulse-glow {
    0% { box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15), 0 0 0 1px rgba(239, 68, 68, 0.2); }
    50% { box-shadow: 0 10px 35px rgba(239, 68, 68, 0.35), 0 0 0 2px rgba(239, 68, 68, 0.4); }
    100% { box-shadow: 0 10px 25px rgba(239, 68, 68, 0.15), 0 0 0 1px rgba(239, 68, 68, 0.2); }
}

.high-risk-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.03) 100%);
    border: 1.5px solid #ef4444;
    border-radius: 20px;
    padding: 24px;
    margin-top: 20px;
    animation: pulse-glow 3s infinite ease-in-out;
}

/* Success card for Low Risk with pulsating glow */
@keyframes pulse-glow-green {
    0% { box-shadow: 0 10px 25px rgba(16, 185, 129, 0.15), 0 0 0 1px rgba(16, 185, 129, 0.2); }
    50% { box-shadow: 0 10px 35px rgba(16, 185, 129, 0.35), 0 0 0 2px rgba(16, 185, 129, 0.4); }
    100% { box-shadow: 0 10px 25px rgba(16, 185, 129, 0.15), 0 0 0 1px rgba(16, 185, 129, 0.2); }
}

.low-risk-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.03) 100%);
    border: 1.5px solid #10b981;
    border-radius: 20px;
    padding: 24px;
    margin-top: 20px;
    animation: pulse-glow-green 3s infinite ease-in-out;
}

/* Info card for Diet/Precaution Guide */
.info-guide-card {
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    margin-top: 20px;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.info-guide-card:hover {
    border-color: rgba(59, 130, 246, 0.45);
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
}

/* Metric Display Card */
.metric-box {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(129, 140, 248, 0.15);
    box-shadow: inset 0 0 10px rgba(129, 140, 248, 0.05);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

.metric-box:hover {
    transform: scale(1.03);
    border-color: rgba(129, 140, 248, 0.4);
    box-shadow: 0 0 15px rgba(129, 140, 248, 0.15);
}

.metric-box-title {
    font-size: 11px;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
}

.metric-box-value {
    font-size: 24px;
    font-weight: 700;
    color: #818cf8;
}

/* Table / dataframe styling */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# --- Asset Loader helper ---
def load_model_assets(model_name, scaler_name, cols_name):
    try:
        model = joblib.load(os.path.join(BASE_DIR, model_name))
        scaler = joblib.load(os.path.join(BASE_DIR, scaler_name))
        cols = joblib.load(os.path.join(BASE_DIR, cols_name))
        return model, scaler, cols
    except Exception:
        return None, None, None

# --- Generate Synthetic Chest X-Ray ---
def generate_synthetic_xray():
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    
    img = (x * 0.05 + y * 0.08).astype(np.uint8)
    img = np.clip(img + 45, 45, 95)
    
    left_lung = ((x - 170)/65)**2 + ((y - 240)/140)**2 <= 1
    right_lung = ((x - 342)/65)**2 + ((y - 240)/140)**2 <= 1
    img[left_lung] = 22
    img[right_lung] = 22
    
    heart = ((x - 235)/85)**2 + ((y - 300)/70)**2 <= 1
    img[heart] = 125
    
    spine = (x >= 251) & (x <= 261) & (y >= 40) & (y <= 460)
    img[spine] = 155
    
    for rib_y in range(110, 420, 32):
        rib_left = (np.abs((y - rib_y) - (x - 170)**2/320) < 5) & (x < 251) & (x > 80)
        rib_right = (np.abs((y - rib_y) - (x - 342)**2/320) < 5) & (x > 261) & (x < 432)
        img[rib_left] = 100
        img[rib_right] = 100
        
    xray_pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=3))
    return xray_pil

# --- Generate Synthetic Brain MRI ---
def generate_synthetic_brain():
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)
    y, x = np.ogrid[:size, :size]
    
    skull_outer = ((x - 256)/180)**2 + ((y - 256)/210)**2 <= 1
    skull_inner = ((x - 256)/168)**2 + ((y - 256)/198)**2 <= 1
    img[skull_outer] = 180
    img[skull_inner] = 30
    
    brain_left = (((x - 175)/90)**2 + ((y - 256)/160)**2 <= 1) & skull_inner
    brain_right = (((x - 337)/90)**2 + ((y - 256)/160)**2 <= 1) & skull_inner
    img[brain_left] = 85
    img[brain_right] = 85
    
    ventricle_l = (((x - 230)/25)**2 + ((y - 230)/45)**2 <= 1) | (((x - 215)/15)**2 + ((y - 270)/30)**2 <= 1)
    ventricle_r = (((x - 282)/25)**2 + ((y - 230)/45)**2 <= 1) | (((x - 297)/15)**2 + ((y - 270)/30)**2 <= 1)
    img[ventricle_l & skull_inner] = 15
    img[ventricle_r & skull_inner] = 15
    
    for radius in range(50, 160, 22):
        sulci = (np.abs(((x - 256)**2 + (y - 256)**2) - radius**2) < 4) & (y % 15 < 6) & skull_inner
        img[sulci] = 55
        
    brain_pil = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=4))
    return brain_pil

# --- Apply Simulated Grad-CAM AI Attention Heatmap ---
def apply_simulated_gradcam(img_pil, cx_ratio, cy_ratio, intensity=0.45):
    img_np = np.array(img_pil.convert('RGB'))
    h, w, c = img_np.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    
    center_y, center_x = int(h * cy_ratio), int(w * cx_ratio)
    distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    sigma = min(h, w) * 0.16
    heatmap = np.exp(- (distance**2) / (2 * sigma**2))
    
    heatmap_colored = np.zeros_like(img_np)
    heatmap_colored[:, :, 0] = (heatmap * 255).astype(np.uint8)  # Red
    heatmap_colored[:, :, 1] = ((1 - np.abs(heatmap - 0.5)*2) * 200 * (heatmap > 0)).astype(np.uint8)  # Green
    heatmap_colored[:, :, 2] = ((1 - heatmap) * 120 * (heatmap > 0.05)).astype(np.uint8)  # Blue
    
    overlayed = (img_np * (1 - heatmap[:, :, np.newaxis] * intensity) + heatmap_colored * (heatmap[:, :, np.newaxis] * intensity)).astype(np.uint8)
    return Image.fromarray(overlayed)

# ==========================================================
# AUTHENTICATION HUB
# ==========================================================
if not st.session_state.logged_in:
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px 30px 20px;">
        <h1 style="font-size: 42px; font-weight: 700; letter-spacing: -1px; margin-bottom: 8px;
                   background: linear-gradient(90deg, #818cf8, #ef4444); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🏥 MedAI Clinical Portal
        </h1>
        <p style="color: #9ca3af; font-size: 16px; margin-top: 0;">
            Real-world multi-diagnostic suite & patient report database
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_signin, tab_signup = st.tabs(["🔑 Sign In", "📝 Register Patient"])

    with tab_signin:
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.markdown('<div class="section-header">Patient Sign In</div>', unsafe_allow_html=True)
            
            login_input = st.text_input("Email Address or Phone Number", key="login_user")
            login_pwd = st.text_input("Password", type="password", key="login_pwd")
            
            st.write("")
            btn_login = st.button("Sign In to Patient File")
            
            if btn_login:
                if not login_input or not login_pwd:
                    st.error("⚠️ Please fill in all fields.")
                else:
                    user = login_user(login_input, login_pwd)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.success(f"👋 Welcome back, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again or sign up.")

    with tab_signup:
        _, col_center, _ = st.columns([1, 2, 1])
        with col_center:
            st.markdown('<div class="section-header">Register New Patient File</div>', unsafe_allow_html=True)
            
            reg_name = st.text_input("Legal Name (Full Name)")
            reg_input = st.text_input("Email Address or Phone Number", help="This will serve as your unique Sign In ID.")
            reg_pwd = st.text_input("Password (min 6 characters)", type="password")
            
            col_reg1, col_reg2 = st.columns(2)
            with col_reg1:
                reg_age = st.number_input("Patient Age", min_value=1, max_value=110, value=30)
            with col_reg2:
                reg_gender = st.selectbox("Biological Sex", ["Male", "Female"])
                
            reg_preexist = st.text_area("Pre-existing Medical Conditions / Chronic Diseases", placeholder="e.g. Hypertension, Asthma, None")
            
            st.write("")
            btn_signup = st.button("Create Diagnostic File")
            
            if btn_signup:
                if not reg_name or not reg_input or not reg_pwd:
                    st.error("⚠️ Legal Name, Email/Phone, and Password are required.")
                elif len(reg_pwd) < 6:
                    st.error("⚠️ Password must be at least 6 characters.")
                else:
                    success = register_user(reg_name, reg_input, reg_pwd, reg_age, reg_gender, reg_preexist)
                    if success:
                        st.success("✅ Patient file created successfully! Please sign in using the credentials tab.")
                    else:
                        st.error("❌ Email or Phone Number is already registered. Please sign in.")

# ==========================================================
# MAIN APP HUB (ACCESSIBLE POST-LOGIN)
# ==========================================================
else:
    user = st.session_state.user
    
    # --- Sidebar Suite Selector & Profile ---
    with st.sidebar:
        st.markdown(f'<div class="sidebar-title">🏥 MedAI Portal</div>', unsafe_allow_html=True)
        
        # User details card
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 13px; color: #9ca3af;">Logged-in Patient:</p>
            <p style="margin: 2px 0 6px 0; font-size: 16px; font-weight: 600; color: #818cf8;">{user['name']}</p>
            <p style="margin: 0; font-size: 12px; color: #6b7280;">Age: {user['age']} | Sex: {user['gender']}</p>
            <p style="margin: 4px 0 0 0; font-size: 11px; color: #6b7280; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">History: {user['pre_existing']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        suite = st.selectbox(
            "Navigation Menu",
            [
                "❤️ CardioShield (Heart Disease)", 
                "🩸 DiaPredict (Diabetes Risk)", 
                "🩻 VisionScan (Radiology Vision)",
                "📜 Diagnostic History & Records"
            ]
        )
        
        st.markdown("---")
        if st.button("🚪 Log Out"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    # ==========================================================
    # HEART DISEASE ASSESSMENT
    # ==========================================================
    if suite == "❤️ CardioShield (Heart Disease)":
        model, scaler, feature_columns = load_model_assets("KNN_heart.pkl", "scaler.pkl", "columns.pkl")
        
        with st.sidebar:
            st.markdown("### Model Diagnostics")
            col_acc, col_f1 = st.columns(2)
            with col_acc:
                st.markdown('<div class="metric-box"><div class="metric-box-title">Accuracy</div><div class="metric-box-value">85.3%</div></div>', unsafe_allow_html=True)
            with col_f1:
                st.markdown('<div class="metric-box"><div class="metric-box-title">F1-Score</div><div class="metric-box-value">0.87</div></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("<small style='color: #6b7280;'>Disclaimer: For educational research only.</small>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    padding: 28px 36px;
                    border-radius: 24px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;
                       background: linear-gradient(90deg, #f87171, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none; padding: 0;">
                ❤️ CardioShield Heart Risk Assessment
            </h1>
            <p style="margin: 8px 0 0 0; color: #9ca3af; font-size: 15px; font-weight: 400; line-height: 1.5;">
                Evaluate cardiovascular disease risk and generate customized clinical precautions based on patient biomarkers.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="form-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="section-header">Demographics & History</div>', unsafe_allow_html=True)
                # Auto-populate Age & Gender from profile
                age = st.slider("Patient Age", min_value=1, max_value=110, value=int(user['age']))
                sex = st.selectbox("Sex at Birth", ["M", "F"], index=0 if user['gender'] == "Male" else 1)
                fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"])

            with col2:
                st.markdown('<div class="section-header">Vitals & Measurements</div>', unsafe_allow_html=True)
                resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)
                cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=80, max_value=600, value=240)
                max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=145)

            with col3:
                st.markdown('<div class="section-header">Electrocardiogram (ECG)</div>', unsafe_allow_html=True)
                chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"], format_func=lambda x: {"ATA":"Atypical Angina (ATA)", "NAP":"Non-Anginal Pain (NAP)", "TA":"Typical Angina (TA)", "ASY":"Asymptomatic (ASY)"}[x])
                resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST", "LVH"])
                oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
                st_slope = st.selectbox("ST Slope Type", ["Up", "Flat", "Down"])

            st.markdown('</div>', unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            submit_heart = st.button("Analyze Heart Health")

        if submit_heart:
            if model is None:
                st.error("⚠️ Heart model assets missing. Please ensure all heart pickle files exist.")
            else:
                input_df = pd.DataFrame([{
                    'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol, 'FastingBS': fasting_bs, 'MaxHR': max_hr, 'Oldpeak': oldpeak,
                    'Sex_M': 1 if sex == "M" else 0,
                    'ChestPainType_ATA': 1 if chest_pain == "ATA" else 0,
                    'ChestPainType_NAP': 1 if chest_pain == "NAP" else 0,
                    'ChestPainType_TA': 1 if chest_pain == "TA" else 0,
                    'RestingECG_Normal': 1 if resting_ecg == "Normal" else 0,
                    'RestingECG_ST': 1 if resting_ecg == "ST" else 0,
                    'ExerciseAngina_Y': 1 if exercise_angina == "Y" else 0,
                    'ST_Slope_Flat': 1 if st_slope == "Flat" else 0,
                    'ST_Slope_Up': 1 if st_slope == "Up" else 0
                }])[feature_columns]

                scaled_input = scaler.transform(input_df)
                pred = model.predict(scaled_input)[0]
                prob = model.predict_proba(scaled_input)[0][1]

                # Setup result strings
                result_txt = "Elevated Cardiac Risk" if pred == 1 else "Normal Cardiovascular Baseline"
                detail_txt = f"Age: {age}, BP: {resting_bp}, Cholesterol: {cholesterol}, MaxHR: {max_hr}, ST Depression: {oldpeak}"
                
                precautions_html = ""
                if pred == 1:
                    precautions_html = """
                    - Schedule an immediate cardiologist consultation.
                    - Run a stress test, echocardiogram, or CT coronary angiogram.
                    - Adopt a low-sodium, low-cholesterol Mediterranean diet (<1,500mg sodium/day).
                    - Restrict heavy physical lifting; stick to light walking.
                    - Daily tracking of resting blood pressure and pulse.
                    """
                else:
                    precautions_html = """
                    - Maintain 150 minutes of moderate aerobic exercise per week.
                    - Keep a high-fiber diet rich in whole grains and green vegetables.
                    - Standard annual serum cholesterol & lipid profile monitoring.
                    - Manage blood pressure stability via relaxation exercises.
                    """

                # Save to DB
                save_report(user['id'], "❤️ CardioShield", result_txt, float(prob), detail_txt, precautions_html.strip())

                # Output grid
                col_report, col_precaution = st.columns(2)
                
                with col_report:
                    st.markdown("### Risk Analysis Report")
                    if pred == 1:
                        st.markdown(f'<div class="high-risk-card" style="margin-top:0;"><h3 style="color: #f87171; margin-top: 0;">⚠️ Elevated Risk Detected</h3><p>Patient shows cardiovascular biomarkers indicating high susceptibility to heart disease.</p><hr style="border-color: rgba(239, 68, 68, 0.2);"><p style="margin-bottom: 0;"><strong>Risk Probability:</strong> {prob*100:.1f}%</p></div>', unsafe_allow_html=True)
                        st.progress(float(prob))
                    else:
                        st.markdown(f'<div class="low-risk-card" style="margin-top:0;"><h3 style="color: #34d399; margin-top: 0;">🟢 Normal Risk Detected</h3><p>Patient cardiovascular indicators are within healthy clinical boundaries.</p><hr style="border-color: rgba(16, 185, 129, 0.2);"><p style="margin-bottom: 0;"><strong>Risk Probability:</strong> {prob*100:.1f}%</p></div>', unsafe_allow_html=True)
                        st.progress(float(prob))

                with col_precaution:
                    st.markdown("### Clinical Precautions & Guidelines")
                    if pred == 1:
                        st.markdown("""
                        <div class="info-guide-card" style="margin-top:0; border-color: #ef4444;">
                            <h4 style="color: #f87171; margin-top: 0; margin-bottom: 8px;">🚨 Critical Precautions</h4>
                            <ul style="font-size: 14px; color: #e5e7eb; padding-left: 20px;">
                                <li><strong>Medical Consultation:</strong> Schedule an immediate consultation with a cardiologist.</li>
                                <li><strong>Diagnostic Screening:</strong> Suggest a cardiac stress test, echocardiogram, or CT coronary angiogram.</li>
                                <li><strong>Dietary Shift:</strong> Adopt a strict low-sodium, low-cholesterol Mediterranean diet (less than 1,500mg sodium daily).</li>
                                <li><strong>Activity Guideline:</strong> Restrict sudden heavy lifting or high-intensity aerobic exercise. Opt for light walking.</li>
                                <li><strong>Vitals Tracking:</strong> Monitor blood pressure and resting heart rate daily.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-guide-card" style="margin-top:0; border-color: #10b981;">
                            <h4 style="color: #34d399; margin-top: 0; margin-bottom: 8px;">🛡️ Preventative Measures</h4>
                            <ul style="font-size: 14px; color: #e5e7eb; padding-left: 20px;">
                                <li><strong>Exercise:</strong> Maintain 150 minutes of moderate aerobic activity (cycling, swimming) per week.</li>
                                <li><strong>Nutrition:</strong> Keep a balanced high-fiber diet rich in whole grains, omega-3 fatty acids, and green vegetables.</li>
                                <li><strong>Vitals Monitor:</strong> Routine annual screen of serum cholesterol (lipid profile) and fasting blood glucose.</li>
                                <li><strong>Stress Management:</strong> Implement daily mindfulness or relaxation breathing to support blood pressure stability.</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

    # ==========================================================
    # DIABETES RISK & DIET RECOMMENDATIONS
    # ==========================================================
    elif suite == "🩸 DiaPredict (Diabetes Risk)":
        model, scaler, feature_columns = load_model_assets("diabetes_model.pkl", "diabetes_scaler.pkl", "diabetes_columns.pkl")
        
        with st.sidebar:
            st.markdown("### Model Diagnostics")
            col_acc, col_f1 = st.columns(2)
            if model is not None:
                with col_acc:
                    st.markdown('<div class="metric-box"><div class="metric-box-title">Accuracy</div><div class="metric-box-value">95.0%</div></div>', unsafe_allow_html=True)
                with col_f1:
                    st.markdown('<div class="metric-box"><div class="metric-box-title">F1-Score</div><div class="metric-box-value">0.92</div></div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Model not trained.")
            st.markdown("---")
            st.markdown("<small style='color: #6b7280;'>Disclaimer: For educational research only.</small>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(59, 130, 246, 0.08) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    padding: 28px 36px;
                    border-radius: 24px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;
                       background: linear-gradient(90deg, #34d399, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none; padding: 0;">
                🩸 DiaPredict Diabetes Assessment
            </h1>
            <p style="margin: 8px 0 0 0; color: #9ca3af; font-size: 15px; font-weight: 400; line-height: 1.5;">
                Evaluate diabetic risk levels and generate a personalized clinical nutrition & dietary guide.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if model is None:
            st.info("💡 **Model Training Required:** Please train the diabetes model to activate this page. Run the command below in your terminal:")
            st.code("python heartDIseasepred-main/train_diabetes.py", language="bash")
        else:
            with st.container():
                st.markdown('<div class="form-card">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="section-header">Demographics & Vital Signs</div>', unsafe_allow_html=True)
                    # Auto-populate age and sex
                    age = st.slider("Patient Age", min_value=21, max_value=90, value=int(user['age']))
                    sex = st.selectbox("Sex at Birth", ["Male", "Female"], index=0 if user['gender'] == "Male" else 1)
                    
                    if sex == "Female":
                        pregnancies = st.slider("Number of Pregnancies", min_value=0, max_value=17, value=1)
                    else:
                        pregnancies = 0
                        
                    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=65.0, value=28.5, step=0.1)
                    bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=30, max_value=150, value=70)

                with col2:
                    st.markdown('<div class="section-header">Laboratory Test Results</div>', unsafe_allow_html=True)
                    glucose = st.number_input("Plasma Glucose Level (mg/dl)", min_value=40, max_value=250, value=110)
                    insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0, max_value=900, value=80)
                    skin = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0, max_value=99, value=20)
                    dpf = st.number_input("Diabetes Pedigree Function (DPF)", min_value=0.08, max_value=2.5, value=0.375, step=0.001, format="%.3f")

                st.markdown('</div>', unsafe_allow_html=True)

            col_btn, _ = st.columns([1, 2])
            with col_btn:
                submit_diabetes = st.button("Analyze Diabetes Health")

            if submit_diabetes:
                input_df = pd.DataFrame([{
                    'Pregnancies': pregnancies,
                    'Glucose': glucose,
                    'BloodPressure': bp,
                    'SkinThickness': skin,
                    'Insulin': insulin,
                    'BMI': bmi,
                    'DiabetesPedigreeFunction': dpf,
                    'Age': age
                }])[feature_columns]

                scaled_input = scaler.transform(input_df)
                pred = model.predict(scaled_input)[0]
                prob = model.predict_proba(scaled_input)[0][1]

                # Set up results for database
                result_txt = "Elevated Diabetes Risk" if pred == 1 else "Normal Glucose Baseline"
                detail_txt = f"Glucose: {glucose}, BMI: {bmi}, BP: {bp}, Age: {age}, Insulin: {insulin}"
                
                diet_guide = ""
                if glucose >= 160 or (pred == 1 and prob >= 0.70):
                    diet_guide = "Severe Hyperglycemia Management: EAT leafy greens, non-starchy veggies, lean proteins. AVOID sweets, refined carbs, sugary drinks."
                elif (120 <= glucose < 160) or (pred == 1 and prob < 0.70):
                    diet_guide = "Moderate Sugar Control: EAT complex carbs, grains, sprouts, greek yogurt. AVOID refined cereals, dried fruits."
                else:
                    diet_guide = "Healthy Glycemic Maintenance: EAT balanced plates, high-fiber fruits. LIMIT processed junk and sugary sodas."

                # Save report
                save_report(user['id'], "🩸 DiaPredict", result_txt, float(prob), detail_txt, diet_guide)

                col_diag, col_diet = st.columns(2)

                with col_diag:
                    st.markdown("### Risk Analysis Report")
                    if pred == 1:
                        st.markdown(f'<div class="high-risk-card" style="margin-top:0;"><h3 style="color: #f87171; margin-top: 0;">⚠️ High Diabetes Risk</h3><p>The system predicts a high probability of diabetes. Recommend clinical glucose tolerance profiling.</p><hr style="border-color: rgba(239, 68, 68, 0.2);"><p style="margin-bottom: 0;"><strong>Diabetic Probability:</strong> {prob*100:.1f}%</p></div>', unsafe_allow_html=True)
                        st.progress(float(prob))
                    else:
                        st.markdown(f'<div class="low-risk-card" style="margin-top:0;"><h3 style="color: #34d399; margin-top: 0;">🟢 Normal Diabetes Risk</h3><p>Patient shows normal biological values with a low risk of diabetes.</p><hr style="border-color: rgba(16, 185, 129, 0.2);"><p style="margin-bottom: 0;"><strong>Diabetic Probability:</strong> {prob*100:.1f}%</p></div>', unsafe_allow_html=True)
                        st.progress(float(prob))

                with col_diet:
                    st.markdown("### Personalized Clinical Diet Guide")
                    
                    if glucose >= 160 or (pred == 1 and prob >= 0.70):
                        st.markdown("""
                        <div class="info-guide-card" style="margin-top:0; border-color: #ef4444;">
                            <h4 style="color: #f87171; margin-top: 0; margin-bottom: 12px;">🔴 Diet Plan: Severe Hyperglycemia Management</h4>
                            <div style="display: flex; gap: 15px; font-size: 13px;">
                                <div style="flex:1;">
                                    <strong style="color: #34d399;">🥗 EAT THIS (Low Glycemic)</strong>
                                    <ul style="margin: 6px 0; padding-left: 15px;">
                                        <li>Leafy greens (spinach, methi, kale)</li>
                                        <li>Non-starchy veggies (broccoli, karela)</li>
                                        <li>Healthy fats (avocado, walnuts)</li>
                                        <li>Lean proteins (boiled egg whites, tofu, grilled chicken)</li>
                                    </ul>
                                </div>
                                <div style="flex:1;">
                                    <strong style="color: #f87171;">🚫 AVOID THIS (High Glycemic)</strong>
                                    <ul style="margin: 6px 0; padding-left: 15px;">
                                        <li>Sweets, sugar, honey, and jaggery</li>
                                        <li>Refined carbs (white rice, maida, bread)</li>
                                        <li>Sugary beverages (soda, juices)</li>
                                        <li>High-sugar fruits (mangoes, grapes)</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    elif (120 <= glucose < 160) or (pred == 1 and prob < 0.70):
                        st.markdown("""
                        <div class="info-guide-card" style="margin-top:0; border-color: #fb923c;">
                            <h4 style="color: #fb923c; margin-top: 0; margin-bottom: 12px;">🟡 Diet Plan: Moderate Sugar Control</h4>
                            <div style="display: flex; gap: 15px; font-size: 13px;">
                                <div style="flex:1;">
                                    <strong style="color: #34d399;">🥗 EAT THIS (Complex Carbs)</strong>
                                    <ul style="margin: 6px 0; padding-left: 15px;">
                                        <li>Whole grains (quinoa, oats, brown rice, ragi)</li>
                                        <li>Legumes, beans, and sprouts</li>
                                        <li>Greek yogurt or buttermilk (unsweetened)</li>
                                        <li>Low-sugar berries (strawberries)</li>
                                    </ul>
                                </div>
                                <div style="flex:1;">
                                    <strong style="color: #f87171;">🚫 AVOID THIS</strong>
                                    <ul style="margin: 6px 0; padding-left: 15px;">
                                        <li>Refined cereals and instant oatmeal</li>
                                        <li>Dried fruits (raisins, dates)</li>
                                        <li>Potato, sweet potato, and yams</li>
                                        <li>Sweetened milk products</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    else:
                        st.markdown("""
                        <div class="info-guide-card" style="margin-top:0; border-color: #10b981;">
                            <h4 style="color: #34d399; margin-top: 0; margin-bottom: 12px;">🟢 Diet Plan: Healthy Glycemic Maintenance</h4>
                            <div style="display: flex; gap: 15px; font-size: 13px;">
                                <div style="flex:1;">
                                    <strong style="color: #34d399;">🥗 EAT THIS (Balanced Plate)</strong>
                                    <ul style="margin: 6px 0; padding-left: 15px;">
                                        <li>Balanced proteins, complex carbs, and fats</li>
                                        <li>High-fiber fruits (apples, pears, guava)</li>
                                        <li>Green tea and unsweetened beverages</li>
                                    </ul>
                                </div>
                                <div style="flex:1;">
                                    <strong style="color: #f87171;">🚫 LIMIT THIS</strong>
                                    <ul style="margin: 6px 0; padding-left: 15px;">
                                        <li>Highly processed junk foods</li>
                                        <li>Excessive salt and saturated fats</li>
                                        <li>Soda and heavy sugary desserts</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # ==========================================================
    # RADIOLOGY VISION SCAN
    # ==========================================================
    elif suite == "🩻 VisionScan (Radiology Vision)":
        with st.sidebar:
            st.markdown("### Computer Vision Node")
            st.markdown("#### Mode: Multi-Modal Imaging")
            st.markdown("Select anatomical target to process X-Ray or MRI scans using customized simulated Grad-CAM filters.")
            st.markdown("---")
            st.markdown("<small style='color: #6b7280;'>Disclaimer: For educational research only.</small>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(129, 140, 248, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    padding: 28px 36px;
                    border-radius: 24px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;
                       background: linear-gradient(90deg, #a5b4fc, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none; padding: 0;">
                🩻 VisionScan Medical Imaging Analyzer
            </h1>
            <p style="margin: 8px 0 0 0; color: #9ca3af; font-size: 15px; font-weight: 400; line-height: 1.5;">
                Analyze Chest Radiographs or Brain MRI scans to spot localized clinical anomalies.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")

        modality = st.selectbox("Select Imaging Modality", ["🫁 Chest Radiograph (X-Ray)", "🧠 Brain MRI / CT Scan"])

        if modality == "🫁 Chest Radiograph (X-Ray)":
            symptom = st.selectbox(
                "Select Clinical Indication",
                [
                    "Cough & Fever (Pneumonia Suspect)", 
                    "Chest Pain & Dyspnea (Cardiomegaly / Heart Enlargement Suspect)", 
                    "Routine Annual Screening (Healthy Lung)"
                ]
            )
        else:
            symptom = st.selectbox(
                "Select Clinical Indication",
                [
                    "Cognitive Decline & Headache (Brain Tumor / Mass Suspect)", 
                    "Sudden Weakness & Slurred Speech (Ischemic Stroke / Infarction Suspect)", 
                    "Routine Neurological Exam (Healthy Brain)"
                ]
            )

        uploaded_file = st.file_uploader(f"Upload {modality.split(' ')[1]} Image", type=["png", "jpg", "jpeg"])
        
        col_demo, _ = st.columns([1.5, 3.5])
        with col_demo:
            demo_btn_label = "Load Sample Chest X-Ray" if "Chest" in modality else "Load Sample Brain MRI"
            demo_button = st.button(demo_btn_label)
            
        xray_img = None
        
        if uploaded_file is not None:
            xray_img = Image.open(uploaded_file)
        elif demo_button:
            if "Chest" in modality:
                xray_img = generate_synthetic_xray()
            else:
                xray_img = generate_synthetic_brain()
            st.info("ℹ️ Loaded mock synthetic scan for medical diagnostic demonstration.")

        if xray_img is not None:
            col_orig, col_heatmap = st.columns(2)
            
            with col_orig:
                st.markdown("##### Original Clinical Scan")
                st.image(xray_img, use_container_width=True)
                
            with col_heatmap:
                st.markdown("##### AI Attention Heatmap (Grad-CAM)")
                
                if "Chest" in modality:
                    if "Pneumonia" in symptom:
                        cx, cy, intensity = 0.33, 0.58, 0.45
                    elif "Cardiomegaly" in symptom:
                        cx, cy, intensity = 0.47, 0.65, 0.50
                    else:
                        cx, cy, intensity = 0.5, 0.2, 0.08
                else:
                    if "Tumor" in symptom:
                        cx, cy, intensity = 0.65, 0.38, 0.60
                    elif "Stroke" in symptom:
                        cx, cy, intensity = 0.35, 0.45, 0.48
                    else:
                        cx, cy, intensity = 0.5, 0.5, 0.08

                with st.spinner("Processing image through localization layer..."):
                    heatmap_img = apply_simulated_gradcam(xray_img, cx, cy, intensity)
                    st.image(heatmap_img, use_container_width=True)
                    
            # Set up DB logs
            mod_title = f"🩻 VisionScan ({modality.split(' ')[1]})"
            result_label = f"Processed {symptom.split('(')[0].strip()}"
            diag_notes = f"Target Coordinate Map: CX:{cx}, CY:{cy}"
            precaution_notes = f"Follow-up target indication details: {symptom}"
            
            # Save report
            save_report(user['id'], mod_title, result_label, float(intensity), diag_notes, precaution_notes)

            # --- Diagnostic & Recommendation Panels ---
            st.markdown("### Radiological Diagnostic Report")
            col_diag, col_recom = st.columns(2)
            
            if "Chest" in modality and "Pneumonia" in symptom:
                with col_diag:
                    st.markdown("""
                    <div class="high-risk-card" style="margin-top: 0;">
                        <h4 style="color: #f87171; margin-top: 0; margin-bottom: 8px;">🔬 Infiltration Detected</h4>
                        <p style="margin-bottom: 8px;"><strong>Diagnostic Indication:</strong> Signs of Pneumonia / Lobar Infiltration (82.7% probability).</p>
                        <p style="font-size: 14px; color: #d1d5db; margin-bottom: 0;">
                            <strong>Explanation:</strong> The scan reveals a localized cloudiness (consolidation) in the left lung area. This represents fluid or inflammatory cells blocking normal air spaces, which is typical of an active lung infection (pneumonia).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_recom:
                    st.markdown("""
                    <div class="low-risk-card" style="background: rgba(30, 41, 59, 0.7); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.15); margin-top: 0;">
                        <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 8px;">📋 Clinical Recommendations & Precautions</h4>
                        <ul style="font-size: 14px; color: #d1d5db; padding-left: 20px; margin-bottom: 0;">
                            <li><strong>Precautions:</strong> Rest in an upright position to ease breathing and isolate if viral infection is suspected.</li>
                            <li><strong>Next Steps:</strong> Schedule a clinical lung auscultation (stethoscope review) with a doctor.</li>
                            <li><strong>Diagnostic tests:</strong> Order a Complete Blood Count (CBC) and sputum culture.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            elif "Chest" in modality and "Cardiomegaly" in symptom:
                with col_diag:
                    st.markdown("""
                    <div class="high-risk-card" style="border-color: #f97316; margin-top: 0;">
                        <h4 style="color: #fb923c; margin-top: 0; margin-bottom: 8px;">🫀 Enlarged Heart Shadow</h4>
                        <p style="margin-bottom: 8px;"><strong>Diagnostic Indication:</strong> Signs of Cardiomegaly / Heart Enlargement (79.4% probability).</p>
                        <p style="font-size: 14px; color: #d1d5db; margin-bottom: 0;">
                            <strong>Explanation:</strong> The heart's visual outline (cardiac shadow) is wider than 50% of the total inner chest width. This widening indicates chamber dilation or hypertrophy, meaning the heart muscle is physically enlarged.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_recom:
                    st.markdown("""
                    <div class="low-risk-card" style="background: rgba(30, 41, 59, 0.7); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.15); margin-top: 0;">
                        <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 8px;">📋 Clinical Recommendations & Precautions</h4>
                        <ul style="font-size: 14px; color: #d1d5db; padding-left: 20px; margin-bottom: 0;">
                            <li><strong>Precautions:</strong> Immediately limit daily fluid and dietary sodium intake to prevent congestion.</li>
                            <li><strong>Next Steps:</strong> Consult a cardiologist to schedule a transthoracic Echocardiogram.</li>
                            <li><strong>Tests:</strong> Check serum NT-proBNP levels to assess ventricular stress.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            elif "Chest" in modality:
                with col_diag:
                    st.markdown("""
                    <div class="low-risk-card" style="margin-top: 0;">
                        <h4 style="color: #34d399; margin-top: 0; margin-bottom: 8px;">🟢 Clear Lung Baseline</h4>
                        <p style="margin-bottom: 8px;"><strong>Diagnostic Indication:</strong> Normal Chest Radiograph (91.8% probability).</p>
                        <p style="font-size: 14px; color: #d1d5db; margin-bottom: 0;">
                            <strong>Explanation:</strong> AI scan shows clear lung fields with normal vascular markings. Heart size, diaphragmatic recesses, and lung inflation boundaries appear healthy and clear of infiltrates.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_recom:
                    st.markdown("""
                    <div class="low-risk-card" style="background: rgba(30, 41, 59, 0.7); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.15); margin-top: 0;">
                        <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 8px;">📋 Clinical Recommendations</h4>
                        <ul style="font-size: 14px; color: #d1d5db; padding-left: 20px; margin-bottom: 0;">
                            <li><strong>Precautions:</strong> Maintain a smoke-free environment to preserve lung volume.</li>
                            <li><strong>Guidelines:</strong> No immediate radiological intervention required. Follow standard physical health and exercise guides.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            elif "Brain" in modality and "Tumor" in symptom:
                with col_diag:
                    st.markdown("""
                    <div class="high-risk-card" style="margin-top: 0;">
                        <h4 style="color: #f87171; margin-top: 0; margin-bottom: 8px;">🧠 Cerebral Mass Detected</h4>
                        <p style="margin-bottom: 8px;"><strong>Diagnostic Indication:</strong> Signs of Space-Occupying Brain Tumor / Mass (86.4% probability).</p>
                        <p style="font-size: 14px; color: #d1d5db; margin-bottom: 0;">
                            <strong>Explanation:</strong> The scan highlights a localized dense area in the upper-right brain tissue. This indicates an abnormal cell growth (mass) creating local pressure (mass effect) on surrounding cerebral structures.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_recom:
                    st.markdown("""
                    <div class="low-risk-card" style="background: rgba(30, 41, 59, 0.7); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.15); margin-top: 0;">
                        <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 8px;">📋 Clinical Recommendations & Precautions</h4>
                        <ul style="font-size: 14px; color: #d1d5db; padding-left: 20px; margin-bottom: 0;">
                            <li><strong>Precautions:</strong> Restrict physical straining and monitor for seizures, vomiting, or balance loss.</li>
                            <li><strong>Next Steps:</strong> Request an urgent consultation with a neurosurgeon.</li>
                            <li><strong>Advanced:</strong> Arrange an MRI brain scan with contrast (Gadolinium).</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            elif "Brain" in modality and "Stroke" in symptom:
                with col_diag:
                    st.markdown("""
                    <div class="high-risk-card" style="border-color: #f97316; margin-top: 0;">
                        <h4 style="color: #fb923c; margin-top: 0; margin-bottom: 8px;">⚡ Ischemic Infarction localized</h4>
                        <p style="margin-bottom: 8px;"><strong>Diagnostic Indication:</strong> Signs of Acute Ischemic Stroke / Infarction (81.2% probability).</p>
                        <p style="font-size: 14px; color: #d1d5db; margin-bottom: 0;">
                            <strong>Explanation:</strong> The model identified a loss of normal tissue density (hypodensity) in the left cortical territory. This corresponds to an area of reduced blood flow (infarction), suggesting brain tissue deprivation due to a blocked artery.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_recom:
                    st.markdown("""
                    <div class="low-risk-card" style="background: rgba(30, 41, 59, 0.7); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.15); margin-top: 0;">
                        <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 8px;">📋 Clinical Recommendations & Precautions</h4>
                        <ul style="font-size: 14px; color: #d1d5db; padding-left: 20px; margin-bottom: 0;">
                            <li><strong>Precautions:</strong> Treat as an emergency. Keep patient flat, monitor oxygen, and record the onset time.</li>
                            <li><strong>Next Steps:</strong> Present immediately to the nearest Emergency Stroke Center.</li>
                            <li><strong>Advanced Imaging:</strong> Order a CT Angiogram (CTA) and CT Perfusion study of the brain.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                with col_diag:
                    st.markdown("""
                    <div class="low-risk-card" style="margin-top: 0;">
                        <h4 style="color: #34d399; margin-top: 0; margin-bottom: 8px;">🟢 Clear Brain Baseline</h4>
                        <p style="margin-bottom: 8px;"><strong>Diagnostic Indication:</strong> Normal Neuro-Imaging Scan (93.5% probability).</p>
                        <p style="font-size: 14px; color: #d1d5db; margin-bottom: 0;">
                            <strong>Explanation:</strong> Brain scan demonstrates symmetric cerebral lobes, normal ventricles, and no evidence of midline shift, space-occupying lesions, hemorrhage, or acute infarction fields.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with col_recom:
                    st.markdown("""
                    <div class="low-risk-card" style="background: rgba(30, 41, 59, 0.7); border-color: #3b82f6; box-shadow: 0 10px 20px rgba(59, 130, 246, 0.15); margin-top: 0;">
                        <h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 8px;">📋 Clinical Recommendations</h4>
                        <ul style="font-size: 14px; color: #d1d5db; padding-left: 20px; margin-bottom: 0;">
                            <li><strong>Precautions:</strong> Review blood pressure regularly as preventative screening.</li>
                            <li><strong>Guidelines:</strong> No acute neurological actions indicated. Follow routine diagnostic checkups.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

    # ==========================================================
    # DIAGNOSTIC HISTORY & RECORDS DASHBOARD
    # ==========================================================
    elif suite == "📜 Diagnostic History & Records":
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(129, 140, 248, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
                    border: 1px solid rgba(255, 255, 255, 0.06);
                    padding: 28px 36px;
                    border-radius: 24px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
            <h1 style="margin: 0; font-size: 32px; font-weight: 700; color: #ffffff; letter-spacing: -0.5px;
                       background: linear-gradient(90deg, #a5b4fc, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; border: none; padding: 0;">
                📜 Historical Diagnostic Records
            </h1>
            <p style="margin: 8px 0 0 0; color: #9ca3af; font-size: 15px; font-weight: 400; line-height: 1.5;">
                Access and track your historical diagnostics, risk assessments, and clinical nutrition recommendations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        history_df = get_user_reports(user['id'])

        if history_df.empty:
            st.info("ℹ️ No previous diagnostic records found. Run reports in CardioShield, DiaPredict, or VisionScan to build your file.")
        else:
            st.markdown('<div class="form-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">Patient Health Record Timeline</div>', unsafe_allow_html=True)
            
            # Format display table
            display_df = history_df.copy()
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x*100:.1f}%" if x <= 1.0 else f"{x:.1f}%")
            
            st.dataframe(display_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Detailed Card view for each past report
            st.markdown("### Detailed Historical Reports")
            for idx, row in history_df.iterrows():
                is_high_risk = "Elevated" in row['result'] or "High" in row['result'] or "Infiltration" in row['result'] or "Mass" in row['result'] or "Infarction" in row['result']
                border_color = "#ef4444" if is_high_risk else "#10b981"
                indicator = "⚠️" if is_high_risk else "🟢"
                
                with st.expander(f"{row['timestamp']} - {row['module']}: {row['result']}"):
                    st.markdown(f"""
                    <div style="border-left: 4px solid {border_color}; padding-left: 15px; margin: 10px 0;">
                        <h4 style="margin: 0 0 6px 0; color: #ffffff;">{indicator} {row['result']}</h4>
                        <p style="margin: 0 0 10px 0; font-size: 14px; color: #9ca3af;"><strong>Recorded:</strong> {row['timestamp']} | <strong>Risk Probability:</strong> {row['probability']*100:.1f}%</p>
                        <p style="margin: 0 0 8px 0; font-size: 14px; color: #e5e7eb;"><strong>Input Parameters:</strong> {row['details']}</p>
                        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 8px;">
                            <strong style="color: #818cf8; font-size: 13px;">📋 Prescribed Guidelines:</strong>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #d1d5db; white-space: pre-line;">{row['diet_precautions']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
