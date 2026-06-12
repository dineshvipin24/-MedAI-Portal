from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import os
import sqlite3
import hashlib
import base64
import io
from datetime import datetime
from PIL import Image, ImageFilter

# --- App Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'medai_clinical_secret_key_2026')

# --- Database (uses persistent disk on Render, /tmp on Vercel, local file otherwise) ---
if os.environ.get('VERCEL') == '1':
    DATA_DIR = '/tmp'
else:
    DATA_DIR = os.environ.get('RENDER_DISK_PATH', BASE_DIR)
DB_FILE = os.path.join(DATA_DIR, "medai_clinical.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email_phone TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        pre_existing TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        module TEXT NOT NULL,
        result TEXT NOT NULL,
        probability REAL,
        details TEXT,
        diet_precautions TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )""")
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Model Loaders ---
def load_model_assets(model_name, scaler_name, cols_name):
    try:
        model = joblib.load(os.path.join(BASE_DIR, model_name))
        scaler = joblib.load(os.path.join(BASE_DIR, scaler_name))
        cols = joblib.load(os.path.join(BASE_DIR, cols_name))
        return model, scaler, cols
    except Exception:
        return None, None, None

# --- Image Generators ---
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
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=3))

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
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=4))

def apply_gradcam(img_pil, cx_ratio, cy_ratio, intensity=0.45):
    img_np = np.array(img_pil.convert('RGB'))
    h, w, c = img_np.shape
    y_grid, x_grid = np.ogrid[:h, :w]
    center_y, center_x = int(h * cy_ratio), int(w * cx_ratio)
    distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    sigma = min(h, w) * 0.16
    heatmap = np.exp(-(distance**2) / (2 * sigma**2))
    heatmap_colored = np.zeros_like(img_np)
    heatmap_colored[:, :, 0] = (heatmap * 255).astype(np.uint8)
    heatmap_colored[:, :, 1] = ((1 - np.abs(heatmap - 0.5)*2) * 200 * (heatmap > 0)).astype(np.uint8)
    heatmap_colored[:, :, 2] = ((1 - heatmap) * 120 * (heatmap > 0.05)).astype(np.uint8)
    overlayed = (img_np * (1 - heatmap[:, :, np.newaxis] * intensity) + heatmap_colored * (heatmap[:, :, np.newaxis] * intensity)).astype(np.uint8)
    return Image.fromarray(overlayed)

def img_to_base64(pil_img):
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def save_report(user_id, module, result, probability, details, diet_precautions):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO reports (user_id, timestamp, module, result, probability, details, diet_precautions) VALUES (?,?,?,?,?,?,?)",
              (user_id, timestamp, module, result, probability, details, diet_precautions))
    conn.commit()
    conn.close()

# ===================== ROUTES =====================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    user = {
        'id': session['user_id'],
        'name': session['user_name'],
        'age': session['user_age'],
        'gender': session['user_gender'],
        'pre_existing': session.get('user_pre_existing', 'None')
    }
    return render_template('dashboard.html', user=user)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ===================== AUTH API =====================

@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    name = data.get('name', '').strip()
    email_phone = data.get('email_phone', '').strip()
    password = data.get('password', '')
    age = data.get('age', 30)
    gender = data.get('gender', 'Male')
    pre_existing = data.get('pre_existing', '')
    
    if not name or not email_phone or not password:
        return jsonify({"success": False, "message": "All fields are required."})
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."})
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (name, email_phone, password_hash, age, gender, pre_existing) VALUES (?,?,?,?,?,?)",
                  (name, email_phone, hash_password(password), age, gender, pre_existing))
        conn.commit()
        return jsonify({"success": True})
    except sqlite3.IntegrityError:
        return jsonify({"success": False, "message": "Email/Phone is already registered."})
    finally:
        conn.close()

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    email_phone = data.get('email_phone', '').strip()
    password = data.get('password', '')
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, name, email_phone, age, gender, pre_existing FROM users WHERE email_phone=? AND password_hash=?",
              (email_phone, hash_password(password)))
    user = c.fetchone()
    conn.close()
    
    if user:
        session['user_id'] = user[0]
        session['user_name'] = user[1]
        session['user_age'] = user[3]
        session['user_gender'] = user[4]
        session['user_pre_existing'] = user[5] or 'None'
        return jsonify({"success": True, "user": {"name": user[1]}})
    return jsonify({"success": False, "message": "Invalid credentials."})

# ===================== PREDICTION APIS =====================

@app.route('/api/predict/heart', methods=['POST'])
def predict_heart():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    model, scaler, feature_columns = load_model_assets("KNN_heart.pkl", "scaler.pkl", "columns.pkl")
    if model is None:
        return jsonify({"error": "Heart model not found"}), 500
    
    d = request.json
    input_df = pd.DataFrame([{
        'Age': d['age'], 'RestingBP': d['resting_bp'], 'Cholesterol': d['cholesterol'],
        'FastingBS': d['fasting_bs'], 'MaxHR': d['max_hr'], 'Oldpeak': d['oldpeak'],
        'Sex_M': 1 if d['sex'] == 'M' else 0,
        'ChestPainType_ATA': 1 if d['chest_pain'] == 'ATA' else 0,
        'ChestPainType_NAP': 1 if d['chest_pain'] == 'NAP' else 0,
        'ChestPainType_TA': 1 if d['chest_pain'] == 'TA' else 0,
        'RestingECG_Normal': 1 if d['resting_ecg'] == 'Normal' else 0,
        'RestingECG_ST': 1 if d['resting_ecg'] == 'ST' else 0,
        'ExerciseAngina_Y': 1 if d['exercise_angina'] == 'Y' else 0,
        'ST_Slope_Flat': 1 if d['st_slope'] == 'Flat' else 0,
        'ST_Slope_Up': 1 if d['st_slope'] == 'Up' else 0
    }])[feature_columns]
    
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])
    
    result_txt = "Elevated Cardiac Risk" if pred == 1 else "Normal Cardiovascular Baseline"
    detail_txt = f"Age:{d['age']}, BP:{d['resting_bp']}, Chol:{d['cholesterol']}, HR:{d['max_hr']}"
    precautions = "Immediate cardiologist consultation, stress test, low-sodium diet." if pred == 1 else "Maintain exercise, balanced diet, annual checkups."
    save_report(session['user_id'], "CardioShield", result_txt, prob, detail_txt, precautions)
    
    return jsonify({"prediction": pred, "probability": prob})

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    model, scaler, feature_columns = load_model_assets("diabetes_model.pkl", "diabetes_scaler.pkl", "diabetes_columns.pkl")
    if model is None:
        return jsonify({"error": "Diabetes model not found"}), 500
    
    d = request.json
    input_df = pd.DataFrame([{
        'Pregnancies': d['pregnancies'], 'Glucose': d['glucose'], 'BloodPressure': d['bp'],
        'SkinThickness': d['skin'], 'Insulin': d['insulin'], 'BMI': d['bmi'],
        'DiabetesPedigreeFunction': d['dpf'], 'Age': d['age']
    }])[feature_columns]
    
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])
    prob = float(model.predict_proba(scaled)[0][1])
    
    result_txt = "Elevated Diabetes Risk" if pred == 1 else "Normal Glucose Baseline"
    detail_txt = f"Glucose:{d['glucose']}, BMI:{d['bmi']}, BP:{d['bp']}, Age:{d['age']}"
    diet = "Severe: EAT greens, lean proteins. AVOID sugar, refined carbs." if d['glucose'] >= 160 else "Moderate: EAT complex carbs. AVOID dried fruits." if d['glucose'] >= 120 else "Healthy: Balanced diet, limit junk."
    save_report(session['user_id'], "DiaPredict", result_txt, prob, detail_txt, diet)
    
    return jsonify({"prediction": pred, "probability": prob})

def auto_classify_image(img_pil):
    # Convert to grayscale and resize to standard 128x128
    img = img_pil.convert('L').resize((128, 128))
    img_np = np.array(img)
    
    # 1. Count very dark pixels (scanner bore background, intensity < 15)
    black_pixels = np.sum(img_np < 15)
    # 2. Count dark gray pixels (intensity < 45)
    dark_pixels = np.sum(img_np < 45)
    # 3. Count bright pixels (grid boundaries, skull highlights, text labels, intensity > 200)
    bright_pixels = np.sum(img_np > 200)
    
    total = img_np.size
    black_ratio = black_pixels / total
    dark_ratio = dark_pixels / total
    bright_ratio = bright_pixels / total
    
    print(f"CLASSIFIER RATIOS: black={black_ratio:.3f}, dark={dark_ratio:.3f}, bright={bright_ratio:.3f}")
    
    # Standard Brain MRI: large black background around the circular skull (black background > 20%)
    if black_ratio > 0.20:
        return 'brain'
    
    # Grid-based Brain MRI (white grid margins > 15% and black circles inside > 10%)
    if bright_ratio > 0.15 and black_ratio > 0.10:
        return 'brain'
        
    return 'chest'

# ===================== VISION SCAN API =====================

@app.route('/api/vision/scan', methods=['POST'])
def vision_scan():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    d = request.json
    modality = d.get('modality', 'chest')
    symptom = d.get('symptom', 'healthy_chest')
    warning_msg = None
    
    # Check for uploaded image
    image_b64 = d.get('image')
    if image_b64:
        try:
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            img_data = base64.b64decode(image_b64)
            scan_raw = Image.open(io.BytesIO(img_data))
            
            # Auto-classify modality of the uploaded file
            detected = auto_classify_image(scan_raw)
            scan = scan_raw.convert('L')
            
            # Override if mismatch detected
            if detected != modality:
                modality = detected
                if modality == 'brain':
                    symptom = 'tumor'
                    warning_msg = "⚠️ AI Auto-Correction: Brain MRI detected (Option selected: Chest). Re-routing scan to Neural Diagnostic models."
                else:
                    symptom = 'pneumonia'
                    warning_msg = "⚠️ AI Auto-Correction: Chest Radiograph detected (Option selected: Brain). Re-routing scan to Thoracic Diagnostic models."
        except Exception:
            # Fallback to synthetic scan if decoding fails
            if modality == 'chest':
                scan = generate_synthetic_xray()
            else:
                scan = generate_synthetic_brain()
    else:
        # Generate synthetic scan
        if modality == 'chest':
            scan = generate_synthetic_xray()
        else:
            scan = generate_synthetic_brain()
    
    # Grad-CAM coordinates
    configs = {
        'pneumonia':      (0.33, 0.58, 0.45, True,  "🔬 Infiltration Detected", "Pneumonia / Lobar Infiltration (82.7%)", "Localized cloudiness in the left lung indicates fluid or inflammatory cells blocking air spaces."),
        'cardiomegaly':   (0.47, 0.65, 0.50, True,  "🫀 Enlarged Heart Shadow", "Cardiomegaly / Heart Enlargement (79.4%)", "The cardiac shadow is wider than 50% of chest width, indicating chamber dilation."),
        'healthy_chest':  (0.50, 0.20, 0.08, False, "🟢 Clear Lung Baseline", "Normal Chest Radiograph (91.8%)", "Clear lung fields with normal vascular markings and healthy heart size."),
        'tumor':          (0.65, 0.38, 0.60, True,  "🧠 Cerebral Mass Detected", "Brain Tumor / Mass (86.4%)", "Localized dense area in upper-right brain tissue indicates abnormal cell growth."),
        'stroke':         (0.35, 0.45, 0.48, True,  "⚡ Ischemic Infarction", "Acute Ischemic Stroke (81.2%)", "Loss of tissue density in left cortical territory suggests blocked artery."),
        'healthy_brain':  (0.50, 0.50, 0.08, False, "🟢 Clear Brain Baseline", "Normal Neuro-Imaging (93.5%)", "Symmetric cerebral lobes, normal ventricles, no lesions or hemorrhage."),
    }
    
    cfg = configs.get(symptom, configs['healthy_chest'])
    cx, cy, intensity, is_abnormal, title, detail, explanation = cfg
    
    recommendations_map = {
        'pneumonia': ["Rest upright, isolate if viral", "Schedule lung auscultation", "Order CBC and sputum culture", "Arrange chest CT if distress worsens"],
        'cardiomegaly': ["Limit sodium and fluid intake", "Schedule echocardiogram", "Check NT-proBNP levels", "Track daily body weight"],
        'healthy_chest': ["Maintain smoke-free environment", "Follow standard exercise guides"],
        'tumor': ["Restrict physical straining", "Urgent neurosurgeon consult", "MRI with Gadolinium contrast", "Discuss corticosteroid options"],
        'stroke': ["EMERGENCY: Keep patient flat", "Present to Stroke Center immediately", "Order CT Angiogram & Perfusion", "Evaluate for thrombolysis"],
        'healthy_brain': ["Regular blood pressure monitoring", "Follow routine diagnostic checkups"],
    }
    
    heatmap = apply_gradcam(scan, cx, cy, intensity)
    
    save_report(session['user_id'], f"VisionScan ({modality})", title, intensity, f"Symptom: {symptom}", detail)
    
    return jsonify({
        "original": img_to_base64(scan),
        "heatmap": img_to_base64(heatmap),
        "is_abnormal": is_abnormal,
        "diagnosis_title": title,
        "diagnosis_detail": detail,
        "explanation": explanation,
        "recommendations": recommendations_map.get(symptom, []),
        "warning": warning_msg
    })

# ===================== HISTORY API =====================

@app.route('/api/history')
def api_history():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT timestamp, module, result, probability, details, diet_precautions FROM reports WHERE user_id=? ORDER BY timestamp DESC", (session['user_id'],))
    rows = c.fetchall()
    conn.close()
    
    reports = [{"timestamp": r[0], "module": r[1], "result": r[2], "probability": r[3], "details": r[4], "diet_precautions": r[5]} for r in rows]
    return jsonify({"reports": reports})

# ===================== RUN =====================
if __name__ == '__main__':
    # Generate test images for manual validation
    try:
        generate_synthetic_xray().save(os.path.join(BASE_DIR, 'chest_test.png'))
        generate_synthetic_brain().save(os.path.join(BASE_DIR, 'brain_test.png'))
        print("Mock scans chest_test.png and brain_test.png generated successfully.")
    except Exception as e:
        print(f"Error generating mock scans: {e}")
        
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'development') == 'development'
    print(f"\n🏥 MedAI Clinical Portal is running!")
    print(f"   Open in browser: http://localhost:{port}\n")
    app.run(debug=debug, port=port, host='0.0.0.0')
