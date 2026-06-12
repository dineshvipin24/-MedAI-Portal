import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train_chest_model():
    print("🧪 Training Thoracic Diagnostic Classifier (Chest X-Ray)...")
    np.random.seed(42)
    n_samples = 1500
    
    # Target label distribution: 0 = healthy_chest, 1 = pneumonia, 2 = cardiomegaly
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
    
    # 1. lung_mean (overall lung density/opacity)
    # Pneumonia has white consolidation (mean 75-95), healthy and cardiomegaly are darker (mean 40-65)
    lung_mean = np.zeros(n_samples)
    lung_mean[y == 0] = np.random.normal(loc=52.0, scale=7.0, size=np.sum(y == 0))
    lung_mean[y == 1] = np.random.normal(loc=82.0, scale=8.0, size=np.sum(y == 1))
    lung_mean[y == 2] = np.random.normal(loc=55.0, scale=7.0, size=np.sum(y == 2))
    lung_mean = np.clip(lung_mean, 10.0, 120.0)
    
    # 2. lung_asymmetry (difference between left and right lung density)
    # Pneumonia (unilateral infiltration) can have high asymmetry (10-30), healthy and cardiomegaly are symmetric (0-6)
    lung_asymmetry = np.zeros(n_samples)
    lung_asymmetry[y == 0] = np.random.normal(loc=2.5, scale=1.5, size=np.sum(y == 0))
    # Some pneumonia is bilateral (low asymmetry) and some is unilateral (high asymmetry)
    pneu_size = np.sum(y == 1)
    lung_asymmetry[y == 1] = np.random.choice([
        np.random.normal(loc=3.0, scale=1.5, size=pneu_size),
        np.random.normal(loc=18.0, scale=5.0, size=pneu_size)
    ], size=pneu_size)[0]
    lung_asymmetry[y == 2] = np.random.normal(loc=2.8, scale=1.5, size=np.sum(y == 2))
    lung_asymmetry = np.clip(lung_asymmetry, 0.0, 50.0)
    
    # 3. heart_shadow (width of cardiac silhouette)
    # Cardiomegaly has high heart shadow width (35-50), healthy and pneumonia are normal (18-32)
    heart_shadow = np.zeros(n_samples)
    heart_shadow[y == 0] = np.random.normal(loc=26.0, scale=3.0, size=np.sum(y == 0))
    heart_shadow[y == 1] = np.random.normal(loc=27.0, scale=3.0, size=np.sum(y == 1))
    heart_shadow[y == 2] = np.random.normal(loc=42.0, scale=4.0, size=np.sum(y == 2))
    heart_shadow = np.clip(heart_shadow, 10.0, 60.0)
    
    df = pd.DataFrame({
        'lung_mean': lung_mean,
        'lung_asymmetry': lung_asymmetry,
        'heart_shadow': heart_shadow
    })
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Chest Model Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['healthy_chest', 'pneumonia', 'cardiomegaly']))
    
    # Save assets
    joblib.dump(model, os.path.join(BASE_DIR, "chest_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "chest_scaler.pkl"))
    joblib.dump(df.columns.tolist(), os.path.join(BASE_DIR, "chest_columns.pkl"))
    print("✅ Chest Model assets saved successfully!\n")

def train_brain_model():
    print("🧪 Training Neural Diagnostic Classifier (Brain MRI/CT)...")
    np.random.seed(42)
    n_samples = 1500
    
    # Target label distribution: 0 = healthy_brain, 1 = tumor, 2 = stroke
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.3, 0.3])
    
    # 1. bright_ratio (max bright ratio of hemispheres)
    # Tumor has high bright ratio (0.07 - 0.18), healthy/stroke are lower (0.01 - 0.05)
    bright_ratio = np.zeros(n_samples)
    bright_ratio[y == 0] = np.random.normal(loc=0.03, scale=0.01, size=np.sum(y == 0))
    bright_ratio[y == 1] = np.random.normal(loc=0.12, scale=0.03, size=np.sum(y == 1))
    bright_ratio[y == 2] = np.random.normal(loc=0.04, scale=0.012, size=np.sum(y == 2))
    bright_ratio = np.clip(bright_ratio, 0.0, 0.3)
    
    # 2. bright_diff_ratio (asymmetry of bright regions)
    # Tumor has high asymmetry (0.08 - 0.22), healthy/stroke are symmetric (0.0 - 0.04)
    bright_diff_ratio = np.zeros(n_samples)
    bright_diff_ratio[y == 0] = np.random.normal(loc=0.015, scale=0.008, size=np.sum(y == 0))
    bright_diff_ratio[y == 1] = np.random.normal(loc=0.14, scale=0.03, size=np.sum(y == 1))
    bright_diff_ratio[y == 2] = np.random.normal(loc=0.02, scale=0.01, size=np.sum(y == 2))
    bright_diff_ratio = np.clip(bright_diff_ratio, 0.0, 0.3)
    
    # 3. dark_diff_ratio (asymmetry of dark regions / stroke cytotoxic edema)
    # Stroke has high dark asymmetry (0.18 - 0.38), healthy/tumor are lower (0.01 - 0.10)
    dark_diff_ratio = np.zeros(n_samples)
    dark_diff_ratio[y == 0] = np.random.normal(loc=0.05, scale=0.03, size=np.sum(y == 0))
    dark_diff_ratio[y == 1] = np.random.normal(loc=0.06, scale=0.03, size=np.sum(y == 1))
    dark_diff_ratio[y == 2] = np.random.normal(loc=0.26, scale=0.06, size=np.sum(y == 2))
    dark_diff_ratio = np.clip(dark_diff_ratio, 0.0, 0.5)
    
    df = pd.DataFrame({
        'bright_ratio': bright_ratio,
        'bright_diff_ratio': bright_diff_ratio,
        'dark_diff_ratio': dark_diff_ratio
    })
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.20, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Brain Model Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['healthy_brain', 'tumor', 'stroke']))
    
    # Save assets
    joblib.dump(model, os.path.join(BASE_DIR, "brain_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "brain_scaler.pkl"))
    joblib.dump(df.columns.tolist(), os.path.join(BASE_DIR, "brain_columns.pkl"))
    print("✅ Brain Model assets saved successfully!\n")

if __name__ == "__main__":
    train_chest_model()
    train_brain_model()
    print("🎉 All medical vision model assets generated and saved!")
