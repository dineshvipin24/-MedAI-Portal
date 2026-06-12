import os
import joblib
import pandas as pd
import numpy as np

def test_heart_pipeline():
    print("\n🩺 Testing CardioShield (Heart Disease) Pipeline...")
    BASE_DIR = os.path.dirname(__file__)
    
    model = joblib.load(os.path.join(BASE_DIR, "KNN_heart.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(BASE_DIR, "columns.pkl"))
    
    print("✅ Heart Disease assets loaded.")
    
    # Mock Low Risk Patient
    low_risk = pd.DataFrame([{
        'Age': 30, 'RestingBP': 110, 'Cholesterol': 170, 'FastingBS': 0, 'MaxHR': 180, 'Oldpeak': 0.0,
        'Sex_M': 0, 'ChestPainType_ATA': 1, 'ChestPainType_NAP': 0, 'ChestPainType_TA': 0,
        'RestingECG_Normal': 1, 'RestingECG_ST': 0, 'ExerciseAngina_Y': 0, 'ST_Slope_Flat': 0, 'ST_Slope_Up': 1
    }])[feature_columns]
    
    # Mock High Risk Patient
    high_risk = pd.DataFrame([{
        'Age': 65, 'RestingBP': 160, 'Cholesterol': 320, 'FastingBS': 1, 'MaxHR': 110, 'Oldpeak': 3.5,
        'Sex_M': 1, 'ChestPainType_ATA': 0, 'ChestPainType_NAP': 0, 'ChestPainType_TA': 0,
        'RestingECG_Normal': 0, 'RestingECG_ST': 1, 'ExerciseAngina_Y': 1, 'ST_Slope_Flat': 1, 'ST_Slope_Up': 0
    }])[feature_columns]
    
    low_scaled = scaler.transform(low_risk)
    high_scaled = scaler.transform(high_risk)
    
    assert model.predict(low_scaled)[0] == 0, "Low-risk heart patient failed check"
    assert model.predict(high_scaled)[0] == 1, "High-risk heart patient failed check"
    print("🎉 CardioShield pipeline passed all tests!")

def test_diabetes_pipeline():
    print("\n🩸 Testing DiaPredict (Diabetes) Pipeline...")
    BASE_DIR = os.path.dirname(__file__)
    
    model_path = os.path.join(BASE_DIR, "diabetes_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "diabetes_scaler.pkl")
    columns_path = os.path.join(BASE_DIR, "diabetes_columns.pkl")
    
    # Check if files exist before loading (in case user hasn't run the training script yet)
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(columns_path)):
        print("⚠️ Diabetes model files not found. Run 'train_diabetes.py' to generate them.")
        return
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(columns_path)
    
    print("✅ Diabetes assets loaded.")
    
    # Mock Healthy Patient (Low glucose, low insulin, low BMI, young)
    healthy = pd.DataFrame([{
        'Pregnancies': 1, 'Glucose': 95, 'BloodPressure': 70, 'SkinThickness': 18,
        'Insulin': 65, 'BMI': 22.5, 'DiabetesPedigreeFunction': 0.250, 'Age': 24
    }])[feature_columns]
    
    # Mock Diabetic Patient (High glucose, high insulin, high BMI, older)
    diabetic = pd.DataFrame([{
        'Pregnancies': 6, 'Glucose': 175, 'BloodPressure': 85, 'SkinThickness': 35,
        'Insulin': 210, 'BMI': 38.2, 'DiabetesPedigreeFunction': 0.750, 'Age': 48
    }])[feature_columns]
    
    healthy_scaled = scaler.transform(healthy)
    diabetic_scaled = scaler.transform(diabetic)
    
    assert model.predict(healthy_scaled)[0] == 0, "Healthy patient failed check"
    assert model.predict(diabetic_scaled)[0] == 1, "Diabetic patient failed check"
    print("🎉 DiaPredict pipeline passed all tests!")

if __name__ == "__main__":
    print("🚀 Starting local pipeline tests...")
    test_heart_pipeline()
    test_diabetes_pipeline()
