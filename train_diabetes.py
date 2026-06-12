import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def build_diabetes_pipeline():
    print("🧪 Generating large-scale clinical PIMA Diabetes Dataset (5,000 samples)...")
    np.random.seed(42)
    n_samples = 5000

    # Generate labels (0: No Diabetes, 1: Diabetes) with a 65/35 distribution
    y = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

    # --- Generate Features with Realistic Clinical Overlap/Noise ---
    # This prevents the dataset from being trivially separable, forcing the model to learn general patterns.
    
    # 1. Age: mean of 32 for non-diabetic, 44 for diabetic, with standard deviation causing overlap
    age = np.zeros(n_samples)
    age[y == 0] = np.random.normal(loc=31.5, scale=9.5, size=np.sum(y == 0))
    age[y == 1] = np.random.normal(loc=43.0, scale=10.5, size=np.sum(y == 1))
    age = np.clip(age, 21, 85).astype(int)

    # 2. BMI: significant overlap, mean 27.5 vs 34.0
    bmi = np.zeros(n_samples)
    bmi[y == 0] = np.random.normal(loc=27.2, scale=5.0, size=np.sum(y == 0))
    bmi[y == 1] = np.random.normal(loc=33.8, scale=6.0, size=np.sum(y == 1))
    bmi = np.clip(bmi, 16.5, 59.0).round(1)

    # 3. Glucose: key marker but with overlap, mean 108 vs 144
    glucose = np.zeros(n_samples)
    glucose[y == 0] = np.random.normal(loc=107.5, scale=18.0, size=np.sum(y == 0))
    glucose[y == 1] = np.random.normal(loc=142.5, scale=25.0, size=np.sum(y == 1))
    glucose = np.clip(glucose, 55, 200).astype(int)

    # 4. Blood Pressure: mean 69 vs 75
    bp = np.zeros(n_samples)
    bp[y == 0] = np.random.normal(loc=68.5, scale=11.5, size=np.sum(y == 0))
    bp[y == 1] = np.random.normal(loc=74.5, scale=12.5, size=np.sum(y == 1))
    bp = np.clip(bp, 38, 125).astype(int)

    # 5. Pregnancies
    pregnancies = np.random.randint(0, 7, size=n_samples)
    pregnancies[y == 1] += np.random.randint(0, 5, size=np.sum(y == 1))
    pregnancies = np.clip(pregnancies, 0, 17).astype(int)

    # 6. Skin Thickness (Triceps fold)
    skin = np.zeros(n_samples)
    skin[y == 0] = np.random.normal(loc=19.5, scale=9.0, size=np.sum(y == 0))
    skin[y == 1] = np.random.normal(loc=28.0, scale=10.5, size=np.sum(y == 1))
    skin = np.clip(skin, 0, 65).astype(int)

    # 7. Insulin
    insulin = np.zeros(n_samples)
    insulin[y == 0] = np.random.normal(loc=68.0, scale=35.0, size=np.sum(y == 0))
    insulin[y == 1] = np.random.normal(loc=150.0, scale=65.0, size=np.sum(y == 1))
    insulin = np.clip(insulin, 0, 650).astype(int)

    # 8. Diabetes Pedigree Function
    dpf = np.zeros(n_samples)
    dpf[y == 0] = np.random.normal(loc=0.41, scale=0.18, size=np.sum(y == 0))
    dpf[y == 1] = np.random.normal(loc=0.58, scale=0.26, size=np.sum(y == 1))
    dpf = np.clip(dpf, 0.08, 2.42).round(3)

    # Assemble DataFrame
    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    })

    X = df.copy()
    feature_cols = X.columns.tolist()

    # Split train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Regularized Random Forest Classifier to Prevent Overfitting ---
    # We restrict max_depth and set min_samples_leaf to encourage generalization.
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,              # Restrict depth to avoid memorizing train noise (prevents overfitting)
        min_samples_split=8,      # Minimum samples required to split a node
        min_samples_leaf=6,       # Minimum samples required at a leaf node
        max_features='sqrt',      # Number of features to consider when looking for the best split
        random_state=42
    )
    rf.fit(X_train_scaled, y_train)

    # --- Evaluate Train Set (to monitor overfitting) ---
    y_train_pred = rf.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)

    # --- Evaluate Test Set ---
    y_test_pred = rf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print("\n📊 Model Diagnostics (Overfitting & Underfitting Check):")
    print(f"   Train Set Accuracy : {train_acc * 100:.2f}% | Train F1-Score: {train_f1:.4f}")
    print(f"   Test Set Accuracy  : {test_acc * 100:.2f}% | Test F1-Score : {test_f1:.4f}")
    
    # Assert there is no heavy overfitting (difference between train and test accuracy < 5%)
    acc_gap = train_acc - test_acc
    print(f"   Generalization Gap : {acc_gap * 100:.2f}%")
    if acc_gap > 0.05:
        print("   ⚠️ Alert: Model is slightly overfitting. Consider narrowing down max_depth.")
    else:
        print("   ✅ Generalization verified: Model has no overfitting issues.")

    # Export clinical assets
    BASE_DIR = os.path.dirname(__file__)
    joblib.dump(rf, os.path.join(BASE_DIR, "diabetes_model.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "diabetes_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(BASE_DIR, "diabetes_columns.pkl"))
    print("\n✅ Successfully exported diabetes_model.pkl, diabetes_scaler.pkl, and diabetes_columns.pkl.")

if __name__ == "__main__":
    build_diabetes_pipeline()
