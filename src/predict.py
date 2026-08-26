"""
Stroke Prediction — Inference Pipeline
Author: Isha Sarwar
Date: 02 August 2025

Usage:
    python src/predict.py

This script loads the saved model and makes predictions on new patient data.
"""

import os
import joblib
import pandas as pd
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "preprocessor.pkl")


def load_model():
    """Load the saved model and preprocessor."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run the full analysis notebook first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict(patient_data: dict) -> dict:
    """
    Predict stroke risk for a single patient.

    Parameters
    ----------
    patient_data : dict
        A dictionary with the following keys:
        - gender         : str  — "Male" or "Female"
        - age            : float — Age in years (must be >= 18)
        - hypertension   : int  — 0 or 1
        - heart_disease  : int  — 0 or 1
        - ever_married   : str  — "Yes" or "No"
        - work_type      : str  — "Private", "Self-employed", "Govt_job"
        - Residence_type : str  — "Urban" or "Rural"
        - avg_glucose_level : float — in mg/dL
        - bmi            : float — kg/m²
        - smoking_status : str  — "formerly smoked", "never smoked", "smokes", "Unknown"

    Returns
    -------
    dict with keys:
        - 'stroke_probability' : float (0–1)
        - 'stroke_predicted'   : int (0 or 1)
        - 'risk_level'         : str ("Low", "Moderate", "High")
    """
    model, preprocessor = load_model()

    # Convert to DataFrame
    df = pd.DataFrame([patient_data])

    # Preprocess
    X = preprocessor.transform(df)

    # Predict
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    # Risk level
    if prob < 0.2:
        risk = "Low"
    elif prob < 0.5:
        risk = "Moderate"
    else:
        risk = "High"

    return {
        "stroke_probability": round(float(prob), 4),
        "stroke_predicted": pred,
        "risk_level": risk,
    }


# ─── Example Usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example patient
    example_patient = {
        "gender": "Male",
        "age": 67.0,
        "hypertension": 0,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": "formerly smoked",
    }

    try:
        result = predict(example_patient)
        print("\n" + "=" * 50)
        print("  STROKE RISK PREDICTION")
        print("=" * 50)
        print(f"  Stroke Probability : {result['stroke_probability']:.2%}")
        print(f"  Prediction         : {'⚠ STROKE RISK' if result['stroke_predicted'] else '✓ LOW RISK'}")
        print(f"  Risk Level         : {result['risk_level']}")
        print("=" * 50 + "\n")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
