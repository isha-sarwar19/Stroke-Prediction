# 🧠 Stroke Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20beff?logo=kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](notebooks/stroke_prediction_analysis.ipynb)

## 📌 Overview

Stroke is the **second leading cause of death globally**, responsible for over 6.5 million deaths annually. Early identification of high-risk individuals can significantly reduce stroke-related mortality and long-term disability.

This project applies **machine learning classification** techniques to the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) to build a model that predicts stroke risk based on demographic and clinical indicators.

---

## 📂 Project Structure

```
Stroke_Prediction/
├── data/
│   └── raw/
│       └── healthcare-dataset-stroke-data.csv   # Raw dataset (Kaggle)
│
├── notebooks/
│   └── stroke_prediction_analysis.ipynb         # Main analysis notebook
│
├── results/
│   └── bmi_imputation_validation.csv            # BMI imputation validation stats
│
├── figures/                                      # Exported plots & visualizations
├── models/                                       # Saved trained models (.pkl)
├── src/
│   └── predict.py                               # Prediction pipeline script
│
├── requirements.txt                             # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) |
| **Total Records** | 5,110 patients |
| **Features** | 11 (after dropping ID) |
| **Target** | `stroke` (0 = No, 1 = Yes) |
| **Class Imbalance** | ~4.87% positive (stroke) cases |
| **Missing Values** | 201 missing BMI values |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Male / Female |
| `age` | Numerical | Patient age (years) |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `ever_married` | Categorical | Yes / No |
| `work_type` | Categorical | Private, Self-employed, Govt\_job, Children |
| `Residence_type` | Categorical | Urban / Rural |
| `avg_glucose_level` | Numerical | Average blood glucose (mg/dL) |
| `bmi` | Numerical | Body Mass Index (kg/m²) |
| `smoking_status` | Categorical | formerly smoked, never smoked, smokes, Unknown |
| `stroke` | Binary | **Target** — 0 = No Stroke, 1 = Stroke |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Statistical summary and data quality checks
- **Pediatric case removal**: Excluded 856 patients aged < 18 (only 2 stroke cases; pediatric stroke is biologically distinct)
- Univariate and bivariate analysis of all features
- Handling of "Unknown" smoking status

### 2. Data Preprocessing
- **BMI Imputation**: Mean imputation validated using KS test, t-test, and Cohen's d
- **Outlier Analysis**: IQR-based detection; compared log, square-root, Winsorization transformations
- **Categorical Encoding**: One-Hot Encoding for multi-class, Label Encoding for binary
- **Feature Scaling**: StandardScaler for numeric features

### 3. Class Imbalance Handling
Severe class imbalance (~95% No Stroke vs ~5% Stroke) addressed via:
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Class-weight adjustment** in models
- **Threshold tuning**

### 4. Models Evaluated
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline; best with class weights + SMOTE |
| Decision Tree | Prone to overfitting |
| Random Forest | Good out-of-the-box performance |
| SVM | Effective on scaled data |
| K-Nearest Neighbors | Sensitive to imbalance |
| XGBoost | Strong ensemble method |
| LightGBM | Fast gradient boosting |

### 5. Evaluation Metrics
Given class imbalance, primary metrics:
- **AUC-ROC** (discrimination ability)
- **Recall/Sensitivity** (minimize false negatives — missing stroke cases is costly)
- **Precision-Recall AUC**
- F1-Score, Accuracy

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stroke-prediction.git
cd stroke-prediction

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Running the Analysis

Open `notebooks/stroke_prediction_analysis.ipynb` and run all cells in order.

### Making Predictions

```bash
python src/predict.py
```

---

## 🗒️ License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Isha Sarwar**  
Date: 02 August 2025

---

## 🙏 Acknowledgements

- Dataset: [fedesoriano on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)
- [World Stroke Organization](https://www.world-stroke.org/) for stroke statistics
