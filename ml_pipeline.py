"""Shared dataset and candidate-model definitions for DiaLens."""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
MODEL_PATH = BASE_DIR / "brfss_model.pkl"
REPORT_PATH = BASE_DIR / "model_report.json"
TARGET = "Diabetes_binary"
FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]
LABELS = {0: "No diabetes reported", 1: "Diabetes or prediabetes reported"}


def load_dataset():
    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURES + [TARGET]) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")
    X = data[FEATURES].astype("float32")
    y = data[TARGET].astype("int8")
    if data.isna().any().any():
        raise ValueError("The source CSV has missing values; review the preprocessing plan before training.")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Expected a binary Diabetes_binary target with values 0 and 1.")
    return X, y


def make_candidates():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", C=1.0),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=140, min_samples_leaf=4, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=42,
        ),
        "Histogram Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=42,
        ),
    }
