"""
ObesityLens - Streamlit demo for ML Assignment 2 (2025NS04037)
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(page_title="2025NS04037 ML Assignment", page_icon="🔍", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f) for name, f in MODEL_FILES.items()}


@st.cache_data
def load_meta():
    return json.loads((MODEL_DIR / "meta.json").read_text())


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    try:
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted",
                            labels=model.classes_)
    except ValueError:      # e.g. uploaded file has only one class
        auc = float("nan")
    return y_pred, {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


models = load_models()
meta = load_meta()
TARGET, FEATURES = meta["target"], meta["features"]

st.title("2025NS04037 ML Assignment")
st.subheader("🔍 ObesityLens")

c1, c2 = st.columns([2, 1.3])
with c1:
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
with c2:
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))

if uploaded is None:
    st.info("Upload a CSV above to get started.")
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

st.dataframe(df.head(10), width="stretch")

_, m = compute_metrics(models[model_name], df[FEATURES], df[TARGET])
st.subheader(f"{model_name}: evaluation metrics")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.3f}")
