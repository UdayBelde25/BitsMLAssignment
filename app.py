"""
ObesityLens - Streamlit demo for ML Assignment 2 (2025NS04037)
Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
BITS_ID = "2025NS04037"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}
SHORT_LABELS = {
    "Insufficient_Weight": "Under", "Normal_Weight": "Normal",
    "Overweight_Level_I": "Over-I", "Overweight_Level_II": "Over-II",
    "Obesity_Type_I": "Obese-I", "Obesity_Type_II": "Obese-II",
    "Obesity_Type_III": "Obese-III",
}

st.set_page_config(page_title=f"{BITS_ID} ML Assignment", page_icon="🔍", layout="wide")


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


def footer():
    st.divider()
    st.caption(f"ObesityLens · ML Assignment 2 · BITS ID: **{BITS_ID}**")


models = load_models()
meta = load_meta()
TARGET, CLASS_ORDER, FEATURES = meta["target"], meta["class_order"], meta["features"]

# ---------------- Header ----------------
st.title(f"{BITS_ID} ML Assignment")
st.subheader("🔍 ObesityLens")
st.write("Predict a person's obesity level from eating habits and physical condition, "
         "and compare how five classical ML models perform.")

with st.expander("ℹ️ About the dataset"):
    st.markdown(
        "**Estimation of Obesity Levels Based on Eating Habits and Physical Condition** "
        "(UCI Machine Learning Repository, ID 544)\n\n"
        "- **2,111 records** from Mexico, Peru and Colombia (2,087 after removing duplicates). "
        "About 23% were collected through a web survey; the rest were generated with SMOTE.\n"
        "- **16 features**: Gender, Age, Height, Weight, family history of overweight, "
        "eating habits (FAVC, FCVC, NCP, CAEC, CH2O, CALC), lifestyle (SMOKE, SCC, FAF, TUE) "
        "and mode of transport (MTRANS).\n"
        "- **Target `NObeyesdad`** with 7 classes: Insufficient Weight, Normal Weight, "
        "Overweight I & II, Obesity Type I, II & III.\n"
        f"- Models were trained on 75% of the data; the remaining {meta['n_test']} rows form "
        "the test set that you can upload below."
    )

# ---------------- Controls (top of page) ----------------
c1, c2, c3 = st.columns([2, 1.3, 1])
with c1:
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
with c2:
    model_name = st.selectbox("Choose a classifier", list(MODEL_FILES))
with c3:
    sample = ROOT / "test_data.csv"
    if sample.exists():
        st.write("")
        st.download_button("⬇️ Sample test_data.csv", sample.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

if uploaded is None:
    st.info("Upload a CSV above to get started. "
            "You can download the sample test file if you need one.")
    footer()
    st.stop()

df = pd.read_csv(uploaded)
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    footer()
    st.stop()

X = df[FEATURES]
has_labels = TARGET in df.columns

with st.expander(f"Preview uploaded data ({df.shape[0]} rows × {df.shape[1]} columns)"):
    st.dataframe(df.head(20), width="stretch")

model = models[model_name]

if not has_labels:
    st.warning(f"No '{TARGET}' column found, so only predictions are shown.")
    out = df.copy()
    out["Predicted"] = model.predict(X)
    st.dataframe(out, width="stretch")
    footer()
    st.stop()

y = df[TARGET]
y_pred, m = compute_metrics(model, X, y)

tab1, tab2, tab3 = st.tabs(["📊 Selected model", "🏆 Compare all models", "🔎 Predictions"])

with tab1:
    st.subheader(f"{model_name}: evaluation on uploaded data")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Confusion matrix**")
        labels = [c for c in CLASS_ORDER if c in set(y) | set(y_pred)]
        cm = confusion_matrix(y, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax,
                    xticklabels=[SHORT_LABELS[l] for l in labels],
                    yticklabels=[SHORT_LABELS[l] for l in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

with tab2:
    st.subheader("All five models on the same data")
    rows = {name: compute_metrics(mdl, X, y)[1] for name, mdl in models.items()}
    comp = pd.DataFrame(rows).T
    best = comp["F1"].idxmax()
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    st.success(f"Best model by F1 score: **{best}** (F1 = {comp.loc[best, 'F1']:.4f})")
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])

with tab3:
    out = df.copy()
    out["Predicted"] = y_pred
    out["Correct"] = out[TARGET] == out["Predicted"]
    st.write(f"{out['Correct'].sum()} of {len(out)} rows predicted correctly.")
    show_wrong = st.checkbox("Show only misclassified rows")
    st.dataframe(out[~out["Correct"]] if show_wrong else out, width="stretch")

footer()
