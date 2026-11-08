"""
Train 5 classifiers on the UCI "Estimation of Obesity Levels" dataset
and save the fitted pipelines + evaluation metrics.

Run from the project root:
    python model/train_models.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "obesity_raw.csv"
MODEL_DIR = ROOT / "model"
TARGET = "NObeyesdad"
SEED = 74         
TEST_SIZE = 0.25

# Ordered from lightest to heaviest so confusion matrices read naturally
CLASS_ORDER = [
    "Insufficient_Weight", "Normal_Weight", "Overweight_Level_I",
    "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II",
    "Obesity_Type_III",
]


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


CAT_COLS = ["Gender", "family_history_with_overweight", "FAVC", "CAEC",
            "SMOKE", "SCC", "CALC", "MTRANS"]


def build_preprocessor(X):
    cat_cols = [c for c in X.columns if c in CAT_COLS]
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=10),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2,
                                                random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", C=10, gamma="scale",
                                      probability=True, random_state=SEED),
    }


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="weighted",
                             labels=model.classes_),
        "Precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_data()
    print(f"Dataset after removing duplicates: {df.shape}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    # Test data (raw, unencoded) is what gets uploaded to the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    results = {}
    for name, clf in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(pipe, MODEL_DIR / fname)
        print(f"Saved {fname}")

    metrics_df = pd.DataFrame(results).T.round(4)
    print("\n", metrics_df.to_string())
    metrics_df.to_csv(MODEL_DIR / "metrics.csv")

    meta = {"target": TARGET, "class_order": CLASS_ORDER,
            "features": X.columns.tolist(), "seed": SEED, "test_size": TEST_SIZE,
            "n_train": len(X_train), "n_test": len(X_test)}
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
