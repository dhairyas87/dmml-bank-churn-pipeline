#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import subprocess
import json
from datetime import datetime

# ---------------- Feature Loader ----------------
def load_features_from_store(db_path="featurestore/featurestore/feature_store.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM engineered_features", conn)
    conn.close()
    return df

# ---------------- Evaluation ----------------
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

# ---------------- Save Git Version Metadata ----------------
def save_version_metadata(version_file="models/model_versions.json", notes=""):
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    entry = {
        "commit_id": commit_id,
        "timestamp": timestamp,
        "notes": notes
    }
    metadata.append(entry)

    with open(version_file, "w") as f:
        json.dump(metadata, f, indent=4)

# ---------------- Training ----------------
def run_training(db_path="featurestore/featurestore/feature_store.db"):
    df = load_features_from_store(db_path)
    os.makedirs("models", exist_ok=True)
    # Drop ID column
    df = df.drop(columns=["CustomerId"])

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    # Categorical & numeric features
    categorical_cols = ["Geography", "Gender", "AgeGroup", "CreditScoreBucket"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    results = {}
    

    # Logistic Regression
    pipe_lr = Pipeline(steps=[("preprocessor", preprocessor),
                              ("classifier", LogisticRegression(max_iter=1000))])
    pipe_lr.fit(X, y)
    y_pred_lr = pipe_lr.predict(X)
    results["Logistic Regression"] = evaluate_model(y, y_pred_lr)
    with open("models/log_reg.pkl", "wb") as f:
        pickle.dump(pipe_lr, f)

    # Random Forest
    pipe_rf = Pipeline(steps=[("preprocessor", preprocessor),
                              ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])
    pipe_rf.fit(X, y)
    y_pred_rf = pipe_rf.predict(X)
    results["Random Forest"] = evaluate_model(y, y_pred_rf)
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(pipe_rf, f)

    # SVM
    pipe_svm = Pipeline(steps=[("preprocessor", preprocessor),
                               ("classifier", SVC(probability=True))])
    pipe_svm.fit(X, y)
    y_pred_svm = pipe_svm.predict(X)
    results["SVM"] = evaluate_model(y, y_pred_svm)
    with open("models/svm.pkl", "wb") as f:
        pickle.dump(pipe_svm, f)

    # Save metrics report
    with open("models/model_results.txt", "w") as f:
        for model_name, metrics in results.items():
            f.write(f"Model: {model_name}\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

    save_version_metadata(notes="Trained models using engineered features (LR, RF, SVM)")

    print("✅ Training complete with engineered features.")
    print("📂 Deliverables: models/, data/model_results.txt, data/model_versions.json")


# In[ ]:




