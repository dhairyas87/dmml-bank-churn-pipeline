#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[ ]:


# ml_training.py

import os
import sqlite3
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from datetime import datetime
import subprocess
import json

# ---------------- Feature Loader ----------------
def load_features_from_store(db_path="data/feature_store.db"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM features", conn)
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
def save_version_metadata(version_file="modelbuilding/model_versions.json", notes=""):
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
def run_training(db_path = "featurestore/featurestore/feature_store.db"):
    df = load_features_from_store(db_path)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    os.makedirs("models", exist_ok=True)

    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    results["Logistic Regression"] = evaluate_model(y_test, y_pred_lr)
    with open("models/log_reg.pkl", "wb") as f:
        pickle.dump(log_reg, f)

    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = evaluate_model(y_test, y_pred_rf)
    with open("models/random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)

    # 3. TensorFlow Neural Network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_tf = (model.predict(X_test_scaled) > 0.5).astype("int32").flatten()
    results["TensorFlow NN"] = evaluate_model(y_test, y_pred_tf)
    model.save("models/tensorflow_nn.h5")

    # Save metrics report
    with open("modelbuilding/model_results.txt", "w") as f:
        for model_name, metrics in results.items():
            f.write(f"Model: {model_name}\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

    # Save version metadata
    save_version_metadata(notes="Trained models with Logistic Regression, RF, TensorFlow NN")

    print("✅ Training complete.")
    print("📂 Deliverables: models/ (saved models), data/model_results.txt (report), data/model_versions.json (versions)")


# In[ ]:





# In[ ]:




