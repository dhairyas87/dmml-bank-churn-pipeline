#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sqlite3
import os
import logging
from datetime import datetime
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_feature_store(transformed_df: pd.DataFrame, base_path: str):
    """
    Create a feature store with selected engineered + original features,
    metadata, and documentation.
    """
    os.makedirs(base_path, exist_ok=True)
    db_path = os.path.join(base_path, "feature_store.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # --- Create Metadata Table ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feature_metadata (
        feature_name TEXT,
        description TEXT,
        source TEXT,
        version TEXT,
        created_at TEXT
    )
    """)

    # --- Create Engineered Features Table ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS engineered_features (
        CustomerId INTEGER PRIMARY KEY,
        CreditScore REAL,
        Geography TEXT,
        Gender TEXT,
        Age INTEGER,
        Tenure INTEGER,
        Balance REAL,
        NumOfProducts INTEGER,
        HasCrCard INTEGER,
        IsActiveMember INTEGER,
        EstimatedSalary REAL,
        Exited INTEGER,
        AgeGroup TEXT,
        BalanceSalaryRatio REAL,
        CreditScoreBucket TEXT
    )
    """)

    # --- Engineer Features ---
    logging.info("Engineering features for feature store...")

    feature_df = pd.DataFrame()
    feature_df["CustomerId"] = transformed_df["CustomerId"]

    # Keep selected original features
    selected_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]
    feature_df[selected_cols] = transformed_df[selected_cols]

    # Engineered feature: BalanceSalaryRatio
    feature_df["BalanceSalaryRatio"] = (
        transformed_df["Balance"] / (transformed_df["EstimatedSalary"] + 1)
    )

    # Engineered feature: AgeGroup
    feature_df["AgeGroup"] = pd.cut(
        transformed_df["Age"],
        bins=[0, 25, 40, 60, 100],
        labels=["Young", "Adult", "Middle-Aged", "Senior"]
    )

    # Engineered feature: CreditScoreBucket
    feature_df["CreditScoreBucket"] = pd.cut(
        transformed_df["CreditScore"],
        bins=[300, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
    )

    # --- Save Features ---
    feature_df.to_sql("engineered_features", conn, if_exists="replace", index=False)

    # --- Metadata Entries ---
    feature_metadata = [
        ("CreditScore", "Customer credit score", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Geography", "Customer geography (country)", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Gender", "Customer gender", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Age", "Customer age", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Tenure", "Years customer stayed with bank", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Balance", "Customer account balance", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("NumOfProducts", "Number of bank products used by customer", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("HasCrCard", "Whether customer has a credit card (1=yes, 0=no)", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("IsActiveMember", "Whether customer is an active member (1=yes, 0=no)", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("EstimatedSalary", "Customer’s estimated salary", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Exited", "Whether customer exited (1=yes, 0=no)", "Original dataset", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("BalanceSalaryRatio", "Balance divided by salary", "Balance & EstimatedSalary", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("AgeGroup", "Categorized age into groups", "Age", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("CreditScoreBucket", "Bucketed credit score (Poor → Excellent)", "CreditScore", "v1.0", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]

    cursor.executemany("INSERT INTO feature_metadata VALUES (?, ?, ?, ?, ?)", feature_metadata)
    conn.commit()

    # --- Generate Documentation ---
    generate_feature_docs(conn, base_path)

    logging.info(f"Feature store created at {db_path}")
    return conn, db_path


def generate_feature_docs(conn, base_path: str):
    """
    Generate documentation of feature metadata and versions.
    """
    doc_file = os.path.join(base_path, "feature_documentation.md")

    df_meta = pd.read_sql_query("SELECT * FROM feature_metadata", conn)

    with open(doc_file, "w") as f:
        f.write("# 📘 Feature Store Documentation\n\n")
        f.write("This document describes all features available in the feature store, along with their metadata and version history.\n\n")

        for feature in df_meta["feature_name"].unique():
            f.write(f"## 🔹 {feature}\n")
            history = df_meta[df_meta["feature_name"] == feature]

            for _, row in history.iterrows():
                f.write(f"- **Version:** {row['version']}\n")
                f.write(f"  - Description: {row['description']}\n")
                f.write(f"  - Source: {row['source']}\n")
                f.write(f"  - Created At: {row['created_at']}\n\n")

    logging.info(f"Feature documentation generated at {doc_file}")


def sample_feature_queries(conn, base_path: str):
    """
    Run some sample queries to demonstrate feature retrieval.
    Save queries and results in text files in base_path.
    """
    queries = {
        "Show feature metadata": "SELECT * FROM feature_metadata",
        "Retrieve engineered features": "SELECT * FROM engineered_features LIMIT 10",
        "Top customers by BalanceSalaryRatio": 
            "SELECT CustomerId, BalanceSalaryRatio FROM engineered_features ORDER BY BalanceSalaryRatio DESC LIMIT 5",
        "Distribution of CreditScoreBucket": 
            "SELECT CreditScoreBucket, COUNT(*) as Count FROM engineered_features GROUP BY CreditScoreBucket"
    }

    query_file = os.path.join(base_path, "sample_queries.txt")
    result_file = os.path.join(base_path, "query_results.txt")

    with open(query_file, "w") as fq, open(result_file, "w") as fr:
        for title, query in queries.items():
            fq.write(f"-- {title} --\n{query}\n\n")
            fr.write(f"\n--- {title} ---\n")
            result = pd.read_sql_query(query, conn)
            fr.write(result.to_string(index=False))
            fr.write("\n")

    logging.info(f"Sample queries saved at {query_file}")
    logging.info(f"Query results saved at {result_file}")



# In[ ]:





# In[ ]:




