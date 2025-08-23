#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sqlite3
import pandas as pd
import logging
import os
from datetime import datetime
from tabulate import tabulate
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def transform_and_store(df: pd.DataFrame, output_dir="transformation_reports", db_name="churn_transformed.db"):
    """
    Perform feature engineering transformations, drop irrelevant fields, 
    and store results in SQLite DB with schema + transformation summary + sample queries + query outputs.
    """

    logging.info("Starting data transformation...")

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Drop irrelevant columns ---
    drop_cols = ["RowNumber", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    logging.info(f"Dropped columns: {drop_cols}")

    # --- Feature Engineering ---
    logging.info("Creating new features...")
    transformation_summary = []

    if "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"], bins=[-1, 25, 40, 60, 100],
            labels=["Young", "Adult", "Mature", "Senior"]
        )
        transformation_summary.append("Created AgeGroup from Age (bins: <25, 25-40, 40-60, >60).")

    if "Balance" in df.columns and "EstimatedSalary" in df.columns:
        df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
        transformation_summary.append("Created BalanceSalaryRatio = Balance / (EstimatedSalary + 1).")

    if "CreditScore" in df.columns:
        df["CreditScoreBucket"] = pd.cut(
            df["CreditScore"], bins=[0, 500, 700, 850],
            labels=["Low", "Medium", "High"]
        )
        transformation_summary.append("Created CreditScoreBucket from CreditScore (Low <500, Medium 500-700, High >700).")

    logging.info("Feature engineering completed.")

    # --- Store in SQLite ---
    db_path = os.path.join(output_dir, db_name)
    conn = sqlite3.connect(db_path)
    df.to_sql("transformed_churn", conn, if_exists="replace", index=False)
    conn.commit()
    logging.info(f"Data successfully stored in SQLite: {db_path}")

    # --- Save schema design ---
    schema_file = os.path.join(output_dir, f"schema_design_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql")
    with open(schema_file, "w") as f:
        f.write("CREATE TABLE transformed_churn (\n")
        for col, dtype in zip(df.columns, df.dtypes):
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = "INTEGER"
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = "REAL"
            else:
                sql_type = "TEXT"
            f.write(f"    {col} {sql_type},\n")
        f.write(");\n")
    logging.info(f"SQL schema design saved at {schema_file}")

    # --- Sample Queries ---
    sample_queries = {
        "Preview data": "SELECT * FROM transformed_churn LIMIT 10;",
        "Customer distribution by AgeGroup": "SELECT AgeGroup, COUNT(*) AS Customers FROM transformed_churn GROUP BY AgeGroup;",
        "Balance-Salary ratio by CreditScoreBucket": "SELECT CreditScoreBucket, AVG(BalanceSalaryRatio) FROM transformed_churn GROUP BY CreditScoreBucket;",
        "Average CreditScore by Active Member": "SELECT IsActiveMember, AVG(CreditScore) FROM transformed_churn GROUP BY IsActiveMember;",
        "Churned customers by Geography": "SELECT Geography, SUM(Exited) AS Churned_Customers FROM transformed_churn GROUP BY Geography;"
    }

    # --- Save Queries into a File ---
    queries_file = os.path.join(output_dir, "sample_queries.sql")
    with open(queries_file, "w") as f:
        f.write("-- === Sample SQL Queries === --\n\n")
        for title, query in sample_queries.items():
            f.write(f"-- {title}\n{query}\n\n")
    logging.info(f"Sample SQL queries saved at {queries_file}")

    # --- Execute Queries and Save Results ---
    query_output_file = os.path.join(output_dir, "sample_query_outputs.txt")
    with open(query_output_file, "w") as f:
        f.write("=== Sample Query Outputs ===\n\n")
        for title, query in sample_queries.items():
            f.write(f"--- {title} ---\n")
            try:
                result = pd.read_sql_query(query, conn)

                # Use tabulate for table-style formatting
                formatted = tabulate(result, headers="keys", tablefmt="grid", showindex=False)
                f.write(formatted)
                f.write("\n\n")
            except Exception as e:
                f.write(f"Error executing query: {e}\n\n")

    logging.info(f"Query outputs saved at {query_output_file}")

    # --- Save Transformation Summary ---
    summary_file = os.path.join(output_dir, "transformation_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=== Transformation Logic Applied ===\n")
        for step in transformation_summary:
            f.write(f"- {step}\n")
    logging.info(f"Transformation summary saved at {summary_file}")

    conn.close()
    return df


# In[ ]:




