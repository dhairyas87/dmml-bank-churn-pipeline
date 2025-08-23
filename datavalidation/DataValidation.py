#!/usr/bin/env python
# coding: utf-8

# In[6]:


import logging
import pandas as pd
from datetime import datetime
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def validate_churn_data(df: pd.DataFrame, output_dir="reports", fmt="csv"):
    """
    Validate churn dataset with anomaly checks + general data quality metrics.
    """
    logging.info("===== Validation started =====")

    issues = []

    # --- Rule-based checks ---
    logging.info("Starting check: RowNumber sequence")
    if not df["RowNumber"].is_monotonic_increasing:
        issues.append(["RowNumber", "Sequence", "RowNumber not strictly increasing"])
        logging.warning("RowNumber anomalies detected")
    logging.info("Completed check: RowNumber sequence")

    logging.info("Starting check: CustomerId uniqueness")
    if df["CustomerId"].duplicated().any():
        dupes = df["CustomerId"].duplicated().sum()
        issues.append(["CustomerId", "Uniqueness", f"{dupes} duplicates found"])
        logging.warning(f"CustomerId anomalies detected: {dupes} duplicates")
    logging.info("Completed check: CustomerId uniqueness")

    logging.info("Starting check: CreditScore validity")
    bad_credit = df[(df["CreditScore"] < 300) | (df["CreditScore"] > 850)]
    if not bad_credit.empty:
        issues.append(["CreditScore", "Range", f"{len(bad_credit)} values outside 300–850"])
        logging.warning(f"CreditScore anomalies detected: {len(bad_credit)} rows")
    logging.info("Completed check: CreditScore validity")

    logging.info("Starting check: Geography validity")
    valid_geo = {"France", "Spain", "Germany"}
    bad_geo = df[~df["Geography"].isin(valid_geo)]
    if not bad_geo.empty:
        issues.append(["Geography", "Invalid", f"{len(bad_geo)} invalid values"])
        logging.warning(f"Geography anomalies detected: {len(bad_geo)} rows")
    logging.info("Completed check: Geography validity")

    logging.info("Starting check: Gender validity")
    valid_gender = {"Male", "Female"}
    bad_gender = df[~df["Gender"].isin(valid_gender)]
    if not bad_gender.empty:
        issues.append(["Gender", "Invalid", f"{len(bad_gender)} invalid values"])
        logging.warning(f"Gender anomalies detected: {len(bad_gender)} rows")
    logging.info("Completed check: Gender validity")

    logging.info("Starting check: Age range")
    bad_age = df[(df["Age"] < 18) | (df["Age"] > 100)]
    if not bad_age.empty:
        issues.append(["Age", "Range", f"{len(bad_age)} invalid ages"])
        logging.warning(f"Age anomalies detected: {len(bad_age)} rows")
    logging.info("Completed check: Age range")

    logging.info("Starting check: Tenure range")
    bad_tenure = df[(df["Tenure"] < 0) | (df["Tenure"] > 10)]
    if not bad_tenure.empty:
        issues.append(["Tenure", "Range", f"{len(bad_tenure)} invalid tenures"])
        logging.warning(f"Tenure anomalies detected: {len(bad_tenure)} rows")
    logging.info("Completed check: Tenure range")

    logging.info("Starting check: Balance non-negative")
    bad_balance = df[df["Balance"] < 0]
    if not bad_balance.empty:
        issues.append(["Balance", "Negative", f"{len(bad_balance)} negative balances"])
        logging.warning(f"Balance anomalies detected: {len(bad_balance)} rows")
    logging.info("Completed check: Balance non-negative")

    logging.info("Starting check: NumOfProducts range")
    bad_products = df[(df["NumOfProducts"] < 1) | (df["NumOfProducts"] > 4)]
    if not bad_products.empty:
        issues.append(["NumOfProducts", "Range", f"{len(bad_products)} invalid values"])
        logging.warning(f"NumOfProducts anomalies detected: {len(bad_products)} rows")
    logging.info("Completed check: NumOfProducts range")

    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        logging.info(f"Starting check: {col} binary values")
        bad_binary = df[~df[col].isin([0, 1])]
        if not bad_binary.empty:
            issues.append([col, "Binary", f"{len(bad_binary)} invalid binary values"])
            logging.warning(f"{col} anomalies detected: {len(bad_binary)} rows")
        logging.info(f"Completed check: {col} binary values")

    logging.info("Starting check: EstimatedSalary outliers")
    high_salary = df[df["EstimatedSalary"] > df["EstimatedSalary"].quantile(0.999)]
    if not high_salary.empty:
        issues.append(["EstimatedSalary", "Anomaly", f"{len(high_salary)} extreme outliers"])
        logging.warning(f"EstimatedSalary anomalies detected: {len(high_salary)} rows")
    logging.info("Completed check: EstimatedSalary outliers")

    # Convert issues to DataFrame
    issues_df = pd.DataFrame(issues, columns=["Column", "IssueType", "Details"])

    # --- General dataset quality metrics ---
    report = {}
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicate_rows"] = int(df.duplicated().sum())
    report["data_types"] = df.dtypes.apply(lambda x: str(x)).to_dict()
    report["numeric_summary"] = df.describe(include=["number"]).to_dict()

    # Save reports
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save issues
    issues_file = os.path.join(output_dir, f"churn_data_issues_{timestamp}.csv")
    issues_df.to_csv(issues_file, index=False)

    # Save general report as CSV
    meta_file = os.path.join(output_dir, f"churn_data_metadata_{timestamp}.csv")
    pd.DataFrame([report]).to_csv(meta_file, index=False)

    # Save general report as PDF (nicer format)
    pdf_file = os.path.join(output_dir, f"churn_data_report_{timestamp}.pdf")
    c = canvas.Canvas(pdf_file, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(30, 750, "Churn Data Quality Report")
    c.setFont("Helvetica", 10)

    y = 730
    for section, details in report.items():
        c.drawString(30, y, f"{section.upper()}:")
        y -= 15
        if isinstance(details, dict):
            for k, v in details.items():
                c.drawString(50, y, f"{k}: {v}")
                y -= 12
        else:
            c.drawString(50, y, str(details))
            y -= 12
        y -= 8
    c.save()

    logging.info(f"Issues report saved at {issues_file}")
    logging.info(f"Metadata report saved at {meta_file}")
    logging.info(f"PDF report saved at {pdf_file}")
    logging.info("===== Validation completed =====")

    return issues_df, report


# In[ ]:





# In[ ]:




