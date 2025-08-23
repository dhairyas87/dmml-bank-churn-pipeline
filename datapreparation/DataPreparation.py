#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def preprocess_and_eda(df: pd.DataFrame, output_dir="preprocessing_reports"):
    """
    Preprocess churn dataset: clean, handle missing values, encode, scale numeric data, and perform EDA.
    Input: Validated dataframe (not CSV file).
    Output: Clean processed dataframe + PDF with visualizations + summary stats CSV.
    """
    logging.info("Starting preprocessing and EDA...")
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()

    # --- Define column groups ---
    categorical_int_cols = ["NumOfProducts", "HasCrCard", "IsActiveMember", "Exited"]
    id_cols = ["RowNumber", "CustomerId", "Surname"]  # identifiers to exclude
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Continuous numeric features (exclude categorical-like ints + IDs)
    continuous_numeric_cols = [c for c in numeric_cols if c not in categorical_int_cols + id_cols]

    # --- Handle missing values ---
    logging.info("Handling missing values...")
    for col in continuous_numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_int_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # --- Encode categorical string variables ---
    logging.info("Encoding categorical string variables...")
    categorical_str_cols = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in categorical_str_cols:
        df[col] = le.fit_transform(df[col])

    # --- Normalize continuous numeric attributes only ---
    logging.info("Normalizing continuous numeric attributes...")
    scaler = StandardScaler()
    df[continuous_numeric_cols] = scaler.fit_transform(df[continuous_numeric_cols])

    # --- Summary statistics ---
    logging.info("Generating summary statistics...")
    summary_stats = df.describe(include="all")
    summary_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_file)

    # --- EDA Visualizations into Single PDF ---
    logging.info("Creating EDA visualizations and saving to PDF...")
    pdf_file = os.path.join(output_dir, f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    with PdfPages(pdf_file) as pdf:

        # Distribution plots for continuous features
        for col in continuous_numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f"Distribution of {col}")
            pdf.savefig()
            plt.close()

        # Countplots for categorical int features
        for col in categorical_int_cols:
            plt.figure(figsize=(6, 4))
            sns.countplot(x=df[col])
            plt.title(f"Countplot of {col}")
            pdf.savefig()
            plt.close()

        # Correlation heatmap (only continuous features)
        plt.figure(figsize=(10, 8))
        corr = df[continuous_numeric_cols].corr()
        sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap (Continuous Features)")
        pdf.savefig()
        plt.close()

        # Boxplots for outliers (continuous only)
        for col in continuous_numeric_cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col])
            plt.title(f"Boxplot of {col}")
            pdf.savefig()
            plt.close()

    # --- Save Cleaned Data ---
    clean_file = os.path.join(output_dir, f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(clean_file, index=False)

    logging.info(f"Preprocessing and EDA completed. Cleaned data saved at {clean_file}")
    logging.info(f"EDA PDF report saved at {pdf_file}")

    return df


# In[ ]:




