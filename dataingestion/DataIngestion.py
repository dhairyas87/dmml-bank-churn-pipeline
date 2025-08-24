#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import requests
import sqlalchemy
import logging
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
import kagglehub
import os


# ---------------------
# Logging Setup
# ---------------------
log_file = "ingestion_job_results.log"

logger = logging.getLogger("IngestionLoggerResult")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# File handler with rotation (5 MB per file, keep last 3 files)
file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------
# Ingestion functions
# ---------------------
def load_csv(path_or_url: str, source: str) -> pd.DataFrame:
    try:
        logger.info(f"[{source}] CSV data loading from {path_or_url}")
        df = pd.read_csv(path_or_url)
        logger.info(f"[{source}] CSV data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"[{source}] CSV ingestion failed: {e}")
        return pd.DataFrame()

def load_api(endpoint: str, params=None, headers=None):
    try:
        logger.info(f"[API] Fetching data from {endpoint}")
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(f"[API] Data fetched with {len(data)} records")
        return data
    except Exception as e:
        logger.error(f"[API] Ingestion failed: {e}")
        return []

def load_db(query: str, connection_string: str) -> pd.DataFrame:
    try:
        logger.info(f"[DB] Executing query on {connection_string}")
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        logger.info(f"[DB] Data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"[DB] Ingestion failed: {e}")
        return pd.DataFrame()

# ---------------------
# Retry wrapper
# ---------------------
def safe_ingest(func, retries=3, delay=5, *args, **kwargs):
    """Retry ingestion function if it fails, with backoff delay."""
    attempt = 0
    while attempt < retries:
        result = func(*args, **kwargs)
        if isinstance(result, pd.DataFrame) and not result.empty:
            return result
        if isinstance(result, list) and len(result) > 0:
            return result

        attempt += 1
        logger.warning(f"Retry {attempt}/{retries} after failure. Waiting {delay} sec...")
        time.sleep(delay)
    logger.error(f"All {retries} attempts failed for {func.__name__}")
    return pd.DataFrame() if func.__name__ != "load_api" else []

# ---------------------
# Scheduler
# ---------------------
def run_periodic_ingestion(interval_seconds=60):
    """Run ingestion periodically at fixed interval."""
    while True:
        logger.info(f"=== Ingestion cycle started at {datetime.now()} ===")

        # Example sources (replace with real)
        logger.info(f"=== Loading from Microsoft data set ===")
        df_csv = safe_ingest(load_csv, 3, 5, "https://synapseaisolutionsa.z13.web.core.windows.net/data/bankcustomerchurn/churn.csv", "CSV_Source")
        


    # Download latest version
        current_directory = os.getcwd()
        path = current_directory+ "/Customer_Churn_kaggle.csv"

        logger.info("Kaggle dataset for customer churn")
        safe_ingest(load_csv, 3, 5, path, "CSV_Source")
        # For monitoring, log sizes
        logger.info(f"CSV shape: {df_csv.shape if isinstance(df_csv, pd.DataFrame) else 'N/A'}")
       

        logger.info("=== Ingestion cycle completed ===\n")
        time.sleep(interval_seconds)  # wait before next cycle
        return df_csv


# In[ ]:




