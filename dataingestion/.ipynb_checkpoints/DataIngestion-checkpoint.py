#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
import sqlalchemy
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_csv(path_or_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_or_url)
        logging.info(f"CSV data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"CSV ingestion failed: {e}")
        return pd.DataFrame()

def load_api(endpoint: str, params=None, headers=None):
    try:
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        logging.info(f"API data fetched with {len(data)} records")
        return data
    except Exception as e:
        logging.error(f"API ingestion failed: {e}")
        return []

def load_db(query: str, connection_string: str) -> pd.DataFrame:
    try:
        engine = sqlalchemy.create_engine(connection_string)
        df = pd.read_sql(query, engine)
        logging.info(f"DB data loaded with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"DB ingestion failed: {e}")
        return pd.DataFrame()


# In[2]:





# In[ ]:




