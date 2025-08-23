#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_partitioned_path(base_dir: str, source: str, ext: str) -> str:
    now = datetime.now()
    folder = os.path.join(
        base_dir,
        source,
        f"year={now.year}",
        f"month={now.month:02d}",
        f"day={now.day:02d}"
    )
    ensure_dir(folder)
    return os.path.join(folder, f"{source}_DataStorage.{ext}")

def save_csv_or_db(df: pd.DataFrame, base_dir: str, source: str):
    if not df.empty:
        path = get_partitioned_path(base_dir, source, "csv")
        df.to_csv(path, index=False)
        logging.info(f"{source.upper()} data stored at {path}")

def save_api(data, base_dir: str, source: str):
    if data:
        path = get_partitioned_path(base_dir, source, "json")
        pd.Series(data).to_json(path, orient="records", indent=2)
        logging.info(f"{source.upper()} data stored at {path}")


# In[8]:


get_ipython().system('jupyter nbconvert --to script DataStorage.ipynb')


# In[ ]:




