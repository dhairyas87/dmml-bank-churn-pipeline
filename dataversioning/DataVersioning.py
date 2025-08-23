#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import subprocess
import json
from datetime import datetime

def save_and_version_dataset(df, path, dataset_name, dataset_type, notes="", remote="origin", branch="main"):
    """
    Save dataset to path, commit with git, push to remote, and track version metadata.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    # Git add & commit
    subprocess.run(["git", "add", path], check=True)
    commit_msg = f"{dataset_type.capitalize()} dataset update: {dataset_name} - {notes}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)

    # Get current commit hash
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    # Store version metadata
    metadata_file = "dataversioning/version_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    entry = {
        "dataset_name": dataset_name,
        "dataset_type": dataset_type,
        "path": path,
        "version_id": commit_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes
    }
    metadata.append(entry)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    subprocess.run(["git", "add", metadata_file], check=True)
    subprocess.run(["git", "commit", "-m", f"Update version metadata for {dataset_name}"], check=True)

    # Push to remote
    subprocess.run(["git", "pull", "--rebase", remote, branch], check=True)
    subprocess.run(["git", "push", remote, branch], check=True)

    print(f"✅ {dataset_type.capitalize()} dataset {dataset_name} saved, versioned, and pushed with commit {commit_id}")


# In[ ]:




