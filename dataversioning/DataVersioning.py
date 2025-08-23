#!/usr/bin/env python
# coding: utf-8

# In[10]:


import os
import subprocess
import json
from datetime import datetime

def run_git_command(cmd):
    """Run git command and handle 'nothing to commit' gracefully."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if "nothing to commit" in str(e):
            print("ℹ️ No changes to commit, skipping...")
        else:
            raise

def save_and_version_both(raw_df, transformed_df, raw_path, transformed_path,
                          dataset_name, notes="", remote="origin", branch="main"):
    """
    Save raw & transformed datasets together, commit all repo changes with git,
    push to remote, and track as a single combined version in metadata.
    """
    # Ensure directories exist
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(transformed_path), exist_ok=True)

    # Save datasets
    raw_df.to_csv(raw_path, index=False)
    transformed_df.to_csv(transformed_path, index=False)

    # Stage all changes (new, modified, deleted)
    subprocess.run(["git", "add", "-A"])

    # Commit changes
    commit_msg = f"Dataset update: {dataset_name} (raw + transformed) - {notes}"
    run_git_command(["git", "commit", "-m", commit_msg])

    # Get commit hash (latest commit regardless of whether new or old)
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    # Prepare metadata storage
    metadata_file = "data/version_metadata.json"
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    entry = {
        "dataset_name": dataset_name,
        "paths": {
            "raw": raw_path,
            "transformed": transformed_path
        },
        "version_id": commit_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": notes
    }
    metadata.append(entry)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)

    # Stage all changes again including metadata
    subprocess.run(["git", "add", "-A"])
    run_git_command(["git", "commit", "-m", f"Update version metadata for {dataset_name}"])

    # Sync & push
    subprocess.run(["git", "pull", "--rebase", remote, branch], check=True)
    subprocess.run(["git", "push", remote, branch], check=True)

    print(f"✅ Raw + Transformed datasets for {dataset_name} saved, versioned, and pushed under commit {commit_id}")


# In[ ]:




