# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 18:57:40 2025

@author: thoma
"""

import os
import requests

GITHUB_API = "https://api.github.com"
REPO = os.environ["REPO"]
TOKEN = os.environ["GH_PAT"]

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

def get_latest_successful_run(workflow_name="Part_1"):
    url = f"{GITHUB_API}/repos/{REPO}/actions/workflows/{workflow_name}.yml/runs?status=success&per_page=1"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    runs = r.json()["workflow_runs"]
    if not runs:
        raise Exception("No successful workflow runs found.")
    return runs[0]["id"]

def get_artifact_id(run_id, artifact_name="intermediate-data"):
    url = f"{GITHUB_API}/repos/{REPO}/actions/runs/{run_id}/artifacts"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    artifacts = r.json()["artifacts"]
    for artifact in artifacts:
        if artifact["name"] == artifact_name:
            return artifact["id"]
    raise Exception(f"Artifact '{artifact_name}' not found.")

def download_artifact(artifact_id, filename="intermediate-data.zip"):
    url = f"{GITHUB_API}/repos/{REPO}/actions/artifacts/{artifact_id}/zip"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)

if __name__ == "__main__":
    run_id = get_latest_successful_run()
    artifact_id = get_artifact_id(run_id)
    download_artifact(artifact_id)
    print("✅ Artifact downloaded successfully.")
