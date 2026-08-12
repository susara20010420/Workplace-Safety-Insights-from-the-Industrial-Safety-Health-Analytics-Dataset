# Run this after the notebook pipeline has written its latest artifacts.
# Adapt SCRIPT_DIR and REPO only. Keep GITHUB_TOKEN in Colab Secrets.

from google.colab import drive, userdata
import os
import subprocess

# Mount Drive if it is not already mounted.
drive.mount('/content/drive')

PROJECT_FOLDER = "/content/drive/MyDrive/Work Place Safety Insights"
SCRIPT_DIR = "/content/YOUR_CLONED_REPOSITORY/scripts"   # <-- change this
REPO = "YOUR_GITHUB_USERNAME/YOUR_REPOSITORY"            # <-- change this
BRANCH = "main"
TARGET = "dashboard/dashboard_snapshot.json"
SNAPSHOT = f"{PROJECT_FOLDER}/dashboard_snapshot.json"

# 1) Recompute every dashboard metric/series from the newest notebook artifacts.
subprocess.run([
    "python", f"{SCRIPT_DIR}/build_dashboard_snapshot.py",
    "--project-folder", PROJECT_FOLDER,
    "--output", SNAPSHOT,
], check=True)

# 2) Publish the one JSON snapshot to GitHub.
# Add GITHUB_TOKEN in Colab: left sidebar -> Secrets -> New secret.
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
subprocess.run([
    "python", f"{SCRIPT_DIR}/publish_snapshot_to_github.py",
    "--repo", REPO,
    "--file", SNAPSHOT,
    "--target", TARGET,
    "--branch", BRANCH,
], check=True)

print("Dashboard snapshot rebuilt and published.")
