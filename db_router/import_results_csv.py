#!/usr/bin/env python3
import os
import requests

API_BASE = os.environ.get("ROUTER_API", "http://localhost:8000")
RESULTS_PATH = "results.csv"  # adjust if it's somewhere else

def main():
    url = f"{API_BASE.rstrip('/')}/import/task_results_csv"
    print(f"POSTing {RESULTS_PATH} to {url} ...")
    with open(RESULTS_PATH, "rb") as fh:
        files = {"file": ("results.csv", fh, "text/csv")}
        # skip_missing=True will silently skip rows whose task_name
        # does NOT have a TaskRow yet
        params = {"skip_missing": "true"}
        resp = requests.post(url, files=files, params=params)

    print("Status:", resp.status_code)
    print("Body:\n", resp.text)

if __name__ == "__main__":
    main()
