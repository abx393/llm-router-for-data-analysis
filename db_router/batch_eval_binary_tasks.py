#!/usr/bin/env python3
"""
Batch evaluation runner for eval_binary_tasks.

For each CSV in eval_binary_tasks matching '*_binary_sentiment.csv':
  - Derive a spec JSON path '<stem>_spec.json'
      e.g. 'airline_binary_sentiment.csv' -> 'airline_spec.json'
  - POST to /query_evaluate with:
      spec (JSON string), k, select_by, min_recall, max_latency, workload
  - Collect:
      * top-3 neighbors (task names)
      * chosen_model, chosen_reason
      * eval_summary: accuracy, avg_latency
  - Write a single CSV with one row per task.

You can re-run this file with different MIN_ACCURACY / MAX_LATENCY values.
"""

import os
import glob
import json
import csv
from typing import Any, Dict, List, Optional

import requests

API_BASE = os.environ.get("ROUTER_API", "http://localhost:8000")
EVAL_DIR = "eval_binary_tasks"
OUT_CSV = "router_eval_runs.csv"

# ----- knobs you’ll change between runs -----
K = 5
SELECT_BY = "latency"      # "latency" or "recall"
MIN_ACCURACY: Optional[float] = 0.85  # e.g. 0.8
MAX_LATENCY: Optional[float] = None   # e.g. 0.05
# -------------------------------------------


def stem_to_spec_path(csv_path: str) -> str:
    """
    Map 'airline_binary_sentiment.csv' -> 'airline_spec.json'
    """
    base = os.path.basename(csv_path)
    stem, _ = os.path.splitext(base)
    if stem.endswith("_binary_sentiment"):
        stem = stem[: -len("_binary_sentiment")]
    return os.path.join(EVAL_DIR, f"{stem}_spec.json")


def call_query_evaluate(
    api: str,
    spec_path: str,
    workload_path: str,
    k: int,
    select_by: str,
    min_accuracy: Optional[float],
    max_latency: Optional[float],
) -> Dict[str, Any]:
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec_obj = json.load(fh)
    spec_str = json.dumps(spec_obj)

    data: Dict[str, Any] = {
        "spec": spec_str,
        "k": str(k),
        "select_by": select_by,
    }
    if min_accuracy is not None:
        data["min_recall"] = str(min_accuracy)
    if max_latency is not None:
        data["max_latency"] = str(max_latency)

    files = {
        "workload": (
            os.path.basename(workload_path),
            open(workload_path, "rb"),
            "text/csv",
        )
    }

    r = requests.post(f"{api}/query_evaluate", data=data, files=files, timeout=3600)
    if not r.ok:
        raise RuntimeError(f"/query_evaluate failed for {workload_path}: {r.status_code} {r.text}")

    return r.json()


def main():
    csv_paths = sorted(glob.glob(os.path.join(EVAL_DIR, "*_binary_sentiment.csv")))
    if not csv_paths:
        print(f"No *_binary_sentiment.csv files found under {EVAL_DIR}")
        return

    rows_out: List[Dict[str, Any]] = []

    print(f"API_BASE={API_BASE}")
    print(f"EVAL_DIR={EVAL_DIR}")
    print(f"Found {len(csv_paths)} workload CSVs.")
    print(f"K={K}, select_by={SELECT_BY}, MIN_ACCURACY={MIN_ACCURACY}, MAX_LATENCY={MAX_LATENCY}")
    print()

    for csv_path in csv_paths:
        spec_path = stem_to_spec_path(csv_path)
        if not os.path.exists(spec_path):
            print(f"[WARN] No spec JSON for {csv_path} (expected {spec_path}), skipping.")
            continue

        base = os.path.basename(csv_path)
        print(f"[TASK] {base}")

        try:
            resp = call_query_evaluate(
                API_BASE,
                spec_path,
                csv_path,
                K,
                SELECT_BY,
                MIN_ACCURACY,
                MAX_LATENCY,
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            rows_out.append(
                {
                    "task_name": os.path.splitext(base)[0],
                    "neighbor_1": "",
                    "neighbor_2": "",
                    "neighbor_3": "",
                    "chosen_model": "",
                    "router_reason": str(e),
                    "eval_accuracy": "",
                    "eval_avg_latency": "",
                    "error": "http_error",
                }
            )
            continue

        neighbors = resp.get("neighbors", [])
        chosen_model = resp.get("chosen_model")
        chosen_reason = resp.get("chosen_reason")
        eval_summary = resp.get("eval_summary") or {}

        # Top 3 neighbor task_names
        n1 = neighbors[0]["task_name"] if len(neighbors) > 0 else ""
        n2 = neighbors[1]["task_name"] if len(neighbors) > 1 else ""
        n3 = neighbors[2]["task_name"] if len(neighbors) > 2 else ""

        acc = eval_summary.get("accuracy")
        lat = eval_summary.get("avg_latency")

        error = ""
        if chosen_model is None:
            error = "no_model_selected"

        rows_out.append(
            {
                "task_name": resp.get("query_vector_id", ""),
                "spec_task_name": resp.get("neighbors", [{}])[0].get("task_name", ""),
                "workload_file": base,
                "neighbor_1": n1,
                "neighbor_2": n2,
                "neighbor_3": n3,
                "chosen_model": chosen_model or "",
                "router_reason": chosen_reason or "",
                "eval_accuracy": acc if acc is not None else "",
                "eval_avg_latency": lat if lat is not None else "",
                "error": error,
            }
        )

    if not rows_out:
        print("No rows to write.")
        return

    fieldnames = [
        "task_name",
        "spec_task_name",
        "workload_file",
        "neighbor_1",
        "neighbor_2",
        "neighbor_3",
        "chosen_model",
        "router_reason",
        "eval_accuracy",
        "eval_avg_latency",
        "error",
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"\nWrote {len(rows_out)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
