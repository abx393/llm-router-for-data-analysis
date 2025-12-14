#!/usr/bin/env python
"""
Single-demo runner for /query_evaluate on the airline workload.

- Reads a QuerySpec from airline_spec.json (or another JSON file).
- Sends airline_binary_sentiment.csv (or another CSV) as `workload`.
- Uses: k=3, select_by="latency", min_recall=0.9, max_latency=None.
- Pretty-prints:
    * Parameters
    * Nearest neighbors (with nice task names)
    * Chosen model + avg accuracy/latency over neighbors
    * Final eval accuracy / latency
- Appends a summary row to demo_results.csv.
"""

import argparse
import json
import os
import csv
import statistics
import requests


def pretty_task_name(raw: str | None) -> str:
    """
    Convert raw task_name like:
        'task475_yelp_polarity_classification'
    ->  'Yelp Polarity Classification'
    """
    if not raw:
        return "(Unknown task)"

    # Drop prefix up to first underscore (e.g. "task475_")
    if "_" in raw:
        _, rest = raw.split("_", 1)
    else:
        rest = raw

    # Replace remaining underscores with spaces and title-case
    return " ".join(word.capitalize() for word in rest.split("_"))


def main():
    parser = argparse.ArgumentParser(
        description="Run /query_evaluate once on a single workload (e.g., airline)."
    )
    parser.add_argument(
        "--api",
        default="http://localhost:8000",
        help="Router API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--spec",
        default="airline_spec.json",
        help="Path to QuerySpec JSON file (default: airline_spec.json)",
    )
    parser.add_argument(
        "--csv",
        default="airline_binary_sentiment.csv",
        help="Path to workload CSV (default: airline_binary_sentiment.csv)",
    )
    parser.add_argument(
        "--out",
        default="../demo_results.csv",
        help="Path to append demo results CSV (default: demo_results.csv)",
    )
    args = parser.parse_args()

    api_base = args.api.rstrip("/")
    spec_path = args.spec
    csv_path = args.csv
    out_path = args.out

    if not os.path.exists(spec_path):
        raise SystemExit(f"[ERROR] Spec JSON not found: {spec_path}")
    if not os.path.exists(csv_path):
        raise SystemExit(f"[ERROR] Workload CSV not found: {csv_path}")

    with open(spec_path, "r") as fh:
        spec = json.load(fh)


    # Params
    k = 5
    select_by = "recall"
    min_recall = 0.01
    max_latency = None


    data = {
        "spec": json.dumps(spec),
        "k": str(k),
        "select_by": select_by,
        "min_recall": str(min_recall),
        # Do NOT send max_latency when None; the backend treats absence as None.
    }

    files = {
        "workload": (
            os.path.basename(csv_path),
            open(csv_path, "rb"),
            "text/csv",
        )
    }

    url = f"{api_base}/query_evaluate"

    print("========================================")
    print("    kNN Router Airline Dataset Demo     ")
    print("========================================")
    print(f"API base       : {api_base}")
    print(f"Spec JSON      : {spec_path}")
    print(f"Workload CSV   : {csv_path}")
    print()
    print("Parameters:")
    print(f"  k            = {k}")
    print(f"  select_by    = {select_by}")
    print(f"  min_recall   = {min_recall}")
    print(f"  max_latency  = {max_latency}")
    print()

    print(f"[INFO] POST {url}")
    resp = requests.post(url, data=data, files=files)
    files["workload"][1].close()

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("[ERROR] HTTP error from router:")
        print(resp.text)
        raise SystemExit(e)

    js = resp.json()

    # ---------------------------------
    # Neighbors (with pretty task names)
    # ---------------------------------
    neighbors = js.get("neighbors", [])
    print("---- Nearest Neighbors ----")
    if not neighbors:
        print("No neighbors returned (index might be empty?).")
    else:
        for i, nbh in enumerate(neighbors, start=1):
            raw_name = nbh.get("task_name")
            pretty_name = pretty_task_name(raw_name)
            score = nbh.get("score", 0.0)
            print(f"{i}. {pretty_name:40s}  (raw: {raw_name}, score={score:.4f})")
    print()

    # ---------------------------------------------------------
    # Chosen model + its accuracy/latency over neighbor tasks
    # ---------------------------------------------------------
    chosen_model = js.get("chosen_model")
    chosen_reason = js.get("chosen_reason")

    print("---- Router Choice ----")
    print(f"Chosen model   : {chosen_model}")
    print(f"Reason         : {chosen_reason}")
    print()

    neighbor_task_names = {
        n.get("task_name")
        for n in neighbors
        if n.get("task_name") is not None
    }

    cands = js.get("candidate_results", [])

    # Filter to rows for the chosen model on neighbor tasks
    chosen_rows = [
        c for c in cands
        if c.get("model") == chosen_model and c.get("task_name") in neighbor_task_names
    ]

    accs = [c.get("accuracy") for c in chosen_rows if c.get("accuracy") is not None]
    lats = [c.get("latency") for c in chosen_rows if c.get("latency") is not None]

    print("---- Chosen Model Performance on Neighbors ----")
    if not chosen_rows:
        print("No candidate rows for the chosen model among neighbors.")
    else:
        for c in chosen_rows:
            raw_name = c.get("task_name")
            pretty_name = pretty_task_name(raw_name)
            acc = c.get("accuracy")
            lat = c.get("latency")
            print(
                f"- {pretty_name:40s}  "
                f"acc={acc:.4f}  lat={lat:.6f} s"
            )

        mean_acc = statistics.mean(accs) if accs else None
        mean_lat = statistics.mean(lats) if lats else None
        print()
        print("Summary across neighbor tasks:")
        if mean_acc is not None:
            print(f"  Mean accuracy      : {mean_acc:.4f}")
        if mean_lat is not None:
            print(f"  Mean latency       : {mean_lat:.6f} s")
            print(f"  Mean latency (ms)  : {mean_lat * 1000:.2f} ms")
    print()

    # ---------------------------
    # Final evaluation on workload
    # ---------------------------
    eval_summary = js.get("eval_summary")
    run_csv_path = js.get("run_csv_path")
    print("Running on query workload...")
    print("---- Final Evaluation on Airline Workload ----")
    if eval_summary:
        eval_acc = eval_summary.get("accuracy")
        eval_lat = eval_summary.get("avg_latency")
        n = eval_summary.get("n")

        if eval_acc is not None:
            print(f"Eval accuracy        : {eval_acc:.4f}")
        if eval_lat is not None:
            print(f"Eval avg latency     : {eval_lat:.6f} s per example")
            print(f"Eval avg latency (ms): {eval_lat * 1000:.2f} ms per example")
        if n is not None:
            print(f"# of examples        : {n}")
    else:
        print("No eval_summary returned (maybe workload was missing?).")

    if run_csv_path:
        print(f"Router wrote per-example CSV to: {run_csv_path}")
    print()

    # --------------------------
    # Append summary to CSV file
    # --------------------------
    print("---- Writing demo_results.csv ----")
    row = {
        "spec_task_name": spec.get("task_name", ""),
        "workload_file": os.path.basename(csv_path),
        "chosen_model": chosen_model or "",
        "router_reason": chosen_reason or "",
        "eval_accuracy": eval_summary.get("accuracy") if eval_summary else "",
        "eval_avg_latency": eval_summary.get("avg_latency") if eval_summary else "",
        "n_examples": eval_summary.get("n") if eval_summary else "",
        "router_run_csv_path": run_csv_path or "",
    }

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "spec_task_name",
                "workload_file",
                "chosen_model",
                "router_reason",
                "eval_accuracy",
                "eval_avg_latency",
                "n_examples",
                "router_run_csv_path",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended demo row to: {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
