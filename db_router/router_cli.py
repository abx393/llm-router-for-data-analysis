#!/usr/bin/env python3
"""
CLI for the LLM Router API, focusing on /query_evaluate.

Examples:

  # Just get a recommendation (no evaluation)
  python router_cli.py query-eval \
      --spec eval_binary_tasks/airline_spec.json \
      --k 3 \
      --min-accuracy 0.8 \
      --max-latency 0.05

  # Plan + optionally run evaluation on the workload
  python router_cli.py query-eval \
      --spec eval_binary_tasks/airline_spec.json \
      --workload eval_binary_tasks/airline_binary_sentiment.csv \
      --k 3 \
      --min-accuracy 0.8 \
      --max-latency 0.05
"""

import argparse
import json
import os
import sys
from typing import Optional

import requests

DEFAULT_API = os.environ.get("ROUTER_API", "http://localhost:8000")


def pretty_neighbors(neighbors):
    lines = []
    for i, n in enumerate(neighbors, start=1):
        lines.append(
            f"  {i}. task={n.get('task_name')} "
            f"(vector_id={n.get('vector_id')}, score={n.get('score'):.4f})"
        )
    return "\n".join(lines)


def do_query_eval(
    api: str,
    spec_path: str,
    workload_path: Optional[str],
    k: int,
    select_by: str,
    min_accuracy: Optional[float],
    max_latency: Optional[float],
    interactive: bool = True,
):
    with open(spec_path, "r", encoding="utf-8") as fh:
        spec_obj = json.load(fh)
    spec_str = json.dumps(spec_obj)

    print(f"API: {api}")
    print(f"Spec: {spec_path}")
    if workload_path:
        print(f"Workload: {workload_path}")
    print(f"k={k}, select_by={select_by}, min_accuracy={min_accuracy}, max_latency={max_latency}")

    # ---------- 1) Plan-only call (no workload) ----------
    data = {
        "spec": spec_str,
        "k": str(k),
        "select_by": select_by,
    }
    if min_accuracy is not None:
        data["min_recall"] = str(min_accuracy)
    if max_latency is not None:
        data["max_latency"] = str(max_latency)

    r = requests.post(f"{api}/query_evaluate", data=data, timeout=300)
    if not r.ok:
        print(f"\n❌ /query_evaluate (plan) failed: {r.status_code} {r.text}")
        sys.exit(1)

    resp = r.json()

    print("\n=== PLAN RESULTS ===")
    print(f"query_vector_id: {resp.get('query_vector_id')}")
    print(f"chosen_model: {resp.get('chosen_model')}")
    print(f"chosen_reason: {resp.get('chosen_reason')}")
    print("\nNearest neighbors:")
    print(pretty_neighbors(resp.get("neighbors", [])) or "  (none)")

    if resp.get("chosen_model") is None:
        print("\n⚠️  No model selected (see chosen_reason above).")
        return

    if not workload_path:
        print("\n(No workload CSV provided, so not running evaluation.)")
        return

    if interactive:
        ans = input("\nRun this model on the workload now? [y/N]: ").strip().lower()
        if ans != "y":
            print("Skipping evaluation.")
            return

    # ---------- 2) Full call with workload ----------
    files = {
        "workload": (
            os.path.basename(workload_path),
            open(workload_path, "rb"),
            "text/csv",
        )
    }
    r2 = requests.post(f"{api}/query_evaluate", data=data, files=files, timeout=3600)
    if not r2.ok:
        print(f"\n❌ /query_evaluate (with workload) failed: {r2.status_code} {r2.text}")
        sys.exit(1)

    resp2 = r2.json()
    print("\n=== EVALUATION RESULTS ===")
    print(f"ran_evaluation: {resp2.get('ran_evaluation')}")
    print(f"run_csv_path: {resp2.get('run_csv_path')}")
    print(f"eval_summary: {resp2.get('eval_summary')}")


def main():
    ap = argparse.ArgumentParser(description="CLI for LLM Router API")
    ap.add_argument("--api", default=DEFAULT_API, help=f"API base URL (default: {DEFAULT_API})")

    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query-eval", help="Run /query_evaluate for a given spec (and optional workload).")
    q.add_argument("--spec", required=True, help="Path to JSON spec (QuerySpec-compatible).")
    q.add_argument("--workload", help="Path to CSV workload (inputs,targets).")
    q.add_argument("--k", type=int, default=3)
    q.add_argument("--select-by", choices=["latency", "recall"], default="latency")
    q.add_argument("--min-accuracy", type=float, default=None, help="Min acceptable accuracy (min_recall).")
    q.add_argument("--max-latency", type=float, default=None, help="Max acceptable avg latency (seconds).")
    q.add_argument("--non-interactive", action="store_true", help="Do not prompt; always run if workload given.")

    args = ap.parse_args()

    if args.cmd == "query-eval":
        do_query_eval(
            api=args.api,
            spec_path=args.spec,
            workload_path=args.workload,
            k=args.k,
            select_by=args.select_by,
            min_accuracy=args.min_accuracy,
            max_latency=args.max_latency,
            interactive=not args.non_interactive,
        )


if __name__ == "__main__":
    main()
