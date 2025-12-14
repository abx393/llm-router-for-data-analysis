#!/usr/bin/env python3
import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


@dataclass
class CsvSummary:
    path: str
    mean_acc: float          # accuracy (0–1)
    mean_lat: float          # latency in seconds
    n_rows: int
    role: str                # "largest", "smallest", "router"
    role_label: str          # e.g. "Largest model (roberta-large-en)"
    majority_model: str


LARGEST_MODELS = {"roberta-large-en", "siebert/sentiment-roberta-large-english"}
SMALLEST_MODELS = {"vader", "bert-tiny-sst2"}


def infer_role_from_models(counts: Dict[str, int]) -> Tuple[str, str]:
    """
    Given counts of chosen_model strings, infer which role this CSV represents.

    Returns:
        (role, role_label)
        role in {"largest", "smallest", "router"}
    """
    if not counts:
        return "router", "Router choice (unknown)"

    majority_model = max(counts.items(), key=lambda kv: kv[1])[0]

    if majority_model in LARGEST_MODELS:
        return "largest", f"Largest model ({majority_model})"
    if majority_model in SMALLEST_MODELS:
        return "smallest", f"Smallest model ({majority_model})"

    return "router", f"Router choice ({majority_model})"


def load_metrics_and_role_from_csv(path: str) -> Tuple[Optional[CsvSummary], Optional[float]]:
    """
    Load eval_accuracy and eval_avg_latency from a results CSV and infer its role.

    Returns:
        (CsvSummary or None, max_latency_seen or None)

    Skips rows that:
      - have missing accuracy/latency, or
      - have a non-empty 'error' column (if present).
    """
    accuracies: List[float] = []
    latencies: List[float] = []
    model_counts: Dict[str, int] = {}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Count chosen_model for role inference (even if row is later skipped for metrics)
            cm = (row.get("chosen_model") or "").strip()
            if cm:
                model_counts[cm] = model_counts.get(cm, 0) + 1

            # Skip rows with an 'error' message if error column exists and is non-empty
            if "error" in row and (row.get("error") or "").strip():
                continue

            acc_raw = (row.get("eval_accuracy") or "").strip()
            lat_raw = (row.get("eval_avg_latency") or "").strip()

            if not acc_raw or not lat_raw:
                continue

            try:
                acc = float(acc_raw)
                lat = float(lat_raw)  # seconds
            except ValueError:
                continue

            accuracies.append(acc)
            latencies.append(lat)

    if not accuracies or not latencies:
        return None, None

    mean_acc = sum(accuracies) / len(accuracies)
    mean_lat = sum(latencies) / len(latencies)  # seconds
    n_rows = len(accuracies)

    role, role_label = infer_role_from_models(model_counts)
    majority_model = max(model_counts.items(), key=lambda kv: kv[1])[0] if model_counts else "unknown"

    summary = CsvSummary(
        path=path,
        mean_acc=mean_acc,
        mean_lat=mean_lat,
        n_rows=n_rows,
        role=role,
        role_label=role_label,
        majority_model=majority_model,
    )
    return summary, max(latencies)


def make_summary_plot(summary: CsvSummary, global_max_latency: float) -> str:
    """
    Create a summary figure for a single CSV:
      - Subplot 1: mean accuracy
      - Subplot 2: mean latency (milliseconds)

    Axes:
      - Accuracy: y in [0.6, 1.0], minor ticks 0.05, major ticks 0.1
      - Latency: y in [0, 250] ms, tick marks every 10 ms
    """
    base = os.path.basename(summary.path)
    name, _ = os.path.splitext(base)
    out_path = os.path.join(os.path.dirname(summary.path), f"{name}_summary.png")

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle(f"{name} (n={summary.n_rows})\n{summary.role_label}", fontsize=12)

    # Accuracy subplot (keep as-is)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(["Accuracy"], [summary.mean_acc])
    ax1.set_ylim(0.6, 1.0)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Mean Accuracy")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # Latency subplot (convert seconds → ms)
    ax2 = fig.add_subplot(1, 2, 2)
    mean_lat_ms = summary.mean_lat * 1000.0
    ax2.bar(["Latency (ms)"], [mean_lat_ms])
    ax2.set_ylim(0.0, 250.0)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(10.0))
    # If you really want only 10ms ticks, you can skip minor ticks;
    # otherwise you can add a finer minor locator.
    ax2.set_ylabel("Avg latency (ms)")
    ax2.set_title("Mean Latency")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.0, 1, 0.90])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return out_path


def make_combined_plot(summaries: List[CsvSummary], global_max_latency: float, out_dir: str) -> Optional[str]:
    """
    Make one combined PNG that aggregates metrics by role:
      - roles: 'largest', 'smallest', 'router'
      - two subplots: accuracy and latency (milliseconds)
      - Accuracy: [0.6, 1.0], Latency: [0, 250] ms
    """
    if not summaries:
        return None

    # Aggregate per role with row-count weighting
    agg: Dict[str, Dict[str, float]] = {}
    for s in summaries:
        role = s.role
        d = agg.setdefault(role, {"sum_acc": 0.0, "sum_lat": 0.0, "sum_n": 0.0})
        d["sum_acc"] += s.mean_acc * s.n_rows
        d["sum_lat"] += s.mean_lat * s.n_rows       # seconds
        d["sum_n"] += s.n_rows

    roles_info: List[Tuple[str, float, float]] = []  # (role, mean_acc, mean_lat_seconds)
    for role, d in agg.items():
        if d["sum_n"] <= 0:
            continue
        mean_acc = d["sum_acc"] / d["sum_n"]
        mean_lat = d["sum_lat"] / d["sum_n"]       # seconds
        roles_info.append((role, mean_acc, mean_lat))

    if not roles_info:
        return None

    # Human-readable labels and colors per role
    role_labels = {
        "largest": "Largest model",
        "smallest": "Smallest model",
        "router": "Router choice",
    }
    role_colors = {
        "largest": "tab:orange",
        "smallest": "tab:green",
        "router": "tab:blue",
    }

    out_path = os.path.join(out_dir, "combined_results_summary.png")

    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Combined Summary (weighted by #rows)", fontsize=14)

    # ---- Accuracy subplot ----
    ax1 = fig.add_subplot(1, 2, 1)
    # Sort roles by accuracy descending
    sorted_by_acc = sorted(roles_info, key=lambda x: x[1], reverse=True)
    x_labels_acc = []
    y_vals_acc = []
    colors_acc = []
    for role, mean_acc, _ in sorted_by_acc:
        x_labels_acc.append(role_labels.get(role, role))
        y_vals_acc.append(mean_acc)
        colors_acc.append(role_colors.get(role, "gray"))

    ax1.bar(x_labels_acc, y_vals_acc, color=colors_acc)
    ax1.set_ylim(0.6, 1.0)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Mean Accuracy (All Configs)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- Latency subplot (ms) ----
    ax2 = fig.add_subplot(1, 2, 2)
    # Sort roles by latency descending
    sorted_by_lat = sorted(roles_info, key=lambda x: x[2], reverse=True)
    x_labels_lat = []
    y_vals_lat_ms = []
    colors_lat = []
    for role, _, mean_lat in sorted_by_lat:
        x_labels_lat.append(role_labels.get(role, role))
        y_vals_lat_ms.append(mean_lat * 1000.0)  # seconds → ms
        colors_lat.append(role_colors.get(role, "gray"))

    ax2.bar(x_labels_lat, y_vals_lat_ms, color=colors_lat)
    ax2.set_ylim(0.0, 250.0)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(10.0))
    ax2.set_ylabel("Avg latency (ms)")
    ax2.set_title("Mean Latency (All Configs)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # Build legend from roles/colors
    legend_roles = ["largest", "smallest", "router"]
    handles = []
    labels = []
    import matplotlib.patches as mpatches

    for role in legend_roles:
        if any(r == role for r, _, _ in roles_info):
            handles.append(mpatches.Patch(color=role_colors[role], label=role_labels[role]))
            labels.append(role_labels[role])

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            bbox_to_anchor=(0.5, -0.02),
        )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean accuracy/latency for router eval results CSVs (per-file + combined)."
    )
    parser.add_argument(
        "csvs",
        nargs="*",
        help="Paths to results CSV files.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        help="Glob pattern for CSVs, e.g. 'eval_binary_tasks/*.csv'.",
    )
    args = parser.parse_args()

    csv_paths: List[str] = []

    if args.glob_pattern:
        csv_paths.extend(sorted(glob.glob(args.glob_pattern)))

    if args.csvs:
        csv_paths.extend(args.csvs)

    # Deduplicate while preserving order
    seen = set()
    unique_paths: List[str] = []
    for p in csv_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)

    if not unique_paths:
        print("No CSV files provided. Use positional paths or --glob.")
        return

    summaries: List[CsvSummary] = []
    global_max_latency = 0.0

    for path in unique_paths:
        if not os.path.isfile(path):
            print(f"[SKIP] Not a file: {path}")
            continue

        summary, max_lat = load_metrics_and_role_from_csv(path)
        if summary is None:
            print(f"[SKIP] No valid rows with eval_accuracy/eval_avg_latency in {path}")
            continue

        summaries.append(summary)
        if max_lat is not None and max_lat > global_max_latency:
            global_max_latency = max_lat

    if not summaries:
        print("No valid summaries computed from the provided CSVs.")
        return

    # Per-file plots
    for s in summaries:
        out_png = make_summary_plot(s, global_max_latency)
        print(
            f"[OK] {s.path}\n"
            f"     role={s.role_label}, mean accuracy = {s.mean_acc:.4f}, "
            f"mean latency = {s.mean_lat:.6f} s (n={s.n_rows})\n"
            f"     saved plot → {out_png}"
        )

    # Combined plot (aggregated by role)
    try:
        common_dir = os.path.commonpath([os.path.dirname(s.path) for s in summaries])
        if not common_dir:
            common_dir = "."
    except ValueError:
        common_dir = "."

    combined_png = make_combined_plot(summaries, global_max_latency, common_dir)
    if combined_png:
        print(f"[OK] Combined summary saved → {combined_png}")
    else:
        print("[INFO] Combined summary not generated (not enough role data).")


if __name__ == "__main__":
    main()
