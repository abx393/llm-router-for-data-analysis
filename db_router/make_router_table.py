#!/usr/bin/env python
"""
Generate a formatted summary table comparing:
- Largest model (roberta-large-en)
- Router choice (distilbert-sst2)
- Smallest model (vader)

Outputs:
- Markdown table printed to stdout
- PNG table saved as router_summary_table.png
"""

import math
from textwrap import dedent

import matplotlib.pyplot as plt


def main():
    # ---- Data (from your script output) ----
    rows = [
        {
            "Role": "Largest model",
            "Model": "roberta-large-en",
            "mean_accuracy": 0.9325,
            "mean_latency_s": 0.226192,
        },
        {
            "Role": "Router choice",
            "Model": "Varied",
            "mean_accuracy": 0.8807,
            "mean_latency_s": 0.036737,
        },
        {
            "Role": "Smallest model",
            "Model": "vader",
            "mean_accuracy": 0.6765,
            "mean_latency_s": 0.000772,
        },
    ]

    # Convert to display strings
    for r in rows:
        r["Accuracy %"] = f"{r['mean_accuracy'] * 100:5.2f}%"
        r["Latency (ms)"] = f"{r['mean_latency_s'] * 1000:6.1f}"

    # Column order for table
    columns = ["Role", "Model", "Accuracy %", "Latency (ms)"]

    # ---- 1) Print Markdown table to stdout ----
    header_row = "| " + " | ".join(columns) + " |"
    separator_row = "| " + " | ".join(["---"] * len(columns)) + " |"
    print(header_row)
    print(separator_row)
    for r in rows:
        print(
            "| "
            + " | ".join(str(r[col]) for col in columns)
            + " |"
        )

    # ---- 2) Create a PNG table with matplotlib ----
    cell_text = []
    for r in rows:
        cell_text.append([r[col] for col in columns])

    fig, ax = plt.subplots(figsize=(9, 1.8))  # tweak size as needed
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4)  # (x, y) scaling

    # Slight bold for header row
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e0e0e0")  # light gray header
        else:
            cell.set_facecolor("white")

    plt.tight_layout()
    out_path = "router_summary_table.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved PNG table to: {out_path}")


if __name__ == "__main__":
    main()
