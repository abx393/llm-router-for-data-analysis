#!/usr/bin/env python3
"""
Generate binary sentiment evaluation tasks under ./eval_binary_tasks.

For each dataset, we create:
  - <stem>_binary_sentiment.csv with columns:
        task_name,task_family,id,definition,inputs,targets
  - <stem>_spec.json with fields compatible with QuerySpec:
        {task_name, definition, example_inputs, example_outputs}

We skip any files that already exist so you can safely re-run.
"""

import os
import json
import random
from typing import Any, Dict, List

from datasets import load_dataset

OUT_DIR = "eval_binary_tasks"
N_PER_TASK = 100
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)

# ---------- task configs ----------

TASK_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "task_imdb_binary_sentiment_eval",
        "csv_stem": "imdb",
        "task_family": "Sentiment Analysis",
        "domain": "Reviews -> Movies",
        "definition": "Given a movie review, classify its sentiment as positive or negative.",
        "dataset_id": "imdb",
        "config": None,
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "filter_fn": None,
    },
    {
        "name": "task_yelp_polarity_binary_sentiment_eval",
        "csv_stem": "yelp_polarity",
        "task_family": "Sentiment Analysis",
        "domain": "Reviews -> Restaurants",
        "definition": "Given a Yelp review, classify its sentiment as positive or negative.",
        "dataset_id": "yelp_polarity",
        "config": None,
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "filter_fn": None,
    },
    {
        "name": "task_amazon_polarity_binary_sentiment_eval",
        "csv_stem": "amazon_polarity",
        "task_family": "Sentiment Analysis",
        "domain": "Reviews -> Products",
        "definition": "Given an Amazon product review, classify its sentiment as positive or negative.",
        "dataset_id": "amazon_polarity",
        "config": None,
        "split": "test",
        "text_field": "content",   # use review content
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "filter_fn": None,
    },
    {
        "name": "task_rotten_tomatoes_binary_sentiment_eval",
        "csv_stem": "rotten_tomatoes",
        "task_family": "Sentiment Analysis",
        "domain": "Reviews -> Movies",
        "definition": "Given a short movie review sentence, classify its sentiment as positive or negative.",
        "dataset_id": "rotten_tomatoes",
        "config": None,
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "label_map": {0: "negative", 1: "positive"},
        "filter_fn": None,
    },
    {
        "name": "task_tweeteval_sentiment_binary_eval",
        "csv_stem": "tweet_eval_sentiment",
        "task_family": "Sentiment Analysis",
        "domain": "Social Media -> Twitter",
        "definition": "Given a tweet, classify its sentiment as positive or negative.",
        "dataset_id": "tweet_eval",
        "config": "sentiment",
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        # label 0 = negative, 1 = neutral, 2 = positive
        "label_map": {0: "negative", 2: "positive"},
        "filter_fn": lambda ex: ex["label"] in (0, 2),
    },
    # NOTE: financial_phrasebank removed because datasets >=3 no longer supports script-based datasets.
]


def sample_indices(n_total: int, n: int) -> List[int]:
    """Sample up to n indices from range(n_total) without replacement."""
    n = min(n, n_total)
    return random.sample(range(n_total), n)


def build_task(config: Dict[str, Any]) -> None:
    name = config["name"]
    csv_stem = config["csv_stem"]
    csv_path = os.path.join(OUT_DIR, f"{csv_stem}_binary_sentiment.csv")
    spec_path = os.path.join(OUT_DIR, f"{csv_stem}_spec.json")

    if os.path.exists(csv_path) and os.path.exists(spec_path):
        print(f"[SKIP] {name}: both CSV and spec already exist.")
        return

    print(f"[TASK] Building {name} from {config['dataset_id']}")

    dataset_id = config["dataset_id"]
    ds_config = config["config"]
    split = config["split"]

    if ds_config is None:
        raw = load_dataset(dataset_id)
    else:
        raw = load_dataset(dataset_id, ds_config)

    ds = raw[split]

    # Optional filter
    filter_fn = config.get("filter_fn")
    if filter_fn is not None:
        ds = ds.filter(filter_fn)

    n_total = len(ds)
    if n_total == 0:
        print(f"  [WARN] No examples left after filtering for {name}, skipping.")
        return

    idxs = sample_indices(n_total, N_PER_TASK)

    text_field = config["text_field"]
    label_field = config["label_field"]
    label_map = config["label_map"]

    rows = []
    examples_inputs: List[str] = []
    examples_outputs: List[str] = []

    for idx in idxs:
        ex = ds[int(idx)]
        text = str(ex[text_field]).replace("\n", " ").strip()

        raw_label = ex[label_field]
        mapped = None
        if isinstance(raw_label, str):
            mapped = label_map.get(raw_label)
        else:
            mapped = label_map.get(int(raw_label))

        if mapped is None:
            # unexpected label; skip
            continue

        task_name = name
        row_id = f"{task_name}-{idx}"

        rows.append({
            "task_name": task_name,
            "task_family": config["task_family"],
            "id": row_id,
            "definition": config["definition"],
            "inputs": text,
            "targets": mapped,
        })

        if len(examples_inputs) < 4:
            examples_inputs.append(text)
            examples_outputs.append(mapped)

    if not rows:
        print(f"  [WARN] No usable rows for {name}, skipping.")
        return

    # Write CSV
    if not os.path.exists(csv_path):
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["task_name", "task_family", "id", "definition", "inputs", "targets"],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  [OK] wrote CSV → {csv_path}")
    else:
        print(f"  [SKIP] CSV already exists: {csv_path}")

    # Write spec JSON (QuerySpec-compatible)
    if not os.path.exists(spec_path):
        spec = {
            "task_name": name,
            "definition": config["definition"],
            "example_inputs": examples_inputs,
            "example_outputs": examples_outputs,
        }
        with open(spec_path, "w", encoding="utf-8") as fh:
            json.dump(spec, fh, indent=2, ensure_ascii=False)
        print(f"  [OK] wrote spec → {spec_path}")
    else:
        print(f"  [SKIP] Spec already exists: {spec_path}")


def main():
    for cfg in TASK_CONFIGS:
        try:
            build_task(cfg)
        except Exception as e:
            print(f"[ERROR] Failed to build {cfg['name']}: {e}")
    print("\nDone. Check the eval_binary_tasks/ folder.")


if __name__ == "__main__":
    main()
