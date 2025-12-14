# label_sanity.py
"""
Label sanity checker for router binary sentiment tasks.

Checks that:
  - Dataset labels can be normalized to {0,1}.
  - Model predictions are in {0,1}.
"""

from typing import List, Dict, Any

from datasets import load_dataset

from router_app import _ModelRunner, _normalize_targets_binary
from populate_workloads import BINARY_SENTIMENT_DATASETS, BINARY_SENTIMENT_MODELS


def _sample_dataset(cfg: Dict[str, Any], n: int = 32) -> Dict[str, List[Any]]:
    ds_name = cfg["dataset"]
    subset = cfg.get("subset")
    split = cfg["split"]
    text_field = cfg["text_field"]
    label_field = cfg["label_field"]

    print(f"Loading sample from {ds_name}{('/' + subset) if subset else ''} [{split}]...")

    if subset:
        ds = load_dataset(ds_name, subset, split=split)
    else:
        ds = load_dataset(ds_name, split=split)

    ds = ds.shuffle(seed=123)
    ds = ds.select(range(min(n, len(ds))))

    texts = [str(ex[text_field]) for ex in ds]
    labels_raw = [str(ex[label_field]) for ex in ds]
    labels = _normalize_targets_binary(labels_raw)

    return {"texts": texts, "labels": labels}


def main() -> None:
    print("=== Label Sanity Check ===\n")

    for cfg in BINARY_SENTIMENT_DATASETS:
        print(f"\n--- Dataset: {cfg['task_name']} ---")
        data = _sample_dataset(cfg, n=32)
        texts = data["texts"]
        labels = data["labels"]
        uniq_labels = sorted(set(labels))
        print(f"  Dataset labels (normalized): {uniq_labels}")
        if not set(uniq_labels).issubset({0, 1}):
            print("  ⚠️  Dataset labels are not binary after normalization!")
            continue

        for model in BINARY_SENTIMENT_MODELS:
            print(f"  -> Checking model '{model}'...")
            preds = _ModelRunner.run(model, texts)
            uniq_preds = sorted(set(preds))
            print(f"     Predictions: {uniq_preds}")
            if not set(uniq_preds).issubset({0, 1}):
                print("     ⚠️  Predictions are not binary {0,1}!")
            else:
                print("     ✅ OK: predictions are binary.")

    print("\n=== Done. ===")


if __name__ == "__main__":
    main()
