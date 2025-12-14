# populate_workloads.py
"""
Populate the router DB/index with a set of binary sentiment tasks
and historical model results.

Datasets (binary sentiment):
  - imdb
  - rotten_tomatoes
  - glue/sst2

Models:
  - vader  (rule-based)
  - distilbert-base-uncased-finetuned-sst-2-english
  - textattack/bert-base-uncased-imdb
  - cardiffnlp/twitter-roberta-base-sentiment-latest

This script:
  1. Creates TaskDoc entries for each dataset.
  2. Embeds them via router_app.add_task_rows_with_vectors.
  3. Evaluates each model on a subset of each dataset.
  4. Inserts ResultRow entries with accuracy + average latency.
"""

import time
from typing import List, Dict, Any

from datasets import load_dataset
from sqlmodel import Session

from router_app import (
    TaskDoc,
    add_task_rows_with_vectors,
    ResultRow,
    engine,
    _ModelRunner,
    _normalize_targets_binary,
)

# ---------------------------------------------------------------------
# Config: datasets and models
# ---------------------------------------------------------------------

# For each dataset, we specify how to load it and which fields to use.
BINARY_SENTIMENT_DATASETS: List[Dict[str, Any]] = [
    {
        "task_name": "binary_sentiment.imdb",
        "dataset": "imdb",
        "subset": None,
        "split": "test",
        "text_field": "text",
        "label_field": "label",
        "definition": "Binary sentiment classification (positive vs negative) on IMDB movie reviews.",
    },
    {
        "task_name": "binary_sentiment.rotten_tomatoes",
        "dataset": "rotten_tomatoes",
        "subset": None,
        "split": "train",
        "text_field": "text",
        "label_field": "label",
        "definition": "Binary sentiment classification on Rotten Tomatoes movie reviews.",
    },
    {
        "task_name": "binary_sentiment.glue_sst2",
        "dataset": "glue",
        "subset": "sst2",
        "split": "validation",
        "text_field": "sentence",
        "label_field": "label",
        "definition": "Binary sentiment classification on the GLUE SST-2 dataset.",
    },
]

# The models we will evaluate on each dataset.
BINARY_SENTIMENT_MODELS: List[str] = [
    "vader",
    "distilbert-base-uncased-finetuned-sst-2-english",
    "textattack/bert-base-uncased-imdb",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_binary_dataset(cfg: Dict[str, Any], max_samples: int) -> Dict[str, List[Any]]:
    """
    Load and preprocess a binary sentiment dataset based on cfg.
    Returns dict with keys: texts, labels (both lists).
    """
    ds_name = cfg["dataset"]
    subset = cfg.get("subset")
    split = cfg["split"]
    text_field = cfg["text_field"]
    label_field = cfg["label_field"]

    print(f"  -> Loading dataset {ds_name}{('/' + subset) if subset else ''} [{split}]")

    if subset:
        ds = load_dataset(ds_name, subset, split=split)
    else:
        ds = load_dataset(ds_name, split=split)

    ds = ds.shuffle(seed=42)
    n = min(max_samples, len(ds))
    ds = ds.select(range(n))

    texts: List[str] = [str(ex[text_field]) for ex in ds]
    labels_raw: List[str] = [str(ex[label_field]) for ex in ds]
    labels: List[int] = _normalize_targets_binary(labels_raw)

    return {"texts": texts, "labels": labels}


def populate_binary_sentiment(max_samples: int = 500) -> None:
    """
    For each dataset in BINARY_SENTIMENT_DATASETS:
      - Insert TaskRow and vector.
      - Evaluate each model in BINARY_SENTIMENT_MODELS.
      - Insert ResultRows with accuracy & latency.
    """
    from router_app import current_index_size  # for logging only

    print("=== Populating binary sentiment workloads and results ===")
    print(f"Using up to {max_samples} examples per dataset.\n")

    for cfg in BINARY_SENTIMENT_DATASETS:
        task_name = cfg["task_name"]
        definition = cfg.get("definition") or f"Binary sentiment classification on {cfg['dataset']}."

        print(f"\n=== Task: {task_name} ===")
        data = _load_binary_dataset(cfg, max_samples=max_samples)
        texts = data["texts"]
        labels = data["labels"]

        # Build TaskDoc and embed it
        doc = TaskDoc(
            task_name=task_name,
            definition=definition,
            example_inputs=texts[:4],
            example_outputs=[str(y) for y in labels[:4]],
        )
        inserted = add_task_rows_with_vectors([doc])
        task_info = inserted[0]
        vector_id = task_info["vector_id"]
        print(f"  -> TaskRow inserted with id={task_info['id']}, vector_id={vector_id}")
        print(f"  -> Current FAISS index size: {current_index_size()} vectors")

        # Evaluate each model
        with Session(engine) as s:
            for model_name in BINARY_SENTIMENT_MODELS:
                print(f"  -> Evaluating model '{model_name}' on {len(texts)} examples...")
                start = time.time()
                preds = _ModelRunner.run(model_name, texts)
                runtime = time.time() - start

                n = len(labels)
                correct = sum(int(p == g) for p, g in zip(preds, labels))
                acc = correct / max(1, n)
                avg_lat = runtime / max(1, n)

                row = ResultRow(
                    task_name=task_name,
                    model=model_name,
                    accuracy=acc,
                    latency=avg_lat,
                    number_of_samples=n,
                    vector_id=vector_id,
                )
                s.add(row)
                s.commit()
                print(
                    f"     -> acc={acc:.4f}, avg_latency={avg_lat:.6f} s "
                    f"(total_runtime={runtime:.3f} s)"
                )

    print("\n=== Done populating workloads and results. ===")


if __name__ == "__main__":
    populate_binary_sentiment(max_samples=500)
