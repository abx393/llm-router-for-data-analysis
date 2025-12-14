# LLM Router – Binary Sentiment Demo

This repo implements a simple **LLM router** that:

- Stores **workload descriptions** (tasks) in a vector DB (FAISS) using a sentence-transformer.
- Stores **historical model results** (accuracy, latency) for those workloads in SQLite.
- At **query time**:
  - You send a JSON description of your workload.
  - The router finds **similar tasks** via KNN over embeddings.
  - It collects all models that have run on those tasks.
  - It applies your constraints (min accuracy, max latency).
  - It returns the **best model** for your use case.
- Optionally, it can **run the chosen model** on your workload CSV and log results.

The reference implementation focuses on **binary sentiment analysis** over multiple public datasets and models, but the structure is easy to extend to other NLP or vision tasks.

---

## Files

- `router_app.py`  
  FastAPI backend with:
  - Sentence-transformer embeddings
  - FAISS index
  - SQLite (via SQLModel)
  - `/import/task_docs_csv` and `/import/task_results_csv`
  - `/query_evaluate` (main router endpoint + optional evaluation)

- `populate_workloads.py`  
  One-shot script that:
  - Loads several **binary sentiment datasets** (`imdb`, `rotten_tomatoes`, `glue/sst2`).
  - Creates one `TaskRow` (and FAISS vector) per dataset.
  - Evaluates several **sentiment models** (`vader`, `distilbert`, `bert-imdb`, `twitter-roberta`) on each dataset.
  - Inserts `ResultRow` entries with accuracy and average latency.

- `router_cli.py`  
  CLI that:
  - Sends a **workload description** to the router.
  - Prints the chosen model, neighbors, and candidate results.
  - Optionally asks: **“Run this model on your workload? y/n”** and runs it.

- `label_sanity.py`  
  Simple script to verify that:
  - Dataset labels normalize to binary {0,1}.
  - Model predictions are binary {0,1}.

---

## Installation

Create a virtual environment and install dependencies (you can adjust versions as needed):

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip

pip install \
  fastapi uvicorn[standard] \
  sqlmodel \
  sentence-transformers \
  faiss-cpu \
  transformers \
  huggingface_hub \
  datasets \
  vaderSentiment \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
