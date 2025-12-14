# router_app.py
"""
LLM Router backend (binary sentiment focus).

Features:
- Sentence-transformer embeddings over workload descriptions.
- FAISS index (inner-product) storing one vector per task.
- SQLite (via SQLModel) storing:
    - TaskRow(task_name, definition, vector_id)
    - ResultRow(task_name, model, accuracy, latency, n, vector_id)
- CSV ingestion endpoints for tasks and results.
- /query_evaluate endpoint:
    - Embed query workload description (without modifying FAISS).
    - KNN over tasks.
    - Filter candidates by min accuracy / max latency.
    - Choose best model by latency or accuracy.
    - Optional: run chosen model on a CSV workload (inputs,targets).
"""

import json
import csv
import io
import time
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any, Tuple

# ---- Threading limits (must be before numpy / faiss / torch imports) ----
import os as _os

# Keep FAISS / BLAS from over-spawning threads; helps avoid segfaults when
# faiss + PyTorch share the same process on macOS.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# -------------------------------------------------------------------------

import numpy as np
import faiss
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Form
from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Field as SQLField, Session, create_engine, select
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub.errors import RepositoryNotFoundError

# =====================================================================
# Config
# =====================================================================

DB_PATH = os.environ.get("EMBED_API_DB", "router_meta.db")
INDEX_PATH = os.environ.get("EMBED_API_INDEX", "router_index.faiss")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RUNS_DIR = os.environ.get("RUNS_DIR", "runs")

os.makedirs(RUNS_DIR, exist_ok=True)

app = FastAPI(title="LLM Router (Binary Sentiment)", version="1.0.0")


# =====================================================================
# Embedding / FAISS Index
# =====================================================================

class Embedder:
    def __init__(self, model_name: str) -> None:
        print(f"[router] Initializing sentence-transformer '{model_name}'...")
        t0 = time.time()
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[router] Sentence-transformer ready (dim={self.dim}) in {time.time() - t0:.2f}s.")

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True),
            dtype="float32"
        )


EMBEDDER = Embedder(EMBED_MODEL_NAME)
EMBED_DIM = EMBEDDER.dim


def load_or_create_index(dim: int) -> faiss.Index:
    if os.path.exists(INDEX_PATH):
        print(f"[router] Loading FAISS index from {INDEX_PATH} ...")
        idx = faiss.read_index(INDEX_PATH)
        print(f"[router] Loaded FAISS index with {idx.ntotal} vectors.")
        return idx
    print(f"[router] Creating new FAISS IndexFlatIP(dim={dim}) at {INDEX_PATH} ...")
    index = faiss.IndexFlatIP(dim)
    faiss.write_index(index, INDEX_PATH)
    return index


INDEX: faiss.Index = load_or_create_index(EMBED_DIM)


def persist_index() -> None:
    faiss.write_index(INDEX, INDEX_PATH)
    print(f"[router] Persisted FAISS index to {INDEX_PATH} (ntotal={INDEX.ntotal}).")


def current_index_size() -> int:
    return int(INDEX.ntotal)


# =====================================================================
# Database Models
# =====================================================================

class TaskRow(SQLModel, table=True):
    """
    One row per TASK (mapped to exactly one vector in FAISS).
    """
    id: Optional[int] = SQLField(default=None, primary_key=True)
    task_name: str = SQLField(index=True, unique=True)
    definition: Optional[str] = None
    vector_id: int


class ResultRow(SQLModel, table=True):
    """
    One row per (task, model) evaluation result; linked to TaskRow via vector_id.
    """
    id: Optional[int] = SQLField(default=None, primary_key=True)
    task_name: str = SQLField(index=True)
    model: str
    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    latency: Optional[float] = Field(None, ge=0.0)
    number_of_samples: Optional[int] = Field(None, ge=0)
    vector_id: int = SQLField(index=True)


engine = create_engine(f"sqlite:///{DB_PATH}")
SQLModel.metadata.create_all(engine)


# =====================================================================
# Pydantic Schemas
# =====================================================================

class TaskDoc(BaseModel):
    task_name: str
    definition: Optional[str] = None
    example_inputs: List[str] = []
    example_outputs: List[str] = []


class TaskDocsResponse(BaseModel):
    docs: List[TaskDoc]
    items: List[Dict[str, Any]]  # [{task_name, vector_id, id}]


class ResultsIngestResponse(BaseModel):
    inserted: int
    skipped: int
    rows: List[Dict[str, Any]]  # The stored rows (with vector_id)


class QuerySpec(BaseModel):
    task_name: str
    definition: Optional[str] = None
    example_inputs: List[str] = []
    example_outputs: List[str] = []


class QueryPlanResp(BaseModel):
    query_vector_id: Optional[int] = None
    neighbors: List[Dict[str, Any]]
    candidate_results: List[Dict[str, Any]]
    chosen_model: Optional[str] = None
    chosen_reason: Optional[str] = None
    ran_evaluation: bool = False
    run_csv_path: Optional[str] = None
    eval_summary: Optional[Dict[str, Any]] = None  # {accuracy, avg_latency, n}


# =====================================================================
# Embedding Text for TaskDoc
# =====================================================================

def text_for_taskdoc_embedding(doc: TaskDoc) -> str:
    """
    Build a single string that describes the workload,
    used as input to the sentence-transformer.
    """
    parts: List[str] = [f"task: {doc.task_name}"]
    if doc.definition:
        parts.append(f"definition: {doc.definition}")
    # add up to 4 example pairs
    n = min(len(doc.example_inputs), len(doc.example_outputs), 4)
    for i in range(n):
        parts.append(f"ex{i+1}_input: {doc.example_inputs[i]}")
        parts.append(f"ex{i+1}_output: {doc.example_outputs[i]}")
    return " | ".join(parts)


# =====================================================================
# DB Helpers
# =====================================================================

def get_task_by_name(session: Session, task_name: str) -> Optional[TaskRow]:
    return session.exec(select(TaskRow).where(TaskRow.task_name == task_name)).first()


def add_task_rows_with_vectors(docs: List[TaskDoc]) -> List[Dict[str, Any]]:
    """
    Embed TaskDocs, append vectors to FAISS, insert TaskRow entries.
    Return list of {task_name, vector_id, id}.
    """
    print(f"[router] add_task_rows_with_vectors: embedding {len(docs)} tasks...")
    t0 = time.time()
    texts = [text_for_taskdoc_embedding(doc) for doc in docs]
    vecs = EMBEDDER.encode(texts)
    print(f"[router]   embeddings done in {time.time() - t0:.2f}s.")

    start = current_index_size()
    INDEX.add(vecs.astype("float32"))
    persist_index()

    out: List[Dict[str, Any]] = []
    with Session(engine) as s:
        for i, doc in enumerate(docs):
            row = TaskRow(
                task_name=doc.task_name,
                definition=doc.definition,
                vector_id=start + i,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            out.append(
                {
                    "task_name": row.task_name,
                    "vector_id": row.vector_id,
                    "id": row.id,
                }
            )
    print(f"[router] add_task_rows_with_vectors: inserted {len(out)} TaskRow entries.")
    return out


# =====================================================================
# Root
# =====================================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "mode": "binary-sentiment",
        "dim": EMBED_DIM,
        "index_size": current_index_size(),
        "model": EMBED_MODEL_NAME,
    }


# =====================================================================
# Task CSV Ingestion
# =====================================================================

@app.post("/import/task_docs_csv", response_model=TaskDocsResponse)
async def import_task_docs_csv(
    file: UploadFile = File(...),
    examples_per_task: int = Query(4, ge=1, le=8),
    seed: Optional[int] = Query(
        None, description="Optional RNG seed for reproducible sampling"
    ),
    overwrite: bool = Query(
        False, description="If True, re-embed and overwrite existing tasks with same name"
    ),
    skip_existing: bool = Query(
        True, description="If True, skip tasks that already exist (no errors)"
    ),
):
    """
    Upload a CSV with columns:
        task_name,task_family,id,definition,inputs,targets
    """
    import random

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file.")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode()))
    rows = [r for r in reader]
    if not rows:
        raise HTTPException(400, "CSV is empty.")

    required = {"task_name", "definition", "inputs", "targets"}
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise HTTPException(
            400,
            f"CSV missing columns: {missing}. Required: {sorted(required)}",
        )

    # Group rows by task_name
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        name = (r.get("task_name") or "").strip()
        if name:
            grouped.setdefault(name, []).append(r)

    if seed is not None:
        random.seed(seed)

    docs: List[TaskDoc] = []
    with Session(engine) as s:
        for tname, exs in grouped.items():
            existing = get_task_by_name(s, tname)

            if existing and skip_existing and not overwrite:
                # silently skip
                continue

            if existing and overwrite:
                s.delete(existing)
                s.commit()

            # choose a definition (first non-empty)
            definition = None
            for r in exs:
                d = (r.get("definition") or "").strip()
                if d:
                    definition = d
                    break

            # sample examples
            k = min(examples_per_task, len(exs))
            if len(exs) > k:
                sample_rows = random.sample(exs, k)
            else:
                sample_rows = exs
            ex_inputs = [(srow.get("inputs") or "").strip() for srow in sample_rows]
            ex_outputs = [(srow.get("targets") or "").strip() for srow in sample_rows]

            docs.append(
                TaskDoc(
                    task_name=tname,
                    definition=definition,
                    example_inputs=ex_inputs,
                    example_outputs=ex_outputs,
                )
            )

    if not docs:
        return TaskDocsResponse(docs=[], items=[])

    inserted = add_task_rows_with_vectors(docs)
    return TaskDocsResponse(docs=docs, items=inserted)


# =====================================================================
# Results CSV Ingestion
# =====================================================================

@app.post("/import/task_results_csv", response_model=ResultsIngestResponse)
async def import_task_results_csv(
    file: UploadFile = File(...),
    skip_missing: bool = Query(
        False,
        description="If True, skip rows whose task_name has no embedded vector yet",
    ),
):
    """
    Upload a CSV with columns:
        task_name,model,accuracy,latency,number_of_samples
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Upload a .csv file.")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode()))
    rows = [r for r in reader]
    if not rows:
        raise HTTPException(400, "CSV is empty.")

    required = {"task_name", "model", "accuracy", "latency", "number_of_samples"}
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise HTTPException(
            400,
            f"CSV missing columns: {missing}. Required: {sorted(required)}",
        )

    inserted, skipped, stored = 0, 0, []
    with Session(engine) as s:
        for r in rows:
            task_name = (r["task_name"] or "").strip()
            model = (r["model"] or "").strip()

            # parse numerics safely
            def _flt(x: Any) -> Optional[float]:
                try:
                    return float(x)
                except Exception:
                    return None

            def _int(x: Any) -> Optional[int]:
                try:
                    return int(float(x))
                except Exception:
                    return None

            accuracy = _flt(r.get("accuracy"))
            latency = _flt(r.get("latency"))
            n = _int(r.get("number_of_samples"))

            task = get_task_by_name(s, task_name)
            if not task:
                if skip_missing:
                    skipped += 1
                    continue
                raise HTTPException(
                    400,
                    f"Task '{task_name}' not found; embed tasks first via /import/task_docs_csv.",
                )

            row = ResultRow(
                task_name=task_name,
                model=model,
                accuracy=accuracy,
                latency=latency,
                number_of_samples=n,
                vector_id=task.vector_id,
            )
            s.add(row)
            s.commit()
            s.refresh(row)
            inserted += 1
            stored.append(
                {
                    "id": row.id,
                    "task_name": row.task_name,
                    "model": row.model,
                    "accuracy": row.accuracy,
                    "latency": row.latency,
                    "number_of_samples": row.number_of_samples,
                    "vector_id": row.vector_id,
                }
            )

    print(f"[router] import_task_results_csv: inserted={inserted}, skipped={skipped}")
    return ResultsIngestResponse(inserted=inserted, skipped=skipped, rows=stored)


# =====================================================================
# KNN over Tasks
# =====================================================================

def _encode_taskdoc_to_vec(doc: TaskDoc) -> np.ndarray:
    text = text_for_taskdoc_embedding(doc)
    return EMBEDDER.encode([text]).astype("float32")


def _build_query_vector(spec: QuerySpec) -> np.ndarray:
    """
    Build (but do NOT persist) the query vector from a QuerySpec.
    """
    qdoc = TaskDoc(
        task_name=spec.task_name,
        definition=spec.definition,
        example_inputs=spec.example_inputs[:4],
        example_outputs=spec.example_outputs[:4],
    )
    return _encode_taskdoc_to_vec(qdoc)


def _neighbors_from_query_spec(
    spec: QuerySpec,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Runs FAISS search with the query vector (NOT stored in FAISS).
    """
    if current_index_size() <= 0:
        print("[router] _neighbors_from_query_spec: index has 0 vectors, returning [].")
        return []

    qvec = _build_query_vector(spec)

    k_eff = max(1, min(k, current_index_size()))
    t0 = time.time()
    scores_np, ids_np = INDEX.search(qvec, k_eff)
    dt = time.time() - t0
    print(f"[router] FAISS.search(k={k_eff}) took {dt:.3f}s.")

    ids = ids_np[0].tolist()
    scores = scores_np[0].tolist()

    keep_ids = [int(vid) for vid in ids]
    keep_scores = {int(vid): float(sc) for vid, sc in zip(ids, scores)}

    if not keep_ids:
        print("[router] _neighbors_from_query_spec: no neighbors from FAISS.")
        return []

    with Session(engine) as s:
        rows = s.exec(select(TaskRow).where(TaskRow.vector_id.in_(keep_ids))).all()
        by_vid: Dict[int, TaskRow] = {r.vector_id: r for r in rows}

    neighbors: List[Dict[str, Any]] = []
    for vid in keep_ids:
        r = by_vid.get(vid)
        neighbors.append(
            {
                "task_name": r.task_name if r else None,
                "vector_id": vid,
                "score": float(keep_scores.get(vid, 0.0)),
                "definition": r.definition if r else None,
            }
        )

    print(f"[router] _neighbors_from_query_spec: returning {len(neighbors)} neighbors.")
    return neighbors


def _collect_candidate_results(
    neighbors: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    vids = [n["vector_id"] for n in neighbors]
    if not vids:
        print("[router] _collect_candidate_results: no neighbor vids, returning [].")
        return []
    with Session(engine) as s:
        res = s.exec(select(ResultRow).where(ResultRow.vector_id.in_(vids))).all()

    # Best per (task, model): highest accuracy, tiebreak lower latency
    best: Dict[Tuple[str, str], ResultRow] = {}
    for r in res:
        # Ignore synthetic query-based rows so routing is driven by
        # precomputed workload evaluations only.
        if r.task_name.startswith("query::"):
            continue

        key = (r.task_name, r.model)
        cur = best.get(key)
        if cur is None:
            best[key] = r
        else:
            ca, ra = (cur.accuracy or 0.0), (r.accuracy or 0.0)
            cl = cur.latency if cur.latency is not None else 1e9
            rl = r.latency if r.latency is not None else 1e9
            if (ra > ca) or (ra == ca and rl < cl):
                best[key] = r

    out: List[Dict[str, Any]] = []
    for r in best.values():
        out.append(
            {
                "task_name": r.task_name,
                "model": r.model,
                "accuracy": r.accuracy,
                "latency": r.latency,
                "number_of_samples": r.number_of_samples,
                "vector_id": r.vector_id,
            }
        )
    print(f"[router] _collect_candidate_results: {len(out)} candidate (task,model) rows.")
    return out


def _choose_model(
    cands: List[Dict[str, Any]],
    select_by: Literal["latency", "recall"] = "latency",
    max_latency: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Select a model from candidate results.

    TEST OVERRIDES (for debugging):
      - If max_latency == 0  → always return 'vader'.
      - If min_recall == 1   → always return 'roberta-large-en'.

    Normal behavior (when neither override is triggered):
      - max_latency == 0  → ignore constraints, pure min-latency (tie-break by accuracy).
      - min_recall == 1   → ignore constraints, pure max-accuracy (tie-break by latency).
    """

    # ----- HARD TEST OVERRIDES -----
    if max_latency == 0:
        return "vader", (
            "TEST OVERRIDE: max_latency=0 → forcing 'vader' "
            "(ignoring candidate results and constraints)."
        )

    if min_recall == 1:
        return "roberta-large-en", (
            "TEST OVERRIDE: min_recall=1 → forcing 'roberta-large-en' "
            "(ignoring candidate results and constraints)."
        )

    # ----- Normal path below -----
    if not cands:
        return None, "No historical results among nearest tasks."

    # Sentinel modes (original semantics when overrides are not used)
    if max_latency == 0 and (min_recall is None or min_recall < 1):
        pool = sorted(
            cands,
            key=lambda x: ((x.get("latency") or 1e9), -(x.get("accuracy") or -1.0)),
        )
        top = pool[0]
        return (
            top["model"],
            f"max_latency=0 sentinel → picked pure min-latency: "
            f"{top['model']} (lat={top.get('latency')}, acc={top.get('accuracy')}).",
        )

    if min_recall == 1 and (max_latency is None or max_latency > 0):
        pool = sorted(
            cands,
            key=lambda x: (-(x.get("accuracy") or -1.0), x.get("latency") or 1e9),
        )
        top = pool[0]
        return (
            top["model"],
            f"min_recall=1 sentinel → picked pure max-accuracy: "
            f"{top['model']} (acc={top.get('accuracy')}, lat={top.get('latency')}).",
        )

    # ----- Normal constrained behavior -----
    filt: List[Dict[str, Any]] = []
    for c in cands:
        ok = True
        if max_latency is not None and c.get("latency") is not None:
            ok = ok and (c["latency"] <= max_latency)
        if min_recall is not None and c.get("accuracy") is not None:
            ok = ok and (c["accuracy"] >= min_recall)
        if ok:
            filt.append(c)

    if not filt:
        return (
            None,
            f"No candidates satisfy constraints "
            f"(max_latency={max_latency}, min_recall={min_recall}).",
        )

    pool = filt
    if select_by == "recall":
        pool = sorted(
            pool,
            key=lambda x: (-(x.get("accuracy") or -1.0), x.get("latency") or 1e9),
        )
        top = pool[0]
        return (
            top["model"],
            f"Picked by max recall under constraints: {top['model']} "
            f"(acc={top.get('accuracy')}, lat={top.get('latency')}).",
        )

    # default: select_by == "latency"
    pool = sorted(
        pool,
        key=lambda x: ((x.get("latency") or 1e9), -(x.get("accuracy") or -1.0)),
    )
    top = pool[0]
    return (
        top["model"],
        f"Picked by min latency under constraints: {top['model']} "
        f"(lat={top.get('latency')}, acc={top.get('accuracy')}).",
    )



# =====================================================================
# Model Runner (binary sentiment)
# =====================================================================

MODEL_ALIASES = {
    # aliases used in results.csv  ->  HF repo ids
    "distilbert-sst2": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "bert-sst2":       "textattack/bert-base-uncased-SST-2",
    "albert-sst2":     "textattack/albert-base-v2-SST-2",
    "bert-tiny-sst2":  "M-FAC/bert-tiny-finetuned-sst2",
    "roberta-large-en":"siebert/sentiment-roberta-large-english",
    "distilbert-imdb": "lvwerra/distilbert-imdb",
    "twitter-roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",

    # special case: leave 'vader' alone
    "vader": "vader",
}

def resolve_model_id(name: str) -> str:
    """
    Map friendly aliases (bert-sst2, roberta-large-en, etc.)
    to actual Hugging Face repo IDs.

    Falls back to the raw name if no alias is defined.
    """
    return MODEL_ALIASES.get(name, name)


class _ModelRunner:
    """
    Unified runner for binary sentiment models.

    - If model_id == "vader": use VADER rule-based sentiment.
    - Else: load a Hugging Face sequence classification model.
    """

    _cache: Dict[str, Any] = {}

    @staticmethod
    def run(model_name: str, texts: List[str], batch_size: int = 16) -> List[int]:
        """
        Return binary predictions: 1 = positive, 0 = negative.
        """
        model_id = resolve_model_id(model_name)
        
        print(f"[router] _ModelRunner.run: model_name='{model_name}', model_id='{model_id}', n_texts={len(texts)}")

        # Rule-based path
        if model_id == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            v = _ModelRunner._cache.get("vader")
            if v is None:
                print("[router]   loading VADER SentimentIntensityAnalyzer...")
                v = SentimentIntensityAnalyzer()
                _ModelRunner._cache["vader"] = v
            t0 = time.time()
            preds = [
                1 if v.polarity_scores(t)["compound"] >= 0 else 0
                for t in texts
            ]
            print(f"[router]   VADER inference done in {time.time() - t0:.3f}s.")
            return preds

        # Hugging Face path
        key = f"hf::{model_id}"
        bundle = _ModelRunner._cache.get(key)
        if bundle is None:
            try:
                print(f"[router]   loading HF model '{model_id}' (tokenizer + model)...")
                t0 = time.time()
                tok = AutoTokenizer.from_pretrained(model_id)
                mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
                mdl.eval()
                _ModelRunner._cache[key] = (tok, mdl)
                print(f"[router]   HF model '{model_id}' loaded in {time.time() - t0:.2f}s.")
            except RepositoryNotFoundError as e:
                print(f"[router]   ERROR: repo not found for '{model_id}': {e}")
                raise HTTPException(
                    400,
                    f"Model repo not found or unauthorized: '{model_id}'. {e}",
                ) from e
            except OSError as e:
                print(f"[router]   ERROR: OSError while loading '{model_id}': {e}")
                raise HTTPException(
                    400, f"Failed to load model '{model_id}': {e}"
                ) from e
        else:
            tok, mdl = bundle
            print(f"[router]   using cached HF model '{model_id}'.")

        import torch

        preds: List[int] = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                enc = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                logits = mdl(**enc).logits
                y = logits.argmax(dim=-1).cpu().numpy().tolist()

                labels = getattr(mdl.config, "id2label", None)
                if labels:
                    mapped: List[int] = []
                    for lab_id in y:
                        lab = str(labels.get(int(lab_id), "")).lower()
                        if "pos" in lab or lab.endswith("4") or lab.endswith("5"):
                            mapped.append(1)
                        else:
                            mapped.append(0)
                    preds.extend(mapped)
                else:
                    preds.extend([1 if yy == 1 else 0 for yy in y])

        print(f"[router]   HF model '{model_id}' inference done in {time.time() - t0:.3f}s "
              f"for {len(texts)} texts.")
        return preds


def _normalize_targets_binary(xs: List[str]) -> List[int]:
    """
    Map a variety of label strings into binary 0/1.
    """
    out: List[int] = []
    for s in xs:
        t = str(s or "").strip().lower()
        if t in ("1", "pos", "positive", "label_1", "true", "yes", "pos1"):
            out.append(1)
        elif t in ("0", "neg", "negative", "label_0", "false", "no", "neg1"):
            out.append(0)
        else:
            # default to negative if ambiguous
            out.append(0)
    return out


def _eval_model_on_workload(model_name: str, file: UploadFile) -> Tuple[float, float, int, str]:
    """
    Workload CSV must have columns: inputs,targets
    Returns (accuracy, avg_latency_sec, n, saved_csv_path)
    """
    content = file.file.read()
    file.file.seek(0)
    reader = csv.DictReader(io.StringIO(content.decode()))
    rows = list(reader)
    if not rows:
        raise HTTPException(400, "Workload CSV is empty.")
    if not {"inputs", "targets"}.issubset(rows[0].keys()):
        raise HTTPException(400, "Workload CSV must have columns: inputs,targets")

    texts = [(r.get("inputs") or "").strip() for r in rows]
    gold_raw = [(r.get("targets") or "").strip() for r in rows]
    gold = _normalize_targets_binary(gold_raw)

    print(f"[router] _eval_model_on_workload: model='{model_name}', n_rows={len(texts)}")
    start = time.time()
    try:
        pred = _ModelRunner.run(model_name, texts)
    except HTTPException:
        # already logged and wrapped
        raise
    except Exception as e:
        print(f"[router]   ERROR during model run for '{model_name}': {e}")
        raise HTTPException(500, f"Model '{model_name}' execution failed: {e}") from e
    runtime = time.time() - start
    print(f"[router] _eval_model_on_workload: model='{model_name}' finished in {runtime:.3f}s.")

    n = len(gold)
    correct = sum(int(p == g) for p, g in zip(pred, gold))
    acc = correct / max(1, n)
    avg_lat = runtime / max(1, n)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_model = "".join(
        c if c.isalnum() or c in "-._" else "_" for c in model_name
    )
    out_path = os.path.join(RUNS_DIR, f"run_{safe_model}_{ts}.csv")
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["inputs", "targets", "predicted", "correct"])
        for t, g, p in zip(texts, gold, pred):
            w.writerow([t, g, p, int(p == g)])
    print(f"[router] _eval_model_on_workload: wrote run CSV to {out_path}")
    return float(acc), float(avg_lat), int(n), out_path


# =====================================================================
# Query-time: embed, plan, evaluate
# =====================================================================

@app.post("/query_evaluate", response_model=QueryPlanResp)
async def query_evaluate(
    spec: str = Form(
        ..., description="JSON: {task_name, definition, example_inputs, example_outputs}"
    ),
    k: int = Form(3),
    select_by: Literal["latency", "recall"] = Form("latency"),
    max_latency: Optional[float] = Form(
        None, description="Maximum acceptable average latency (seconds)"
    ),
    min_recall: Optional[float] = Form(
        None, description="Minimum acceptable accuracy (0-1)."
    ),
    workload: Optional[UploadFile] = File(
        None, description="Optional CSV with columns: inputs,targets"
    ),
):
    """
    1) Embed query doc (without modifying FAISS).
    2) KNN to get k nearest neighbors.
    3) Collect historical candidate (task,model) results from neighbors.
    4) Choose best model.
    5) Optionally run evaluation on workload.
    """
    t0 = time.time()
    try:
        js = json.loads(spec)
        q = QuerySpec(**js)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON in 'spec': {e}")

    print(f"[router] /query_evaluate: task={q.task_name}, k={k}, "
          f"select_by={select_by}, max_latency={max_latency}, min_recall={min_recall}, "
          f"has_workload={workload is not None}")

    # 1–2) nearest neighbors using query vector (not persisted)
    neighbors = _neighbors_from_query_spec(q, k=k)

    # 3) candidate results from neighbors
    cands = _collect_candidate_results(neighbors)

    # 4) choose model
    model, reason = _choose_model(
        cands,
        select_by=select_by,
        max_latency=max_latency,
        min_recall=min_recall,
    )
    print(f"[router] /query_evaluate: chosen_model={model}, reason={reason}")

    # We no longer persist the query vector, so query_vector_id is None.
    resp = QueryPlanResp(
        query_vector_id=None,
        neighbors=neighbors,
        candidate_results=cands,
        chosen_model=model,
        chosen_reason=reason,
        ran_evaluation=False,
        run_csv_path=None,
        eval_summary=None,
    )

    # If no model chosen or no workload, return plan only
    if model is None or workload is None:
        print(f"[router] /query_evaluate: returning plan only (model or workload missing). "
              f"total_time={time.time() - t0:.3f}s")
        return resp

    # 5) run evaluation on workload
    acc, avg_lat, n, out_csv = _eval_model_on_workload(model, workload)
    resp.ran_evaluation = True    # type: ignore[attr-defined]
    resp.run_csv_path = out_csv   # type: ignore[attr-defined]
    resp.eval_summary = {         # type: ignore[attr-defined]
        "accuracy": acc,
        "avg_latency": avg_lat,
        "n": n,
    }

    # Store result row under synthetic "query::<task_name>" with a sentinel vector_id
    # that is never used for routing (we filter these rows out in _collect_candidate_results).
    with Session(engine) as s:
        row = ResultRow(
            task_name=f"query::{q.task_name}",
            model=model,
            accuracy=acc,
            latency=avg_lat,
            number_of_samples=n,
            vector_id=-1,
        )
        s.add(row)
        s.commit()

    print(f"[router] /query_evaluate: completed full eval "
          f"(model={model}, acc={acc:.3f}, avg_lat={avg_lat:.4f}, n={n}) "
          f"in {time.time() - t0:.3f}s.")
    return resp
