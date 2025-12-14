import dask.dataframe as dd
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from tqdm import tqdm
import os

PARQUET_FILE = "natural_instructions_sample_balanced.parquet"
OUTPUT_FILE = "llm_inference_results.parquet"
MODEL_NAME = "google/flan-t5-xxl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_BATCH_SIZE = 1
MAX_NEW_TOKENS = 128
N_PARTITIONS = 500

CHUNKED_SAVE = True
SAVE_DIR = "results_chunks_" + MODEL_NAME.replace("/", "_")
os.makedirs(SAVE_DIR, exist_ok=True)

# Include targets
cols = ["id", "task_name", "task_family", "definition", "inputs", "targets"]


existing_parts = set()
for f in os.listdir(SAVE_DIR):
    if f.startswith("results_part_") and f.endswith(".parquet"):
        try:
            idx = int(f.split("_")[-1].split(".")[0])
            existing_parts.add(idx)
        except:
            pass

print(f"Auto-resume: Found {len(existing_parts)} completed partitions.")

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)
model.eval()

print("Reading dataset lazily with Dask...")
ddf = dd.read_parquet(PARQUET_FILE, columns=cols).repartition(npartitions=N_PARTITIONS)
print(f"Dataset has {len(ddf):,} rows across {ddf.npartitions} partitions")

for part_idx in tqdm(range(ddf.npartitions), desc="Processing partitions"):

    # Skip completed partitions
    if part_idx in existing_parts:
        # print(f"Skipping partition {part_idx}, already completed.")
        continue

    print(f"\n--- Processing partition {part_idx} ---")
    chunk_df = ddf.get_partition(part_idx).compute()

    if len(chunk_df) == 0:
        continue

    # LLM input = definition + inputs
    llm_texts = (chunk_df["definition"] + " " + chunk_df["inputs"]).tolist()

    all_results = []
    i = 0

    while i < len(llm_texts):
        batch_size = TARGET_BATCH_SIZE
        batch_success = False

        while not batch_success:
            try:
                batch_texts = llm_texts[i:i+batch_size]

                # Tokenize
                inputs = tokenizer(batch_texts, return_tensors="pt",
                                   padding=True, truncation=True).to(DEVICE)

                # Run model
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS
                )
                end_time = time.time()

                # Decode
                batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_latency = (end_time - start_time) / len(batch_texts)

                # Add to results
                for j, text in enumerate(batch_texts):
                    row = chunk_df.iloc[i+j]
                    all_results.append({
                        "id": row["id"],
                        "task_name": row["task_name"],
                        "task_family": row["task_family"],
                        "definition": row["definition"],
                        "inputs": row["inputs"],
                        "targets": row["targets"],
                        "output_text": batch_outputs[j],
                        "latency_sec": batch_latency,
                    })

                batch_success = True
                i += batch_size

            except RuntimeError as e:
                if "out of memory" in str(e):
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"OOM detected. Reducing batch size to {batch_size}")
                else:
                    raise e

    # Save partition results
    part_file = os.path.join(SAVE_DIR, f"results_part_{part_idx}.parquet")
    pd.DataFrame(all_results).to_parquet(part_file, index=False)
    print(f"Saved partition {part_idx} -> {part_file}")

print("Inference completed.")
