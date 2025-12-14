import dask.dataframe as dd
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from tqdm import tqdm
import os

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

PARQUET_FILE = "natural_instructions_sample_balanced.parquet"
OUTPUT_FILE = "llm_inference_results.parquet"


MODEL_NAME = "allenai/tk-instruct-3b-def"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
N_PARTITIONS = 100

SAVE_DIR = "results_chunks_" + MODEL_NAME.replace("/", "_")
os.makedirs(SAVE_DIR, exist_ok=True)

cols = ["id", "task_name", "task_family", "definition", "inputs", "targets"]

existing_parts = {
    int(f.split("_")[-1].split(".")[0])
    for f in os.listdir(SAVE_DIR)
    if f.startswith("results_part_") and f.endswith(".parquet")
}

print(f"Auto-resume: Found {len(existing_parts)} completed partitions.")


name = MODEL_NAME.lower()

IS_TK_INSTRUCT  = "tk-instruct" in name
IS_T5GEMMA_UL2  = "t5gemma" in name and "ul2" in name
IS_T5_11        = "t5_1.1" in name   


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

print(f"Loaded {MODEL_NAME} on {DEVICE}")

print("\nReading dataset lazily with Dask...")
ddf = dd.read_parquet(PARQUET_FILE, columns=cols).repartition(npartitions=N_PARTITIONS)
print(f"Dataset has {len(ddf):,} rows across {ddf.npartitions} partitions")


for part_idx in tqdm(range(ddf.npartitions), desc="Processing partitions"):

    if part_idx in existing_parts:
        continue

    print(f"\n--- Processing partition {part_idx} ---")
    chunk_df = ddf.get_partition(part_idx).compute()
    if len(chunk_df) == 0:
        continue

    # Prompt format (good for all three models)
    llm_texts = [
        f"Instruction: {row.definition}\nInput: {row.inputs}\nOutput:"
        for _, row in chunk_df.iterrows()
    ]

    all_results = []
    i = 0

    while i < len(llm_texts):
        batch_size = TARGET_BATCH_SIZE
        batch_success = False

        while not batch_success:
            try:
                batch_texts = llm_texts[i:i+batch_size]

                max_len = 2048 if (IS_T5_11 or IS_T5GEMMA_UL2 or IS_TK_INSTRUCT) else 1024

                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_len
                )
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1,
                    do_sample=False
                )
                end_time = time.time()

                batch_latency = (end_time - start_time) / len(batch_texts)
                batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for j in range(len(batch_texts)):
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
                if "out of memory" in str(e).lower():
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"OOM -> reducing batch size to {batch_size}")
                else:
                    raise e

    part_file = os.path.join(SAVE_DIR, f"results_part_{part_idx}.parquet")
    pd.DataFrame(all_results).to_parquet(part_file, index=False)
    print(f"Saved partition {part_idx} -> {part_file}")

print("Inference completed.")
