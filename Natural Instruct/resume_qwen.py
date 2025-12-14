import dask.dataframe as dd
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
import os

PARQUET_FILE = "natural_instructions_sample_balanced.parquet"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

SAVE_DIR = f"results_chunks_{MODEL_NAME.replace('/', '_')}"
os.makedirs(SAVE_DIR, exist_ok=True)

N_PARTITIONS = 100
TARGET_BATCH_SIZE = 4  
MAX_NEW_TOKENS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


print("\nLoading Qwen2 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=DTYPE
)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()

print("Model loaded.\n")


ddf = dd.read_parquet(PARQUET_FILE).repartition(npartitions=N_PARTITIONS)
print(f"Dataset size: {len(ddf):,} rows across {ddf.npartitions} partitions.\n")


existing_parts = {
    int(f.split("_")[-1].split(".")[0])
    for f in os.listdir(SAVE_DIR)
    if f.startswith("results_part_")
}

print(f"Auto-resume: Found {len(existing_parts)} completed partitions.\n")


def build_messages(defn, inp):
    return [
        {
            "role": "user",
            "content": (
                f"Here is a task definition and input.\n\n"
                f"Task:\n{defn}\n\n"
                f"Input:\n{inp}\n\n"
                f"Provide only the final answer."
            )
        }
    ]


for part_idx in tqdm(range(ddf.npartitions), desc="Processing partitions"):

    if part_idx in existing_parts:
        continue

    print(f"\n--- Processing partition {part_idx} ---")

    chunk = ddf.get_partition(part_idx).compute()
    if len(chunk) == 0:
        continue

    # Build messages for each row
    prompts = [
        build_messages(row.definition, row.inputs)
        for _, row in chunk.iterrows()
    ]

    results = []
    i = 0

    while i < len(prompts):

        batch_size = TARGET_BATCH_SIZE
        batch_success = False

        while not batch_success:
            try:
                batch_prompts = prompts[i:i + batch_size]

                prompt_texts = tokenizer.apply_chat_template(
                    batch_prompts,
                    add_generation_prompt=True,
                    tokenize=False   # returns raw strings
                )
                encoded = tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096
                ).to(model.device)

                start_time = time.time()
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                end_time = time.time()

                batch_latency = (end_time - start_time) / len(batch_prompts)

                input_len = encoded["input_ids"].shape[1]

                for j in range(len(batch_prompts)):
                    row = chunk.iloc[i + j]
                    continuation = outputs[j][input_len:]
                    text = tokenizer.decode(continuation, skip_special_tokens=True).strip()

                    results.append({
                        "id": row["id"],
                        "task_name": row["task_name"],
                        "task_family": row["task_family"],
                        "definition": row["definition"],
                        "inputs": row["inputs"],
                        "targets": row["targets"],
                        "output_text": text,
                        "latency_sec": batch_latency,
                    })

                i += batch_size
                batch_success = True

            except RuntimeError as e:
                # Out of memory -> shrink batch
                if "out of memory" in str(e).lower():
                    batch_size = max(batch_size // 2, 1)
                    torch.cuda.empty_cache()
                    print(f"OOM -> reducing batch size to {batch_size}")
                    if batch_size == 1:
                        # If still failing, raise
                        continue
                else:
                    raise e
    out_file = os.path.join(SAVE_DIR, f"results_part_{part_idx}.parquet")
    pd.DataFrame(results).to_parquet(out_file, index=False)
    print(f"Saved -> {out_file}")

print("\nInference complete.")
