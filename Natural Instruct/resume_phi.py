import dask.dataframe as dd
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
import os

PARQUET_FILE = "natural_instructions_sample_balanced.parquet"

MODEL_NAME = "microsoft/Phi-4-mini-instruct"

SAVE_DIR = f"results_{MODEL_NAME.replace('/', '_')}"
os.makedirs(SAVE_DIR, exist_ok=True)

N_PARTITIONS = 100
TARGET_BATCH_SIZE = 2
MAX_NEW_TOKENS = 200

DEVICE_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

cols = ["id", "task_name", "task_family", "definition", "inputs", "targets"]


PART_START = int(os.getenv("PART_START", 0))
PART_END   = int(os.getenv("PART_END", N_PARTITIONS - 1))

print(f"\n>>> Running partition range: {PART_START} -> {PART_END}\n")


print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.padding_side = "left"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
model.eval()

print("Model loaded.")


ddf = dd.read_parquet(PARQUET_FILE, columns=cols).repartition(npartitions=N_PARTITIONS)
print(f"Dataset size: {len(ddf):,} rows across {ddf.npartitions} partitions.")


existing_parts = {
    int(f.split("_")[-1].split(".")[0])
    for f in os.listdir(SAVE_DIR)
    if f.startswith("results_part_")
}

print(f"Auto-resume: Found {len(existing_parts)} completed partitions.")


def build_chat(defn, inp):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {
            "role": "user",
            "content": (
                f"Task definition:\n{defn}\n\n"
                f"Input:\n{inp}\n\n"
                "Return only the correct answer for this task."
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


for part_idx in tqdm(range(PART_START, PART_END + 1), desc="Processing partitions"):

    if part_idx in existing_parts:
        continue

    print(f"\n--- Processing partition {part_idx} ---")

    chunk = ddf.get_partition(part_idx).compute()
    if len(chunk) == 0:
        continue

    prompts = [
        build_chat(row.definition, row.inputs)
        for _, row in chunk.iterrows()
    ]

    results = []
    i = 0


    while i < len(prompts):

        batch_size = TARGET_BATCH_SIZE
        batch_success = False

        while not batch_success:

            try:
                batch = prompts[i : i + batch_size]

                encoded = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(model.device)

                start = time.time()
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                end = time.time()

                latency = (end - start) / len(batch)

                # Parse outputs
                for j, out_ids in enumerate(outputs):
                    row = chunk.iloc[i + j]

                    decoded = tokenizer.decode(out_ids, skip_special_tokens=False)

                    if "<|assistant|>" in decoded:
                        assistant = decoded.split("<|assistant|>")[-1]
                        assistant = assistant.split("<|end|>")[0]
                        answer = assistant.strip()
                    else:
                        answer = decoded.strip()

                    results.append({
                        "id": row["id"],
                        "task_name": row["task_name"],
                        "task_family": row["task_family"],
                        "definition": row["definition"],
                        "inputs": row["inputs"],
                        "targets": row["targets"],
                        "output_text": answer,
                        "latency_sec": latency,
                    })

                i += batch_size
                batch_success = True

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_size = max(1, batch_size // 2)
                    torch.cuda.empty_cache()
                    print(f"OOM -> reducing batch size to {batch_size}")
                else:
                    raise e


    out_file = os.path.join(SAVE_DIR, f"results_part_{part_idx}.parquet")
    pd.DataFrame(results).to_parquet(out_file, index=False)
    print(f"Saved -> {out_file}")

print("\nALL DONE.")
