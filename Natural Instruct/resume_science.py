import dask.dataframe as dd
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
from tqdm import tqdm
import os

PARQUET_FILE = "natural_instructions_sample_balanced.parquet"
OUTPUT_FILE = "llm_inference_results.parquet"

# can reuse for these modesls?
#  facebook/bart-large
#  google/pegasus-large
#  microsoft/prophetnet-large-uncased-xlarge 
#  facebook/blenderbot-3B        
#  bigscience/T0_3B       
#  allenai/led-base-16384
#  allenai/led-large-16384

MODEL_NAME = "bigscience/T0_3B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_BATCH_SIZE = 1
MAX_NEW_TOKENS = 128
N_PARTITIONS = 500

# For chunked saving
CHUNKED_SAVE = True
SAVE_DIR = "results_chunks_" + MODEL_NAME.replace("/", "_")
os.makedirs(SAVE_DIR, exist_ok=True)

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

def detect_model_type(name):
    name_low = name.lower()
    if "led" in name_low:
        return "led"
    else:
        return "seq2seq"  

MODEL_TYPE = detect_model_type(MODEL_NAME)
print("MODEL_TYPE =", MODEL_TYPE)

print("\nLoading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False if "pegasus" in MODEL_NAME else True)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)
model.eval()

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

    # LLM text construction
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

                if MODEL_TYPE == "led":
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=4096
                    ).to(DEVICE)

                    # LED requires global attention mask
                    global_attention_mask = torch.zeros_like(inputs["attention_mask"])
                    global_attention_mask[:, 0] = 1

                    generate_kwargs = dict(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        global_attention_mask=global_attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        attention_window=4096,
                    )

                else:
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(DEVICE)

                    generate_kwargs = dict(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS
                    )

                start_time = time.time()
                outputs = model.generate(**generate_kwargs)
                end_time = time.time()

                batch_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch_latency = (end_time - start_time) / len(batch_texts)

                # Save results
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

    part_file = os.path.join(SAVE_DIR, f"results_part_{part_idx}.parquet")
    pd.DataFrame(all_results).to_parquet(part_file, index=False)
    print(f"Saved partition {part_idx} -> {part_file}")

print("Inference completed.")
