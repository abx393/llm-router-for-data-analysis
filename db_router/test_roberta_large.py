from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import psutil, os

MODEL_ID = "siebert/sentiment-roberta-large-english"

print("Loading tokenizer + model...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
mdl.eval()
proc = psutil.Process(os.getpid())
print("Loaded model. RSS MB:", proc.memory_info().rss / 1024**2)

texts = ["this movie was great!"] * 32
enc = tok(
    texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)
print("Encoded batch. RSS MB:", proc.memory_info().rss / 1024**2)

with torch.no_grad():
    out = mdl(**enc).logits

print("Ran inference. logits.shape =", out.shape)
print("Done.")
