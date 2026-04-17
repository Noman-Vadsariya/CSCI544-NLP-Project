import os
import torch
import torch.nn.functional as F

from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ----------------------------
# setup
# ----------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

DATA_PATH = "./data/raw_datasets/hotpotQA_compact/test/ds.parquet"
MODEL_NAME = "BAAI/bge-large-en-v1.5"

CHUNK_SIZE = 480
CHUNK_OVERLAP = 80

TOP_K = 10
NUM_SAMPLES = 100
BATCH_SIZE = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# load data
# ----------------------------

ds = load_dataset("parquet", data_files=DATA_PATH)["train"]

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]

print("Total QA pairs:", len(queries))

# ----------------------------
# model
# ----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
encoder.eval()

# ----------------------------
# chunking
# ----------------------------

def chunk_text(text):
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + CHUNK_SIZE]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk.strip():
            chunks.append(chunk)

    return chunks


print("Building chunked contexts...")
chunked_contexts = []
for c in contexts:
    chunked_contexts.extend(chunk_text(c))

print("Total chunks:", len(chunked_contexts))

# ----------------------------
# streaming encoding
# ----------------------------

SAVE_DIR = "./colbert_batches"
os.makedirs(SAVE_DIR, exist_ok=True)

def encode_and_save_batches(texts):
    print("Encoding with streaming save...")

    for i in tqdm(range(0, len(texts), 8)):
        batch = texts[i:i+8]

        inputs = tokenizer(
            ["passage: " + t for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480
        ).to(device)

        with torch.inference_mode():
            outputs = encoder(**inputs)

        hidden = F.normalize(outputs.last_hidden_state, dim=-1)
        mask = inputs["attention_mask"]

        batch_embs = []
        for j in range(len(batch)):
            valid_len = int(mask[j].sum())
            emb = hidden[j, 1:valid_len-1].cpu()
            batch_embs.append(emb)

        torch.save(batch_embs, f"{SAVE_DIR}/batch_{i}.pt")

        del inputs, outputs, hidden, batch_embs
        torch.cuda.empty_cache()

# comment out later
encode_and_save_batches(chunked_contexts)

# ----------------------------
# encode query
# ----------------------------

def encode_query(query):
    inputs = tokenizer(
        ["query: " + query],
        return_tensors="pt",
        truncation=True,
        max_length=64
    ).to(device)

    with torch.inference_mode():
        outputs = encoder(**inputs)

    hidden = F.normalize(outputs.last_hidden_state, dim=-1)
    mask = inputs["attention_mask"][0]
    valid_len = int(mask.sum())

    return hidden[0, 1:valid_len-1]

# ----------------------------
# full bert retrieval streamed
# ----------------------------

def retrieve(query):
    q_emb = encode_query(query).to(device)
    scores = []

    for i in range(0, len(chunked_contexts), BATCH_SIZE):
        batch_texts = chunked_contexts[i:i+BATCH_SIZE]

        inputs = tokenizer(
            ["passage: " + t for t in batch_texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480
        ).to(device)

        with torch.inference_mode():
            outputs = encoder(**inputs)

        hidden = F.normalize(outputs.last_hidden_state, dim=-1)
        mask = inputs["attention_mask"]

        for j in range(len(batch_texts)):
            valid_len = int(mask[j].sum())
            p_emb = hidden[j, 1:valid_len-1]

            sim = torch.matmul(q_emb, p_emb.T)
            max_sim = sim.max(dim=1).values
            score = max_sim.sum().item()

            scores.append((i + j, score))

        del inputs, outputs, hidden
        torch.cuda.empty_cache()

    scores.sort(key=lambda x: x[1], reverse=True)

    return [chunked_contexts[i] for i, _ in scores[:TOP_K]]

# ----------------------------
# evaluation
# ----------------------------

num_correct = 0
num_samples = min(NUM_SAMPLES, len(queries))

print("\nRunning full ColBERT eval...")

for i in tqdm(range(num_samples)):
    q = queries[i]
    ans = answers[i]

    retrieved = retrieve(q)

    found = any(
        ans.lower() in ctx.lower()
        or all(w in ctx.lower() for w in ans.lower().split())
        for ctx in retrieved
    )

    if found:
        num_correct += 1

accuracy = num_correct / num_samples
print("\nAccuracy:", round(accuracy, 4))