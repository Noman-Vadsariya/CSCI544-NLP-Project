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
NUM_SAMPLES = 100  # start small first!!

BATCH_SIZE = 64  # batched scoring

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

chunked_contexts = []
for c in contexts:
    chunked_contexts.extend(chunk_text(c))

print("Total chunks:", len(chunked_contexts))

# ----------------------------
# encode all passages once
# ----------------------------

def encode_passages(texts):
    all_embeddings = []

    print("Encoding passages...")

    for i in tqdm(range(0, len(texts), 8)):
        batch = texts[i:i+8]

        inputs = tokenizer(
            ["passage: " + t for t in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480
        ).to(device)

        outputs = encoder(**inputs)
        hidden = F.normalize(outputs.last_hidden_state, dim=-1)

        mask = inputs["attention_mask"]

        for j in range(len(batch)):
            valid_len = int(mask[j].sum())
            emb = hidden[j, 1:valid_len-1].detach().cpu()
            all_embeddings.append(emb)

    return all_embeddings

passage_embeddings = encode_passages(chunked_contexts)

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

    outputs = encoder(**inputs)
    hidden = F.normalize(outputs.last_hidden_state, dim=-1)

    mask = inputs["attention_mask"][0]
    valid_len = int(mask.sum())

    return hidden[0, 1:valid_len-1]

# ----------------------------
# batched colbert retrieval
# ----------------------------

def retrieve(query):
    q_emb = encode_query(query).to(device)

    scores = []

    for i in range(0, len(passage_embeddings), BATCH_SIZE):
        batch = passage_embeddings[i:i+BATCH_SIZE]

        # pad to same length
        max_len = max(p.shape[0] for p in batch)

        padded = []
        for p in batch:
            pad_len = max_len - p.shape[0]
            if pad_len > 0:
                pad = torch.zeros(pad_len, p.shape[1])
                p = torch.cat([p, pad], dim=0)
            padded.append(p)

        p_batch = torch.stack(padded).to(device)  # [B, p_len, dim]

        # compute similarity
        sim = torch.matmul(q_emb.unsqueeze(0), p_batch.transpose(1, 2))
        sim = sim.squeeze(0)  # [B, q_len, p_len]

        # max over passage tokens
        max_sim = sim.max(dim=2).values  # [B, q_len]

        # sum over query tokens
        batch_scores = max_sim.sum(dim=1)  # [B]

        for j, s in enumerate(batch_scores):
            scores.append((i + j, s.item()))

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