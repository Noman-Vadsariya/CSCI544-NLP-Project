##### BGE-Large + Hybrid Retrieval + NO Decomposition + MiniLM Reranker + MRR

import os
import torch
import random
import time
import gc

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm  

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

path = "/home1/vuvaness/CSCI544-NLP-Project/data/raw_datasets/hotpotQA_compact/test/ds.parquet"
ds = load_dataset("parquet", data_files=path)["train"]

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]

print("Total QA pairs:", len(queries))


### chunking
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

def chunk_text(text, chunk_size=480, overlap=80):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        if chunk.strip():
            chunks.append(chunk)

    return chunks


chunked_contexts = []
for c in contexts:
    chunked_contexts.extend(chunk_text(c))

print("Total chunked contexts:", len(chunked_contexts))

too_long = [c for c in chunked_contexts if len(tokenizer.encode(c)) > 512]
print("Chunks over 512 tokens:", len(too_long))


### BM25
tokenized_corpus = [c.lower().split() for c in chunked_contexts]
bm25 = BM25Okapi(tokenized_corpus)


### embedding model
torch.set_num_threads(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)


### Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-index-v6"

if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


### upload embeddings (already done)
UPLOAD_EMBEDDINGS = True


### reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


### retrieval
def retrieve_contexts(query, top_k=25):
    query_embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default",
    )

    return [m["metadata"]["text"] for m in results["matches"]]


def retrieve_bm25(query, top_k=15):
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunked_contexts[i] for i in top_idx]


### rerank (same function, but only one query)
def retrieve_with_rerank(query, top_k=10):
    dense = retrieve_contexts(query)
    sparse = retrieve_bm25(query)

    candidates = list(set(dense + sparse))
    random.shuffle(candidates)
    candidates = candidates[:50]

    pairs = [[query, c] for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [c for c, _ in ranked[:top_k]]


### metrics
def compute_mrr(ranked, answer):
    for i, ctx in enumerate(ranked):
        if answer.lower() in ctx.lower():
            return 1 / (i + 1)
    return 0


### eval
num_correct = 0
mrr_total = 0
num_samples = len(queries)  # test on all samples

print("\nRunning evaluation...")

for i in tqdm(range(num_samples), desc="Eval Progress"):

    q = queries[i]
    ans = answers[i]

    retrieved = retrieve_with_rerank(q, top_k=10)

    found = any(ans.lower() in ctx.lower() for ctx in retrieved)

    if found:
        num_correct += 1

    mrr_total += compute_mrr(retrieved, ans)

    # time.sleep(0.05)


accuracy = num_correct / num_samples
mrr = mrr_total / num_samples

print(f"\nAccuracy@10: {accuracy:.4f}")
print(f"MRR@10: {mrr:.4f}")