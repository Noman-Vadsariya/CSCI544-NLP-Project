##### BGE-Large + Hybrid Retrieval + Query Decomposition + MiniLM Reranker + MRR

import os
import torch
import random
import time
import gc

from collections import defaultdict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi
from openai import OpenAI
from tqdm import tqdm  

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

'''
load data
'''

path = "/home1/vuvaness/CSCI544-NLP-Project/data/raw_datasets/hotpotQA_compact/test/ds.parquet"
ds = load_dataset("parquet", data_files=path)["train"]

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]

print("Total QA pairs:", len(queries))


'''
chunking
'''

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

def chunk_text(text, chunk_size=480, overlap=80):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
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


'''
BM25
'''

tokenized_corpus = [c.lower().split() for c in chunked_contexts]
bm25 = BM25Okapi(tokenized_corpus)


'''
embedding model
'''

torch.set_num_threads(1)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)


'''
pinecone
'''

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


'''
reranker
'''

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


'''
query decomposition
'''

def decompose_query_llm(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            prompt = f"""
You are helping a retrieval system answer multi-hop questions that require
connecting information from multiple documents.

Break the question into 2-3 specific sub-questions, where each targets
a single fact or entity. Always include the original question.

For example:
Q: "What country is the birthplace of the director of Inception?"
Sub-questions:
- Who directed Inception?
- What country was Christopher Nolan born in?
- What country is the birthplace of the director of Inception?

Question: {query}

Return ONLY the sub-questions, one per line. No numbering or bullets.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            text = response.choices[0].message.content.strip()

            subqueries = [
                line.strip("- ").strip()
                for line in text.split("\n")
                if len(line.strip()) > 3
            ]

            if query in subqueries:
                subqueries.remove(query)

            subqueries = [query] + subqueries

            return subqueries[:3]

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(1)

    raise RuntimeError("LLM decomposition failed")


'''
retrieval
'''

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


def retrieve_bm25(query, top_k=25):
    scores = bm25.get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [chunked_contexts[i] for i in top_idx]


def reciprocal_rank_fusion(result_lists, k=60):
    scores = defaultdict(float)
    for results in result_lists:
        for rank, doc in enumerate(results):
            scores[doc] += 1 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def retrieve_with_rerank(query, subqueries, top_k=10):
    all_result_lists = []

    for i, q_sub in enumerate(subqueries):
        fetch_k = 25 if i == 0 else 15
        all_result_lists.append(retrieve_contexts(q_sub, top_k=fetch_k))
        all_result_lists.append(retrieve_bm25(q_sub, top_k=fetch_k))

    candidates = reciprocal_rank_fusion(all_result_lists)[:60]

    best_scores = [-999.0] * len(candidates)
    for q_sub in subqueries:
        pairs = [[q_sub, c] for c in candidates]
        scores = reranker.predict(pairs)
        for j, s in enumerate(scores):
            if s > best_scores[j]:
                best_scores[j] = s

    ranked = sorted(zip(candidates, best_scores), key=lambda x: x[1], reverse=True)

    return [c for c, _ in ranked[:top_k]]


'''
metrics
'''

def compute_mrr(ranked, answer):
    for i, ctx in enumerate(ranked):
        if answer.lower() in ctx.lower():
            return 1 / (i + 1)
    return 0


'''
eval
'''

num_correct = 0
mrr_total = 0
num_samples = 500

print("\nRunning evaluation...")

for i in tqdm(range(num_samples), desc="Eval Progress"):
    q = queries[i]
    ans = answers[i]

    subqueries = decompose_query_llm(q)
    retrieved = retrieve_with_rerank(q, subqueries, top_k=10)

    found = any(ans.lower() in ctx.lower() for ctx in retrieved)

    if found:
        num_correct += 1

    mrr_total += compute_mrr(retrieved, ans)

accuracy = num_correct / num_samples
mrr = mrr_total / num_samples

print(f"\nAccuracy: {accuracy:.4f}")
print(f"MRR: {mrr:.4f}")