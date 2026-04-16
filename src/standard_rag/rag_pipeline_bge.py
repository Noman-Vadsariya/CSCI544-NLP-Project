##### BGE encoder version (with chunking)

import os
import torch

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi   

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

path = "/home1/vuvaness/CSCI544-NLP-Project/data/raw_datasets/hotpotQA_compact/test/ds.parquet"
ds = load_dataset("parquet", data_files=path)["train"]

print("num rows:", len(ds))
print("columns:", ds.column_names)

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]

print("Total QA pairs:", len(queries))
print("Total contexts:", len(contexts))


### chunking
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
chunk_id_to_original = []

for i, c in enumerate(contexts):
    chunks = chunk_text(c)
    chunked_contexts.extend(chunks)
    chunk_id_to_original.extend([i] * len(chunks))

print("Total chunked contexts:", len(chunked_contexts))


### bm25 index
tokenized_corpus = [c.lower().split() for c in chunked_contexts]
bm25 = BM25Okapi(tokenized_corpus)


### embeddings
torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

context_inputs = ["passage: " + c for c in chunked_contexts]

context_embeddings = model.encode(
    context_inputs,
    batch_size=32,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
    device=device,
).astype("float32")


### pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag-index-v2"

existing_indexes = [idx.name for idx in pc.list_indexes()]

if index_name not in existing_indexes:
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=context_embeddings.shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

index = pc.Index(index_name)

UPLOAD_EMBEDDINGS = True

if UPLOAD_EMBEDDINGS:
    print("Uploading embeddings to Pinecone...")

    vectors = [
        (
            str(i),
            context_embeddings[i].tolist(),
            {
                "text": chunked_contexts[i],
                "original_id": chunk_id_to_original[i]
            }
        )
        for i in range(len(chunked_contexts))
    ]

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(
            vectors=vectors[i:i + batch_size],
            namespace="default",
        )

    print("Pinecone upload complete!")
else:
    print("Using existing Pinecone index...")


### reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


def retrieve_contexts(query, top_k=5):
    query_embedding = model.encode(
        ["query: " + query + " passage:"],   
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )[0].astype("float32").tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default",
    )

    return [match["metadata"]["text"] for match in results["matches"]]


### bm25 retrieval
def retrieve_bm25(query, top_k=10):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return [chunked_contexts[i] for i in top_indices]


def retrieve_with_rerank(query, top_k=5):
    dense_candidates = retrieve_contexts(query, top_k=30)
    bm25_candidates = retrieve_bm25(query, top_k=30)

    # combine + deduplicate
    candidates = list(set(dense_candidates + bm25_candidates))

    pairs = [[query, ctx] for ctx in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    return [ctx for ctx, _ in ranked[:top_k]]


### eval
num_correct = 0
num_samples = min(200, len(queries))

for i in range(num_samples):
    query = queries[i]
    true_answer = answers[i]

    retrieved_contexts = retrieve_with_rerank(query, top_k=10)

    found_ctx = any(
        true_answer.lower() in ctx.lower()
        or all(word in ctx.lower() for word in true_answer.lower().split())
        for ctx in retrieved_contexts
    )

    if found_ctx:
        num_correct += 1

    print("\n-------------------------------")
    print("Query:", query)
    print("True Answer:", true_answer)
    print("Found in top 5:", found_ctx)

accuracy = num_correct / num_samples if num_samples > 0 else 0.0
print(f"\nRetrieval Accuracy@5: {accuracy:.4f}")