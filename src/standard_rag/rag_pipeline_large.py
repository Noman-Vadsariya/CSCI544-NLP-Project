import os
import glob
import torch
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer, CrossEncoder

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH_GLOB = "data/raw_datasets/hotpotQA_compact/test/ds.parquet"
INDEX_NAME = "rag-index-bge-large"
NAMESPACE = "default"

# Retrieval tuning:
CANDIDATE_K = 100   # retrieve more candidates before reranking
FINAL_K = 5         # set to 2 for top-2 evaluation
BATCH_SIZE = 32

# =============================================================================
# LOAD DATA
# =============================================================================
contexts = []
queries = []
answers = []

dataset_paths = glob.glob(DATA_PATH_GLOB)

for path in dataset_paths:
    ds = load_dataset("parquet", data_files=path)["train"]
    for ex in ds:
        contexts.append(ex["context"])
        queries.append(ex["prompts"][0])
        answers.append(ex["responses"][0])

print("Total QA pairs:", len(queries))
print("Total contexts:", len(set(contexts)))

# Limit dataset size for local testing
contexts = contexts[:20000]
queries = queries[:20000]
answers = answers[:20000]

# =============================================================================
# EMBEDDINGS
# =============================================================================
torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"

# CHANGED: use the larger embedder
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)

# CHANGED: BGE v1.5 uses query instruction for queries
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# CHANGED: passages are embedded as-is (no passage prefix)
context_embeddings = embedder.encode(
    contexts,
    batch_size=BATCH_SIZE,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype("float32")

# =============================================================================
# PINECONE INDEX
# =============================================================================
print("Uploading embeddings to Pinecone...")

# CHANGED: read key from environment instead of hardcoding it
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=context_embeddings.shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

vectors = [
    (str(i), context_embeddings[i].tolist(), {"text": contexts[i]})
    for i in range(len(contexts))
]

upsert_batch_size = 100

for i in range(0, len(vectors), upsert_batch_size):
    index.upsert(
        vectors=vectors[i:i + upsert_batch_size],
        namespace=NAMESPACE
    )

print("Pinecone index ready!")

# =============================================================================
# RERANKER
# =============================================================================
# CHANGED: stronger reranker
reranker = CrossEncoder("BAAI/bge-reranker-large", device=device)

def embed_query(query: str):
    return embedder.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].tolist()

def retrieve_contexts(query: str, top_k: int = CANDIDATE_K):
    query_embedding = embed_query(query)

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE
    )

    return [match["metadata"]["text"] for match in results["matches"]]

def retrieve_with_rerank(query: str, final_k: int = FINAL_K, candidate_k: int = CANDIDATE_K):
    # CHANGED: retrieve many more candidates before reranking
    candidates = retrieve_contexts(query, top_k=candidate_k)

    if not candidates:
        return []

    pairs = [(query, ctx) for ctx in candidates]

    # CHANGED: rerank with a stronger cross-encoder
    scores = reranker.predict(pairs, batch_size=16)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [ctx for ctx, _ in ranked[:final_k]]

# =============================================================================
# EVALUATION
# =============================================================================
# NOTE: this still checks whether the answer string appears in the retrieved text.
# For true retrieval quality, use gold supporting passages if your dataset has them.
num_correct = 0
num_samples = 200

for i in range(num_samples):
    query = queries[i]
    true_answer = answers[i]

    retrieved_contexts = retrieve_with_rerank(query, final_k=FINAL_K, candidate_k=CANDIDATE_K)

    found_ctx = any(true_answer.lower() in ctx.lower() for ctx in retrieved_contexts)

    if found_ctx:
        num_correct += 1

    print("\n-------------------------------")
    print("Query:", query)
    print("True Answer:", true_answer)
    print(f"Found in top {FINAL_K}:", found_ctx)

accuracy = num_correct / num_samples
print(f"\nRetrieval Accuracy@{FINAL_K}: {accuracy:.4f}")