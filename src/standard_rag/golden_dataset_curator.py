##### BGE encoder version

import os
import glob
import re

# import faiss
import torch

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder
from datasets import Dataset
import gc
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================================== load data ==================================

contexts = []
queries = []
answers = []

dataset_paths = glob.glob("data/raw_datasets/combined_noisy_dataset/train/ds.parquet")

for path in dataset_paths:
    ds = load_dataset("parquet", data_files=path)["train"]

    for ex in ds:
        contexts.append(ex["context"])
        queries.append(ex["prompts"][0])
        answers.append(ex["responses"][0])

print("Total QA pairs:", len(queries))
print("Total contexts:", len(set(contexts)))  # set to get unique contexts

# limit dataset size for local testing
contexts = contexts
queries = queries
answers = answers

# ================================== embeddings ==================================

# limit pytorch to 1 thread to avoid oversubscription
torch.set_num_threads(1)

### set up sentence transformer model for embedding contexts - use BGE instead
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

context_inputs = ["passage: " + c for c in contexts]

context_embeddings = model.encode(
    context_inputs,
    batch_size=32,  # larger batches = faster
    convert_to_numpy=True,
    normalize_embeddings=True,  # improves retrieval quality
    show_progress_bar=True,
).astype("float32")

# ================================== build vector index - pinecone ==================================

print("Uploading embeddings to Pinecone...")

# in terminal: export PINECONE_API_KEY='your_key_here'
pc = Pinecone(
    api_key=os.getenv('PINECONE_API_KEY')
)

index_name = "rag-index"

# create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=context_embeddings.shape[1],
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)  # connect to index

MAX_CONTEXT_CHARS = 20000  # adjust as needed to stay under Pinecone's limit

def safe_context(ctx):
    return ctx[:MAX_CONTEXT_CHARS]

# prepare data for upsert
vectors = [
    (str(i), context_embeddings[i].tolist(), {"text": safe_context(contexts[i])})
    for i in range(len(contexts))
]

# upsert in batches (important!)
batch_size = 100

### upsert all vectors with metadata in batches
for i in range(0, len(vectors), batch_size):
    index.upsert(vectors=vectors[i : i + batch_size], namespace="default")

print("Pinecone index ready!")

# ================================== build vector index - faiss ==================================

# dim = context_embeddings.shape[1]

# index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity
# index.add(context_embeddings)

# print("FAISS index built:", index.ntotal)


# ================================== bert reranker ==================================

### initialize bert cross encoder for reranking retrieved contexts
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


### function to rerank retrieved contexts based on relevance to wuery
def retrieve_with_rerank(query, top_k=5):

    # step 1: get more candidates from Pinecone
    candidates = retrieve_contexts(query, top_k=10)

    # step 2: prepare query-context pairs
    pairs = [[query, ctx] for ctx in candidates]

    # step 3: get BERT relevance scores
    scores = reranker.predict(pairs)

    # step 4: sort by score (descending)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # step 5: return top-k reranked results
    return [ctx for ctx, _ in ranked[:top_k]]


# ================================== test retrieval ==================================


### function to retrieve contexts from piecone based on query embedding
def retrieve_contexts(query, top_k=5):
    # embed query using same modell + settings as contexts
    query_embedding = model.encode(
        ["query: " + query], normalize_embeddings=True, convert_to_numpy=True
    )[0].tolist()

    # query pinecone idx for top k similar contexts
    results = index.query(
        vector=query_embedding, top_k=top_k, include_metadata=True, namespace="default"
    )
    # extract + return text of referenced contexts
    return [match["metadata"]["text"] for match in results["matches"]]


### evaluate retrieval accuracy on sample of queries - check if true answer appears in retrieved contxts
# num_correct = 0
num_samples = len(queries)  # start small for testing

# for i in range(num_samples):
#     query = queries[i]
#     true_answer = answers[i]

#     retrieved_contexts = retrieve_with_rerank(
#         query
#     )  # retrieve from pinecone + rerank with BERT

#     # check if answer appears in ANY retrieved chunk
#     found_ctx = any(true_answer.lower() in ctx.lower() for ctx in retrieved_contexts)

#     # count as correct if answer found in any retrieved context
#     if found_ctx:
#         num_correct += 1

#     print("\n-------------------------------")
#     print("Query:", query)
#     print("True Answer:", true_answer)
#     print("Found in top 5:", found_ctx)

# # compute retrieval accuracy
# accuracy = num_correct / num_samples
# print(f"\nRetrieval Accuracy@5: {accuracy:.4f}")


# ===================== Golden Dataset Curation =====================

def curate_golden_dataset_streaming(
    queries, answers, retrieve_with_rerank, output_path, batch_size=10
):
    """
    For each query, retrieve top 5 contexts, and save as triplets.
    Appends to parquet file every `batch_size` queries to avoid memory issues.
    Output: Parquet file with rows {context: str, prompts: [str], responses: [str]}
    """
    ctx_qa_dict = dict()
    num_samples = len(queries)
    schema = pa.schema(
        [
            ("context", pa.string()),
            ("prompts", pa.list_(pa.string())),
            ("responses", pa.list_(pa.string())),
        ]
    )
    writer = None
    for i in range(num_samples):
        query = queries[i]
        answer = answers[i]
        retrieved_contexts = retrieve_with_rerank(query, top_k=5)
        ctx = "\n".join(retrieved_contexts)  # combine retrieved contexts into one string
        ctx = safe_context(ctx)  # ensure context length is within Pinecone's limit
        if ctx not in ctx_qa_dict:
            ctx_qa_dict[ctx] = {"prompts": [], "responses": []}
        ctx_qa_dict[ctx]["prompts"].append(query)
        ctx_qa_dict[ctx]["responses"].append(answer)
        # Write batch to parquet every batch_size queries
        if (i + 1) % batch_size == 0 or (i + 1) == num_samples:
            samples = [
                {
                    "context": ctx,
                    "prompts": ctx_qa_dict[ctx]["prompts"],
                    "responses": ctx_qa_dict[ctx]["responses"],
                }
                for ctx in ctx_qa_dict
            ]
            table = pa.Table.from_pylist(samples, schema=schema)
            if writer is None:
                writer = pq.ParquetWriter(output_path, schema)
            writer.write_table(table)
            print(
                f"Appended {len(samples)} rows to {output_path} (up to query {i + 1})"
            )
            ctx_qa_dict.clear()
    if writer is not None:
        writer.close()
    print(f"Golden dataset saved to {output_path} (streaming mode)")


curate_golden_dataset_streaming(queries, answers, retrieve_with_rerank, output_path="data/raw_datasets/golden_rag_compact/train/ds.parquet", batch_size=10)
