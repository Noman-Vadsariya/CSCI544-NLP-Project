import os
import glob
import faiss

from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#================================== load data ==================================

contexts = []
queries = []
answers = []

dataset_paths = glob.glob('data/raw_datasets/combined_compact/train/ds.parquet')

for path in dataset_paths:
    ds = load_dataset('parquet', data_files=path)['train']

    for ex in ds:
        contexts.append(ex['context'])
        queries.append(ex['prompts'][0])
        answers.append(ex['responses'][0])

print('Dataset loaded sucessfully!')
print('Total QA pairs:', len(queries))
print('Total contexts:', len(set(contexts)))  # set to get unique contexts

# limit dataset size for local testing
contexts = contexts[:20000]
queries = queries[:20000]
answers = answers[:20000]

#================================== embeddings ==================================

import torch    
from sentence_transformers import SentenceTransformer

torch.set_num_threads(1)

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

context_embeddings = model.encode(
    contexts,
    batch_size=32,              # larger batches = faster
    convert_to_numpy=True,
    normalize_embeddings=True,  # improves retrieval quality
    show_progress_bar=True
).astype("float32")

#================================== build vector index - pinecone ==================================

print('Uploading embeddings to Pinecone...')

# in terminal: export PINECONE_API_KEY='your_key_here'
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

index_name = 'rag-index'

# create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=context_embeddings.shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

index = pc.Index(index_name)

vectors = [
    (str(i), context_embeddings[i].tolist(), {"text": contexts[i]})
    for i in range(len(contexts))
]

# upsert in batches (important!)
batch_size = 100

for i in range(0, len(vectors), batch_size):
    index.upsert(
    vectors=vectors[i:i+batch_size],
    namespace='default'
)

print('Pinecone index ready!')

#================================== build vector index - faiss ==================================

# dim = context_embeddings.shape[1]

# index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity
# index.add(context_embeddings)

# print("FAISS index built:", index.ntotal)

#================================== test retrieval ==================================

def retrieve_contexts(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    )[0].tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
)

    return [match["metadata"]["text"] for match in results["matches"]]

for i in range(5):

    query = queries[i]
    true_answer = answers[i]

    retrieved_contexts = retrieve_contexts(query)

    print("\n==============================")
    print("Query:", query)
    print("True Answer:", true_answer)

    for j, ctx in enumerate(retrieved_contexts):
        print(f"\nRetrieved Context {j+1}:")
        print(ctx[:300])