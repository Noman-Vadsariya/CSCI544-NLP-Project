import numpy as np
import faiss
import glob

from sentence_transformers import SentenceTransformer
from datasets import load_dataset

#================================== load data ==================================

contexts = []
queries = []
answers = []

# load all compact datasets
dataset_paths = glob.glob('data/raw_datasets/*_compact/*/ds.parquet')

for path in dataset_paths:

    ds = load_dataset('parquet', data_files=path)['train']

    for ex in ds:
        ctx = ex['context']

        for q, a in zip(ex['prompts'], ex['responses']):
            contexts.append(ctx)
            queries.append(q)
            answers.append(a)

print('Dataset loaded sucessfully!')
print('Total QA pairs:', len(queries))
print('Total contexts:', len(set(contexts)))  # set to get unique contexts

# limit dataset size for local testing
contexts = contexts[:20000]
queries = queries[:20000]
answers = answers[:20000]

#================================== embeddings ==================================


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer('all-MiniLM-L6-v2')

context_embeddings = model.encode(
    contexts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    device='cpu'
).astype('float32')

#================================== build vector index - faiss ==================================

dim = context_embeddings.shape[1]

# build the faiss index
index = faiss.IndexFlatL2(dim)
index.add(context_embeddings)


#================================== test retrieval ==================================

# function to retrieve top-k relevant contexts for a given query
def retrieve_contexts(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_contexts = [contexts[i] for i in indices[0]]
    return retrieved_contexts

# test for example queries
for i in range(5):
    query = queries[i]
    true_answer = answers[i]

    retrieved_contexts = retrieve_contexts(query, top_k=5)

    print('\n==============================')
    print('Query:', query)
    print('True Answer:', true_answer)

    for j, ctx in enumerate(retrieved_contexts):
        print(f'\nRetrieved Context {j+1}:')
        print(ctx[:300])  # print only first 300 chars