import os
import time
import torch
import gc
import re
import string
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, CrossEncoder
from datasets import load_dataset
from pinecone import Pinecone
from tqdm import tqdm 

# ================================== 1. RAG SETUP (Independent) ==================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)

# Connect to the HotpotQA Pinecone Index (Requires PINECONE_API_KEY in terminal)
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('rag-index-hotpot') 

def retrieve_contexts(query, top_k=10):
    query_embedding = embedder.encode(['query: ' + query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace='default')
    return [match['metadata']['text'] for match in results['matches']]

def retrieve_with_rerank(query, top_k=3):
    candidates = retrieve_contexts(query, top_k=10)
    if not candidates: return []
    pairs = [[query, ctx] for ctx in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [ctx for ctx, _ in ranked[:top_k]]

# ================================== 2. EVALUATION SETUP ==================================
rouge = evaluate.load('rouge')

ds = load_dataset('parquet', data_files='data/raw_datasets/hotpotQA_compact/test/ds.parquet')['train']
queries = [ex['prompts'][0] for ex in ds]
answers = [ex['responses'][0] for ex in ds]

def normalize_answer(s):
    def remv_article(txt): return re.sub(r'\b(a|an|the)\b', ' ', txt)
    def fix_white_space(txt): return ' '.join(txt.split())
    def remv_punc(txt): return ''.join(ch for ch in txt if ch not in set(string.punctuation))
    return fix_white_space(remv_article(remv_punc(s.lower())))

def em_score(prediction, ground_truth): return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def measure_system_metrics(model, inputs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    begin_time = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)

    if torch.cuda.is_available(): torch.cuda.synchronize()

    latency = time.time() - begin_time
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    return outputs, latency, max_memory_mb

# ================================== 3. THE MARATHON RUN ==================================
models_to_evaluate = {
    "google/gemma-2-2b": "official_gemma_HOTPOT_baseline.txt",
    "Qwen/Qwen3-8B": "official_qwen_HOTPOT_baseline.txt"
}

test_queries = queries
test_answers = answers

for model_name, output_file in models_to_evaluate.items():
    print(f"\n{'='*50}\nLoading {model_name}...\n{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

    predictions, references, latencies, memory_usages = [], [], [], []
    for query, true_answer in tqdm(zip(test_queries, test_answers), total=len(test_queries), desc=f"Evaluating {model_name.split('/')[-1]}"):
        references.append(true_answer)

        top_chunks = retrieve_with_rerank(query, top_k=3)
        context_str = " ".join(top_chunks)

        prompt = f"Context: {context_str}\n\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs, latency, peak_mem = measure_system_metrics(model, inputs)
        latencies.append(latency)
        memory_usages.append(peak_mem)

        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        predictions.append(pred_text)

    rouge_results = rouge.compute(predictions=predictions, references=references)
    em_scores = [em_score(p, r) for p, r in zip(predictions, references)]
    avg_em = sum(em_scores) / len(em_scores)
    f1_scores = [f1_score(p, r) for p, r in zip(predictions, references)]
    avg_f1 = sum(f1_scores) / len(f1_scores)
    avg_latency = sum(latencies)/len(latencies)
    avg_mem = sum(memory_usages)/len(memory_usages)

    results_str = (
        f"--- Baseline RAG Results for {model_name} (HOTPOT QA) ---\n"
        f"ROUGE-L: {rouge_results['rougeL']:.4f}\n"
        f"Exact Match: {avg_em:.4f}\n"
        f"F1: {avg_f1:.4f}\n"
        f"Avg Latency per query: {avg_latency:.4f} seconds\n"
        f"Avg Peak Memory: {avg_mem:.2f} MB\n"
    )
    
    print(results_str)
    
    with open(output_file, "w") as f: 
        f.write(results_str)

    print(f"Wiping {model_name} from GPU memory...\n")
    del model
    del tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

print("All models evaluated on HotpotQA successfully!")
