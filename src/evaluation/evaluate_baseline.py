import time
import torch
import re
import string
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 

from datasets import load_dataset
from standard_rag.rag_pipeline_bge import retrieve_with_rerank

rouge = evaluate.load('rouge')

ds = load_dataset('parquet', data_files='data/raw_datasets/combined_compact/test/ds.parquet')['train']
queries = [ex['prompts'][0] for ex in ds]
answers = [ex['responses'][0] for ex in ds]

def normalize_answer(s):
    def remv_article(txt): return re.sub(r'\b(a|an|the)\b', ' ', txt)
    def fix_white_space(txt): return ' '.join(txt.split())
    def remv_punc(txt): return ''.join(ch for ch in txt if ch not in set(string.punctuation))
    return fix_white_space(remv_article(remv_punc(s.lower())))

def em_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    if len(common) == 0:
        return 0.0
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
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    latency = time.time() - begin_time
    max_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    return outputs, latency, max_memory_mb

model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

predictions, references, latencies, memory_usages = [], [], [], []

test_queries = queries
test_answers = answers

for query, true_answer in tqdm(zip(test_queries, test_answers), total=len(test_queries)):
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

print(f"\n--- Baseline RAG Results ---")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"Exact Match: {avg_em:.4f}")
print(f"F1: {avg_f1:.4f}")
print(f"Avg Latency per query: {sum(latencies)/len(latencies):.4f} seconds")
print(f"Avg Peak Memory: {sum(memory_usages)/len(memory_usages):.2f} MB")
