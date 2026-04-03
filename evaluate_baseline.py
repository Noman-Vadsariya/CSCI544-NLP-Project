import time
import torch
import re
import string
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 


from rag_pipeline import retrieve_contexts, queries, answers

rouge = evaluate.load('rouge')

def normalize_answer(s):
    def remv_article(txt): return re.sub(r'\b(a|an|the)\b', ' ', txt)
    def fix_white_space(txt): return ' '.join(txt.split())
    def remv_punc(txt): return ''.join(ch for ch in txt if ch not in set(string.punctuation))
    return fix_white_space(remv_article(remv_punc(s.lower())))

def em_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

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

# Using 1000 samples from dataset to ensure context matches the query
test_queries = queries[:1000]
test_answers = answers[:1000]

for query, true_answer in tqdm(zip(test_queries, test_answers), total=len(test_queries)):
    references.append(true_answer)

    top_chunks = retrieve_contexts(query, top_k=3)
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

print(f"\n--- Baseline RAG Results ---")
print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
print(f"Exact Match: {avg_em:.4f}")
print(f"Avg Latency per query: {sum(latencies)/len(latencies):.4f} seconds")
print(f"Avg Peak Memory: {sum(memory_usages)/len(memory_usages):.2f} MB")
