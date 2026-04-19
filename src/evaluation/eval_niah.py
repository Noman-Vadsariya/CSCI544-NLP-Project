import os
import time
import torch
import glob
import random
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from rouge_score import rouge_scorer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Metrics Functions ---
def compute_exact_match(prediction, truth):
    return int(prediction.strip().lower() == truth.strip().lower())

def compute_word_f1(prediction, truth):
    pred_tokens = prediction.strip().lower().split()
    truth_tokens = truth.strip().lower().split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
        
    common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common_tokens.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# Load the Haystack Data (RoPES)
contexts = []
dataset_paths = glob.glob('data/raw_datasets/ropes_compact/validation/ds.parquet')
for path in dataset_paths:
    ds = load_dataset('parquet', data_files=path)['train']
    for ex in ds:
        contexts.append(ex['context'])

print(f"Loaded {len(contexts)} Haystack contexts.")

# Setup the "Needle"
needle_fact = "The secret password to access the mainframe is 'ORION-77'."
needle_question = "What is the secret password to access the mainframe?"
true_answer = "ORION-77"

# Models
models = {
    "Gemma-2-2B": "google/gemma-2-2b",
    "Qwen-3-4B": "Qwen/Qwen3-4B"
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

num_samples = 20 
haystack_size = 5 

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

for model_name, model_id in models.items():
    print(f"\n======================================")
    print(f"Starting Advanced NIAH Eval for: {model_name}")
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.float16,
        token=os.getenv("HF_TOKEN")
    )
    
    # Reset peak memory stats before starting inference
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    exact_matches = 0
    total_word_f1 = 0.0
    total_rouge_l = 0.0
    total_latency = 0.0
    
    for i in tqdm(range(num_samples)):
        # Build Context
        selected_contexts = random.sample(contexts, haystack_size)
        insert_idx = random.randint(0, haystack_size)
        selected_contexts.insert(insert_idx, needle_fact)
        massive_context = "\n\n".join(selected_contexts)
        
        prompt = f"Context:\n{massive_context}\n\nQuestion: {needle_question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # --- Measure Latency ---
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                pad_token_id=tokenizer.eos_token_id
            )
        latency = time.time() - start_time
        total_latency += latency
        
        # Decode
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        # --- Calculate Metrics ---
        exact_matches += compute_exact_match(response, true_answer)
        total_word_f1 += compute_word_f1(response, true_answer)
        total_rouge_l += scorer.score(true_answer, response)['rougeL'].fmeasure
        
    # --- Calculate Averages & Memory ---
    avg_em = (exact_matches / num_samples) * 100
    avg_f1 = (total_word_f1 / num_samples) * 100
    avg_rouge_l = (total_rouge_l / num_samples) * 100
    avg_latency = total_latency / num_samples
    
    peak_memory_gb = 0.0
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    
    # --- Print Final Report ---
    print(f"\n--- Metrics Report for {model_name} ---")
    print(f"Exact Match (EM):      {avg_em:.2f}%")
    print(f"Word-Level F1 Score:   {avg_f1:.2f}%")
    print(f"ROUGE-L Score:         {avg_rouge_l:.2f}%")
    print(f"Avg Latency per Query: {avg_latency:.4f} seconds")
    print(f"Peak GPU Memory:       {peak_memory_gb:.2f} GB")
    
    #cleanup
    del model, tokenizer
    torch.cuda.empty_cache()

print("\nEvaluation Complete!")
