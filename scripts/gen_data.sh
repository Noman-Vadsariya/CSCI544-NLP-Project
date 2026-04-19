# generate the teacher responses for the hypernetwork training 
sbatch slurm/run_gpu.sbatch data/self_generate_qa.py \
  --vllm_model Qwen/Qwen3-4B-Instruct-2507 \
  --ds_names gen_hotpotQA_compact gen_hotpotQA_gold_compact gen_asqa_compact \
  --split train \
  --closed_qa_prob 1.0

sbatch slurm/run_gpu.sbatch data/self_generate_qa.py \
  --vllm_model Qwen/Qwen3-4B-Instruct-2507 \
  --ds_names gen_prontoQA_compact \
  --split train \
  --closed_qa_prob 0.0 \


sbatch slurm/run_gpu.sbatch data/self_generate_qa.py \
  --vllm_model Qwen/Qwen3-4B-Instruct-2507 \
  --ds_names gen_combined_gold_dataset \
  --split train \
  --closed_qa_prob 1.0 \
  --do_truncate
