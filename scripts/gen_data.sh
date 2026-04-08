# generate the teacher responses for the hypernetwork training 
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --ds_names hotpotQA_compact hotpotQA_gold_compact asqa_compact --split train --closed_qa_prob 0.1
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --ds_names prontoQA_compact --split train --closed_qa_prob 0.0