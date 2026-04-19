sbatch slurm/run_gpu.sbatch src/standard_rag/rag_colbert_reranker.py \
  --pipeline doc2lora \
  --model_path /project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/qwen/stage1_combined_gold_dataset_finetune/pytorch_model.bin \
  --embedding_model /project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/bge-large-en-v1.5 \
  --retrieved_output ./data/retrieved/hotpotQA_colbert_retrieved_d2l_qwen.json