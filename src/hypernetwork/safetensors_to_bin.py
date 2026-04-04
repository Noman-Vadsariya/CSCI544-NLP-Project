import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "google/gemma-2-2b-it"
out_dir = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/gemma-2-2b-it-bin"

os.makedirs(out_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained(out_dir)
torch.save(model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))
model.config.save_pretrained(out_dir)