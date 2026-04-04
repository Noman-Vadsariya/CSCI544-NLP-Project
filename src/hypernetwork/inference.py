import torch 

from datasets import Dataset
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# load the model 
checkpoint_path = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

# load the dataset
