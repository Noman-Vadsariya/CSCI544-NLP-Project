import torch 

from datasets import Dataset
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# load the model 
# checkpoint_path = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
# checkpoint_path = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_hotpot_scratch/pytorch_model.bin"
checkpoint_path = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_hotpot/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

# initialize inputs
input = {
  "question": "What government position was held by the author of The Hobbit?",
  "context": [
    ["The Hobbit", [
      "The Hobbit is a fantasy novel by J. R. R. Tolkien.",
      "It was published in 1937."
    ]],
    ["J. R. R. Tolkien", [
      "J. R. R. Tolkien was an English writer and academic.",
      "He served as Rawlinson and Bosworth Professor of Anglo-Saxon at Oxford."
    ]]
  ],
  "answer": "Rawlinson and Bosworth Professor of Anglo-Saxon"
}

# tokenize
chat_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": input["question"]}],
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
answer_ids = tokenizer(input['answer'])

# generate and apply the LoRAs using the context
for context in input["context"]:
    title, sentences = context
    context_str = f"{title}: {' '.join(sentences)}"
    
    model.internalize(context_str)

print(chat_ids.keys())
outputs = model.generate(input_ids=chat_ids['input_ids'], max_new_tokens=512)
print(tokenizer.decode(outputs[0]))
