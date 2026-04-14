import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
# HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_hotpotqa_gold_finetune/pytorch_model.bin"
BASELINE_MODEL_PATH = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/gemma-2-2b-it-bin"

EXAMPLE_INPUT = {
    "question": "What government position was held by the author of The Hobbit?",
    "context": [
        ["The Hobbit", [
            "The Hobbit is a fantasy novel by J. R. R. Tolkien.",
            "It was published in 1937.",
        ]],
        ["J. R. R. Tolkien", [
            "J. R. R. Tolkien was an English writer and academic.",
            "He served as Rawlinson and Bosworth Professor of Anglo-Saxon at Oxford.",
        ]],
    ],
    "answer": "Rawlinson and Bosworth Professor of Anglo-Saxon",
}


def format_contexts(contexts):
    return [f"{title}: {' '.join(sentences)}" for title, sentences in contexts]


def run_hypernet(example, max_new_tokens=512):
    state_dict = torch.load(HYPERNET_CHECKPOINT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.reset()
    tokenizer = get_tokenizer(model.base_model.name_or_path)

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}],
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    for context_str in format_contexts(example["context"]):
        model.internalize(context_str)

    outputs = model.generate(input_ids=chat_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def run_baseline(example, max_new_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained(
        BASELINE_MODEL_PATH, extra_special_tokens={}
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASELINE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    context_block = "\n".join(format_contexts(example["context"]))
    user_content = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {example['question']}"
    )

    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(input_ids=chat_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=False)


def main():
    example = EXAMPLE_INPUT

    print("=" * 80)
    print("QUESTION:", example["question"])
    print("GOLD ANSWER:", example["answer"])
    print("=" * 80)

    print("\n[Hypernetwork model]")
    start_time = time.time()
    print(run_hypernet(example))
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

    print("\n[Baseline Gemma-2-2b-it]")
    start_time = time.time()
    print(run_baseline(example))
    end_time = time.time()
    print("Time taken: ", end_time - start_time)


if __name__ == "__main__":
    main()
