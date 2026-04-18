import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hypernetwork.ctx_to_lora.model_loading import get_tokenizer
from src.hypernetwork.ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"
# HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_hotpotqa_gold_finetune/pytorch_model.bin"
# HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_combined_noisy_dataset_finetune/pytorch_model.bin"
HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_hotpotQA_gold_compact_finetune/pytorch_model.bin"
# HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage1_hotpotQA_gold_comapct_scratch/pytorch_model.bin"
HYPERNET_CHECKPOINT = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/train_outputs/stage2_hotpotQA_gold_compact_finetune/pytorch_model.bin"
BASELINE_MODEL_PATH = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/gemma-2-2b-it-bin"

EXAMPLE_INPUT = {
    "context": "\n".join([
        "The Hobbit is a fantasy novel by J. R. R. Tolkien.",
        "It was published in 1937.",
        "J. R. R. Tolkien was an English writer and academic.",
        "He served as Rawlinson and Bosworth Professor of Anglo-Saxon at Oxford.",
    ]),
    "prompts": ["What government position was held by the author of The Hobbit?"],
    "responses": ["Rawlinson and Bosworth Professor of Anglo-Saxon"],
}

# EXAMPLE_INPUT = {
#     "prompts": "What does Doc-to-LoRA do and why is it useful for long input sequences?",
#     "context": [
#         ["Long input sequences in LLMs", [
#             "Long input sequences are central to in-context learning, document understanding, and multi-step reasoning of Large Language Models (LLMs).",
#             "However, the quadratic attention cost of Transformers makes inference memory-intensive and slow.",
#         ]],
#         ["Context distillation and Doc-to-LoRA", [
#             "While context distillation (CD) can transfer information into model parameters, per-prompt distillation is impractical due to training costs and latency.",
#             "Doc-to-LoRA (D2L) is a lightweight hypernetwork that meta-learns to perform approximate context distillation within a single forward pass.",
#             "Given an unseen prompt, D2L generates a LoRA adapter for a target LLM, enabling subsequent queries to be answered without re-consuming the original context.",
#             "This reduces latency and KV-cache memory consumption during inference of the target LLM.",
#         ]],
#         ["Performance and applications", [
#             "On a long-context needle-in-a-haystack task, D2L learns to map contexts into adapters that store the needle information.",
#             "It achieves near-perfect zero-shot accuracy at sequence lengths exceeding the target LLM’s native context window by more than 4×.",
#             "On real-world QA datasets with limited compute, D2L outperforms standard context distillation while significantly reducing peak memory consumption and update latency.",
#             "D2L may enable rapid adaptation of LLMs, including frequent knowledge updates and personalized chat behavior.",
#         ]],
#     ],
#     "responses": "Doc-to-LoRA is a lightweight hypernetwork that converts an unseen context into a LoRA adapter in a single forward pass, allowing a target LLM to answer later queries without rereading the original context, thereby reducing latency and KV-cache memory use."
# }

EXAMPLE_INPUT = {
    "context": "GRPO is a reinforcement learning method that updates a model by comparing groups of sampled responses using relative rewards.",
    "prompts": ["What is GRPO?"],
    "responses": ["A reinforcement learning method that updates a model using relative rewards across groups of sampled responses."],
}

EXAMPLE_INPUT = {
    "context": "\n".join([
        "GRPO stands for Group Relative Policy Optimization.",
        "It is a reinforcement learning method used to improve language models.",
        "Instead of judging each sampled response in isolation, GRPO compares a group of sampled responses to the same prompt.",
        "The rewards are normalized relative to the other samples in the group, which helps the model learn which responses are better within that set.",
        "This relative comparison can reduce the need for a separate learned value function.",
        "GRPO is commonly discussed in the context of training reasoning models and reward-based post-training.",
    ]),
    "prompts": ["What is GRPO and how does it work?"],
    "responses": ["GRPO is a reinforcement learning method for language models that improves behavior by comparing groups of sampled responses to the same prompt and updating the model using rewards that are normalized relative to the group."],
}


def load_hypernet():
    state_dict = torch.load(HYPERNET_CHECKPOINT, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    tokenizer = get_tokenizer(model.base_model.name_or_path)
    return model, tokenizer


def load_baseline():
    tokenizer = AutoTokenizer.from_pretrained(
        BASELINE_MODEL_PATH, extra_special_tokens={}
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASELINE_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def run_hypernet(model, tokenizer, example, max_new_tokens=512):
    model.reset()
    model.internalize(example["context"])

    decoded = []
    for prompt in example["prompts"]:
        chat_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        outputs = model.generate(input_ids=chat_ids, max_new_tokens=max_new_tokens)
        decoded.append(tokenizer.decode(outputs[0], skip_special_tokens=False))
    return decoded


def run_baseline(model, tokenizer, example, max_new_tokens=512):
    decoded = []
    for prompt in example["prompts"]:
        user_content = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{example['context']}\n\n"
            f"Question: {prompt}"
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
        decoded.append(tokenizer.decode(outputs[0], skip_special_tokens=False))
    return decoded


def main():
    example = EXAMPLE_INPUT
    device = torch.cuda.current_device()

    print("=" * 80)
    print("QUESTIONS:", example["prompts"])
    print("GOLD ANSWERS:", example["responses"])
    print("=" * 80)

    hypernet_model, hypernet_tokenizer = load_hypernet()
    baseline_model, baseline_tokenizer = load_baseline()

    print("\n[Hypernetwork model]")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_time = time.time()
    hypernet_output = run_hypernet(hypernet_model, hypernet_tokenizer, example)
    torch.cuda.synchronize(device)
    end_time = time.time()

    peak_vram_hypernet = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(hypernet_output)
    print("Time taken:", end_time - start_time)
    print(f"Peak VRAM: {peak_vram_hypernet:.3f} GB")

    print("\n[Baseline Gemma-2-2b-it]")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_time = time.time()
    baseline_output = run_baseline(baseline_model, baseline_tokenizer, example)
    torch.cuda.synchronize(device)
    end_time = time.time()

    peak_vram_baseline = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(baseline_output)
    print("Time taken:", end_time - start_time)
    print(f"Peak VRAM: {peak_vram_baseline:.3f} GB")


if __name__ == "__main__":
    main()