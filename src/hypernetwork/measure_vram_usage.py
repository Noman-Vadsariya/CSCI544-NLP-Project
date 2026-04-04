"""Test VRAM usage for different context sizes.

Usage:
    uv run python3 src/finetune_hypernetwork_test.py --tokens 1024
    uv run python3 src/finetune_hypernetwork_test.py --tokens 2048 --chunk_len 512
"""
import argparse
import gc
from math import ceil

import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.data.definitions import CTX_AFFIXES

CHECKPOINT_PATH = "/project2/robinjia_875/lijc/CSCI544-NLP-Project/checkpoints/trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin"

PARAGRAPH = (
    "Machine learning is a subset of artificial intelligence that focuses on "
    "building systems that learn from data. Unlike traditional programming where "
    "rules are explicitly coded, ML algorithms identify patterns in data and make "
    "decisions with minimal human intervention. "
)


def load_model():
    state_dict = torch.load(CHECKPOINT_PATH, weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False
    )
    model.reset()
    model.eval()
    model.to("cuda")
    return model


def build_ctx_ids(n_tokens, ctx_tokenizer):
    """Build tokenized context of approximately n_tokens."""
    repeated = (PARAGRAPH + " ") * (n_tokens // 15 + 1)
    ids = ctx_tokenizer.encode(repeated, add_special_tokens=False)[:n_tokens]
    return ids


def chunk_ctx_ids(ctx_ids, chunk_len, model_name):
    """Split ctx_ids into chunks, applying affixes from CTX_AFFIXES."""
    if chunk_len <= 0 or len(ctx_ids) <= chunk_len:
        return [ctx_ids]

    n_chunks = max(1, ceil(len(ctx_ids) / chunk_len))
    avg_len = ceil(len(ctx_ids) / n_chunks)
    chunks = [ctx_ids[i : i + avg_len] for i in range(0, len(ctx_ids), avg_len)]

    if model_name in CTX_AFFIXES:
        prefix = CTX_AFFIXES[model_name]["prefix"]
        suffix = CTX_AFFIXES[model_name]["suffix"]
        chunks[0] = chunks[0] + suffix
        for i in range(1, len(chunks) - 1):
            chunks[i] = prefix + chunks[i] + suffix
        if len(chunks) > 1:
            chunks[-1] = prefix + chunks[-1]

    return chunks


def try_forward(model, ctx_ids_list, tokenizer, ctx_tokenizer):
    """Run model.forward() with chunked ctx and a dummy QA input. Returns peak VRAM in GB."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    device = "cuda"
    pad_id = ctx_tokenizer.pad_token_id or 0

    # Pad chunks to same length and stack
    max_len = max(len(c) for c in ctx_ids_list)
    padded = [c + [pad_id] * (max_len - len(c)) for c in ctx_ids_list]
    ctx_ids = torch.tensor(padded, device=device)           # [n_chunks, ctx_len]
    ctx_attn_mask = (ctx_ids != pad_id).long()
    n_ctx_chunks = torch.tensor([len(ctx_ids_list)], device=device)

    # Dummy QA input (like trainer does)
    prompt = "What is machine learning?"
    response = "ML is a subset of AI."
    chat = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    encoded = tokenizer.apply_chat_template(chat, tokenize=True, return_tensors="pt", return_dict=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    n_queries = torch.ones(1, dtype=torch.int32, device=device)

    model.reset()
    model.patch_lora_forward()

    with torch.no_grad():
        outputs, (gen_loras, _) = model(
            ctx_ids=ctx_ids,
            ctx_attn_mask=ctx_attn_mask,
            n_ctx_chunks=n_ctx_chunks,
            n_queries=n_queries,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_generated_lora=True,
        )

    peak = torch.cuda.max_memory_allocated() / 1e9
    model.reset()
    return peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512, help="Number of context tokens")
    parser.add_argument("--chunk_len", type=int, default=0, help="Chunk length (0 = no chunking)")
    args = parser.parse_args()

    print("Loading model...")
    model = load_model()
    tokenizer = get_tokenizer(model.base_model.name_or_path)
    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    ctx_encoder_name = model.ctx_encoder.base_model.name_or_path

    ctx_max_pos = model.ctx_encoder.base_model.config.max_position_embeddings
    print(f"Base model: {model.base_model.name_or_path}")
    print(f"Ctx encoder: {ctx_encoder_name} (max_pos={ctx_max_pos})")
    print(f"GPU: {torch.cuda.get_device_name()}, "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build and chunk context
    raw_ids = build_ctx_ids(args.tokens, ctx_tokenizer)
    chunks = chunk_ctx_ids(raw_ids, args.chunk_len, ctx_encoder_name)

    n_chunks = len(chunks)
    chunk_lens = [len(c) for c in chunks]
    print(f"\nContext: {len(raw_ids)} tokens -> {n_chunks} chunk(s), lengths: {chunk_lens}")

    try:
        peak = try_forward(model, chunks, tokenizer, ctx_tokenizer)
        print(f"OK  |  peak VRAM: {peak:.2f} GB")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("OOM!")
        else:
            raise


if __name__ == "__main__":
    main()
