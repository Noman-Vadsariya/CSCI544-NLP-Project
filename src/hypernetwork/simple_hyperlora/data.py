"""
Data loading, preprocessing, tokenization, packing, and collation.

Supports multiple datasets (SQuAD, DROP, ROPES, PwC, LongBench, etc.)
with sequence packing for efficient training at batch_size=1.
"""

import logging
import random
from math import isclose
from typing import Any

import numpy as np
import torch
from datasets import Dataset, interleave_datasets, load_dataset
from transformers import PreTrainedTokenizerBase
from transformers.data import DataCollatorWithFlattening, default_data_collator

logger = logging.getLogger()

IGNORE_INDEX = -100

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

DS_KWARGS = {
    "squad": dict(
        train=dict(path="data/raw_datasets/squad", split="train"),
        validation=dict(path="data/raw_datasets/squad", split="validation[:1000]"),
        test=dict(path="data/raw_datasets/squad", split="validation"),
    ),
    "squad_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/squad_compact/train/ds.parquet",
            split="train[180:]",
        ),
    ),
    "drop": dict(
        train=dict(path="ucinlp/drop", split="train"),
        validation=dict(path="ucinlp/drop", split="validation[:900]"),
        test=dict(path="ucinlp/drop", split="validation"),
    ),
    "drop_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/drop_compact/train/ds.parquet",
            split="train",
        ),
    ),
    "ropes": dict(
        train=dict(path="allenai/ropes", split="train"),
        validation=dict(path="allenai/ropes", split="validation[:900]"),
        test=dict(path="allenai/ropes", split="validation"),
    ),
    "ropes_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/ropes_compact/train/ds.parquet",
            split="train",
        ),
    ),
    "pwc": dict(
        train=dict(path="sggetao/PwC", split="train"),
        validation=dict(path="sggetao/PwC", split="test[:900]"),
        test=dict(path="sggetao/PwC", split="test"),
    ),
    "pwc_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/pwc_compact/train/ds.parquet",
            split="train",
        ),
    ),
}

# Add LongBench datasets
for _lb in ["longbench/qasper", "longbench/multifieldqa_en", "longbench/2wikimqa"]:
    DS_KWARGS[_lb] = dict(
        test=dict(path="THUDM/LongBench", name=_lb.split("/")[-1], split="test")
    )

# Add synthetic toy datasets
_tok_bins = [(64, 128), (128, 256), (256, 512)] + [
    (512 + 256 * i, 512 + 256 * (i + 1)) for i in range(14)
]
_tok_bins += [(32, 128), (128, 256), (256, 512), (512, 1024), (32, 1024)] + [
    (1024 * i, 1024 * (i + 1)) for i in range(1, 16)
]
for _toy_name in ["ctx_numbers", "ctx_kv", "ctx_magic_number"]:
    for _bin in _tok_bins:
        _key = f"{_toy_name}_{_bin[0]}_{_bin[1]}"
        DS_KWARGS[_key] = {
            s: dict(path="json", data_files=f"data/raw_datasets/{_key}/{s}.jsonl", split="train")
            for s in ["train", "validation", "test"]
        }

CLOSED_QA_DATASETS = {
    "squad", "drop", "ropes",
    "longbench/qasper", "longbench/multifieldqa_en", "longbench/2wikimqa",
}

CLOSED_QA_INTX_TEMPLATES = [
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "Answer without any explanation.\n\nQuestion: {input}",
    "Based on the provided text, what is the answer to the following question? Provide only the answer.\n\nQuestion: {input}",
    "Extract the answer to the question from the text. Be concise. Do not explain.\n\nQuestion: {input}",
    "What is the answer to this question, based on the context? Respond with the answer only.\n\nQuestion: {input}",
    "Provide a direct answer to the question using the given passages. Do not give any explanation.\n\nQuestion: {input}",
    "Answer the question using only information from the provided text. No extra words.\n\nQuestion: {input}",
    "From the passages, answer the question. Just the answer, please.\n\nQuestion: {input}",
    "Give the answer to the question. Do not include any other text.\n\nQuestion: {input}",
    "The answer to the question is in the text. Find it and state it clearly. No need for explanation.\n\nQuestion: {input}",
]

EVAL_INTX_TEMPLATES = {
    "ropes": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "drop": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "squad": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "longbench/qasper": 'Answer the question as concisely as you can, using a single phrase or sentence if possible.\nIf the question cannot be answered based on the information in the article, write "unanswerable".\nIf the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}',
    "longbench/multifieldqa_en": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
    "longbench/2wikimqa": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
}


# ---------------------------------------------------------------------------
# Preprocessing: raw dataset -> {context, prompts, responses}
# ---------------------------------------------------------------------------

def _closed_qa_prompting(prompt: str) -> str:
    return random.choice(CLOSED_QA_INTX_TEMPLATES).format(input=prompt)


def get_preprocessing_fn(ds_name: str, is_eval: bool):
    """Return a function that maps a raw sample to {context, prompts, responses}."""

    def identity(sample):
        return sample

    if ds_name.endswith("_compact"):
        return identity

    fn = None

    if ds_name == "squad":
        def fn(sample):
            q = sample["question"]
            prompt = _closed_qa_prompting(q) if not is_eval else q
            return {"context": sample["context"], "prompt": prompt, "response": sample["answers"]["text"][0]}

    elif ds_name == "drop":
        def fn(sample):
            q = sample["question"]
            prompt = _closed_qa_prompting(q) if not is_eval else q
            return {"context": sample["passage"], "prompt": prompt, "response": sample["answers_spans"]["spans"][0]}

    elif ds_name == "ropes":
        def fn(sample):
            ctx = f"{sample['background']}\n{sample['situation']}"
            q = sample["question"]
            prompt = _closed_qa_prompting(q) if not is_eval else q
            return {"context": ctx, "prompt": prompt, "response": sample["answers"]["text"][0]}

    elif ds_name in ("pwc", "pwc_tiny"):
        def fn(sample):
            return {"context": sample["input"], "prompt": sample["prompt"], "response": sample["answer"]}

    elif ds_name.startswith("longbench"):
        def fn(sample):
            return {"context": sample["context"], "prompt": sample["input"], "response": sample["answers"][0]}

    else:
        fn = identity

    # Apply eval template if available
    if is_eval and ds_name in EVAL_INTX_TEMPLATES:
        template = EVAL_INTX_TEMPLATES[ds_name]
        orig_fn = fn
        def fn(sample):
            s = orig_fn(sample)
            if "prompt" in s:
                s["prompt"] = template.format(input=s["prompt"])
            return s

    # Normalize to prompts/responses lists
    orig_fn2 = fn
    def fn(sample):
        s = orig_fn2(sample)
        if s is None:
            return {"context": None, "prompts": None, "responses": None}
        if "prompt" in s:
            s["prompts"] = [s.pop("prompt")]
        if "response" in s:
            s["responses"] = [s.pop("response")]
        if "responses" in s and s["responses"]:
            s["responses"] = [r.strip() if isinstance(r, str) else r for r in s["responses"]]
        return s

    return fn


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _concat_list(lst):
    out = []
    for x in lst:
        out += x
    return out


def convert_to_messages(example: dict, add_ctx_to_chat: bool) -> dict:
    """Convert {context, prompts, responses} -> {messages_list} for chat template."""
    messages_list = []
    for prompt, response in zip(example["prompts"], example["responses"]):
        user_msg = prompt.strip()
        if add_ctx_to_chat:
            user_msg = example["context"].strip() + "\n\n" + user_msg
        msgs = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
        messages_list.append(msgs)
    return {"messages_list": messages_list}


def tokenize_messages(samples: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
    """Batch tokenize messages_list -> input_ids + labels."""
    messages_list = samples["messages_list"]
    n_queries = [len(x) for x in messages_list]
    flat_messages = _concat_list(messages_list)

    tokens = tokenizer.apply_chat_template(
        flat_messages,
        tokenize=True,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
    )

    labels = []
    for tok_ids, masks in zip(tokens["input_ids"], tokens["assistant_masks"]):
        labels.append([id_ if mask else IGNORE_INDEX for id_, mask in zip(tok_ids, masks)])

    per_ctx = {"input_ids": [], "labels": []}
    i = 0
    for n in n_queries:
        per_ctx["input_ids"].append(tokens["input_ids"][i : i + n])
        per_ctx["labels"].append(labels[i : i + n])
        i += n
    return per_ctx


def tokenize_ctx_text(samples: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
    """Tokenize context text for the context encoder."""
    if tokenizer.chat_template:
        tokenized = tokenizer.apply_chat_template(
            [
                [{"role": "user", "content": ctx.strip()}] if isinstance(ctx, str) else ctx
                for ctx in samples["context"]
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_attention_mask=False,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_dict=True,
        )
    else:
        raise NotImplementedError("Only chat models supported")
    return {"ctx_ids": tokenized["input_ids"]}


# ---------------------------------------------------------------------------
# Dataset loading + tokenization pipeline
# ---------------------------------------------------------------------------

def get_tokenized_dataset(
    ds_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    ctx_tokenizer: PreTrainedTokenizerBase,
    max_qas_len: int = -1,
    max_qas_per_sample: int = -1,
    add_ctx_to_chat: bool = False,
    use_kl_loss: bool = False,
) -> Dataset:
    """Load, preprocess, and tokenize a dataset."""
    is_eval = split != "train"

    # Load raw dataset
    if ds_name in DS_KWARGS and split in DS_KWARGS[ds_name]:
        kwargs = DS_KWARGS[ds_name][split]
    else:
        kwargs = dict(path=ds_name, split=split)
    ds = load_dataset(**kwargs, trust_remote_code=True)

    # Preprocess
    preprocess_fn = get_preprocessing_fn(ds_name, is_eval)
    ds = ds.map(preprocess_fn, num_proc=16)
    ds = ds.filter(lambda x: all(v is not None for v in x.values()), num_proc=16)

    # Convert to messages
    if "input_ids" not in ds.column_names:
        ds = ds.map(convert_to_messages, fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat}, num_proc=16)
        ds = ds.map(tokenize_messages, fn_kwargs={"tokenizer": tokenizer}, batched=True, batch_size=100_000)

    cols_to_keep = ["input_ids", "labels", "context", "ctx_ids"]
    if use_kl_loss:
        cols_to_keep += ["logprobs_vals", "logprobs_indices"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in cols_to_keep])
    ds = ds.filter(lambda x: bool(x["input_ids"]), num_proc=16)

    # Tokenize context for ctx encoder
    if "ctx_ids" not in ds.column_names:
        ds = ds.map(
            tokenize_ctx_text,
            fn_kwargs={"tokenizer": ctx_tokenizer},
            batched=True,
            batch_size=100_000,
            remove_columns=["context"] if "context" in ds.column_names else [],
        )

    # Single chunk: wrap ctx_ids in list
    if ds[0]["ctx_ids"] and not isinstance(ds[0]["ctx_ids"][0], list):
        ds = ds.map(lambda x: {"ctx_ids": [x["ctx_ids"]]}, num_proc=16)

    # Split long QAs if needed
    if max_qas_len > 0 or max_qas_per_sample > 0:
        ds = ds.map(
            _split_too_long_qas,
            fn_kwargs={"max_qas_len": max_qas_len, "max_qas_per_sample": max_qas_per_sample},
            batched=True,
            batch_size=12_500,
            num_proc=16,
        )

    if is_eval:
        ds = ds.map(_squeeze_tokens, num_proc=4)

    return ds


def _squeeze_tokens(sample):
    for key in ["input_ids", "labels"]:
        if key in sample and sample[key] and isinstance(sample[key][0], list):
            sample[key] = sample[key][0]
    return sample


def _split_too_long_qas(samples, max_qas_len, max_qas_per_sample):
    """Split QA pairs that exceed length limits."""
    input_ids = samples["input_ids"]
    labels = samples["labels"]
    ctx_ids = samples["ctx_ids"]

    total_lengths = [sum(len(x) for x in seq) for seq in input_ids]
    if (max_qas_len < 0 or all(l <= max_qas_len for l in total_lengths)) and \
       (max_qas_per_sample < 0 or all(len(seq) <= max_qas_per_sample for seq in input_ids)):
        return samples

    out = {k: [] for k in samples}
    for i, tot_len in enumerate(total_lengths):
        if (max_qas_len < 0 or tot_len <= max_qas_len) and \
           (max_qas_per_sample < 0 or len(input_ids[i]) <= max_qas_per_sample):
            for k in samples:
                out[k].append(samples[k][i])
            continue

        new_len, new_inp, new_lab = 0, [], []
        for inp, lab in zip(input_ids[i], labels[i]):
            if max_qas_len > 0 and len(inp) > max_qas_len:
                continue
            can_add = (max_qas_len < 0 or new_len + len(inp) <= max_qas_len) and \
                      (max_qas_per_sample < 0 or len(new_inp) < max_qas_per_sample)
            if can_add:
                new_len += len(inp)
                new_inp.append(inp)
                new_lab.append(lab)
            else:
                if new_inp:
                    out["input_ids"].append(new_inp)
                    out["labels"].append(new_lab)
                    out["ctx_ids"].append(ctx_ids[i])
                    for k in samples:
                        if k not in ("input_ids", "labels", "ctx_ids"):
                            out[k].append(samples[k][i])
                new_len, new_inp, new_lab = len(inp), [inp], [lab]
        if new_inp:
            out["input_ids"].append(new_inp)
            out["labels"].append(new_lab)
            out["ctx_ids"].append(ctx_ids[i])
            for k in samples:
                if k not in ("input_ids", "labels", "ctx_ids"):
                    out[k].append(samples[k][i])
    return out


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------

def _check_is_iterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def pack_data_points_by_length(
    lens: list[list[int]],
    ctx_lens: list[list[int]],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_size: int = -1,
) -> list[tuple[int, int]]:
    """Group consecutive samples that fit within length limits."""
    if not lens:
        return []
    len_arr = np.array([sum(l) for l in lens], dtype=np.int64)
    ctx_len_arr = np.array([sum(l) for l in ctx_lens], dtype=np.int64)
    n = len(len_arr)
    cumsum_inp = np.cumsum(len_arr)
    cumsum_ctx = np.cumsum(ctx_len_arr)

    idx_pairs = []
    i = 0
    while i < n:
        start_inp = cumsum_inp[i - 1] if i > 0 else 0
        start_ctx = cumsum_ctx[i - 1] if i > 0 else 0
        valid = ((cumsum_inp[i:] - start_inp) <= max_packed_inp_len) & \
                ((cumsum_ctx[i:] - start_ctx) <= max_packed_ctx_len)
        if not np.any(valid):
            i += 1
            continue
        max_j = i + np.where(valid)[0][-1]
        if max_size > 0:
            max_j = min(max_j, i + max_size - 1)
        idx_pairs.append((i, max_j + 1))
        i = max_j + 1
    return idx_pairs


def pack_data_points_FA(batch: dict) -> dict:
    """Flatten a group of samples into packed tensors with position_ids."""
    total_ctx_len = sum(len(y) for x in batch["ctx_ids"] for y in x)
    total_inp_len = sum(len(y) for x in batch["input_ids"] for y in x)

    ctx_ids = np.empty(total_ctx_len, dtype=np.int64)
    ctx_position_ids = np.empty(total_ctx_len, dtype=np.int64)
    input_ids = np.empty(total_inp_len, dtype=np.int64)
    position_ids = np.empty(total_inp_len, dtype=np.int64)
    labels_arr = np.empty(total_inp_len, dtype=np.int64)

    has_logprobs = "logprobs_vals" in batch
    if has_logprobs:
        n_labels = sum(len(y) for x in batch["logprobs_vals"] for y in x)
        k = len(batch["logprobs_vals"][0][0][0])
        logprobs_vals = np.empty((n_labels, k), dtype=np.float32)
        logprobs_indices = np.empty((n_labels, k), dtype=np.int32)
        logits_offset = 0

    offset = 0
    for idx in range(len(batch["input_ids"])):
        inp_b, lab_b = batch["input_ids"][idx], batch["labels"][idx]
        local = offset
        for ids in inp_b:
            end = local + len(ids)
            position_ids[local:end] = np.arange(len(ids), dtype=np.int32)
            local = end
        flat_inp = _concat_list(inp_b)
        flat_lab = _concat_list(lab_b)
        inp_len = len(flat_inp)
        input_ids[offset : offset + inp_len] = flat_inp
        labels_arr[offset : offset + inp_len] = flat_lab
        offset += inp_len

        if has_logprobs:
            vals = _concat_list(batch["logprobs_vals"][idx])
            inds = _concat_list(batch["logprobs_indices"][idx])
            l = len(vals)
            logprobs_vals[logits_offset : logits_offset + l] = vals
            logprobs_indices[logits_offset : logits_offset + l] = inds
            logits_offset += l

    ctx_offset = 0
    for ctx_b in batch["ctx_ids"]:
        local = ctx_offset
        for chunk in ctx_b:
            end = local + len(chunk)
            ctx_position_ids[local:end] = np.arange(len(chunk), dtype=np.int32)
            local = end
        flat_ctx = _concat_list(ctx_b)
        ctx_ids[ctx_offset : ctx_offset + len(flat_ctx)] = flat_ctx
        ctx_offset += len(flat_ctx)

    out = {
        "ctx_ids": ctx_ids,
        "ctx_position_ids": ctx_position_ids,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels_arr,
    }
    if has_logprobs:
        out["logprobs_vals"] = logprobs_vals
        out["logprobs_indices"] = logprobs_indices
    return out


def pack_batch(batch, max_packed_inp_len, max_packed_ctx_len, max_packed_size=-1, metadata_path=""):
    """Pack a batch of samples into groups that fit within length limits."""
    inp_lens = [[len(y) for y in x] for x in batch["input_ids"]]
    ctx_lens = [[len(y) for y in x] for x in batch["ctx_ids"]]
    n_queries = [len(x) for x in batch["input_ids"]]
    n_ctx_chunks = [len(x) for x in batch["ctx_ids"]]

    idx_pairs = pack_data_points_by_length(inp_lens, ctx_lens, max_packed_inp_len, max_packed_ctx_len, max_packed_size)

    packed = {
        "ctx_ids": [], "ctx_position_ids": [], "input_ids": [],
        "position_ids": [], "labels": [], "n_queries": [], "n_ctx_chunks": [],
    }
    has_logprobs = "logprobs_vals" in batch
    if has_logprobs:
        packed["logprobs_vals"] = []
        packed["logprobs_indices"] = []

    for start, end in idx_pairs:
        group = {
            "ctx_ids": batch["ctx_ids"][start:end],
            "input_ids": batch["input_ids"][start:end],
            "labels": batch["labels"][start:end],
        }
        if has_logprobs:
            group["logprobs_vals"] = batch["logprobs_vals"][start:end]
            group["logprobs_indices"] = batch["logprobs_indices"][start:end]
        item = pack_data_points_FA(group)
        for k in ["ctx_ids", "ctx_position_ids", "input_ids", "position_ids", "labels"]:
            packed[k].append(item[k])
        if has_logprobs:
            packed["logprobs_vals"].append(item["logprobs_vals"])
            packed["logprobs_indices"].append(item["logprobs_indices"])
        packed["n_queries"].append(n_queries[start:end])
        packed["n_ctx_chunks"].append(n_ctx_chunks[start:end])

    return packed


def get_ds_prob(ds_lens: list[int], total: int) -> list[float]:
    """Compute dataset interleaving probabilities (min 1% per dataset)."""
    probs = [0.0] * len(ds_lens)
    for i, l in enumerate(ds_lens):
        if l / total <= 0.01:
            probs[i] = 0.01
    residual = 1 - sum(probs)
    residual_total = sum(l for l in ds_lens if l / total > 0.01)
    for i, l in enumerate(ds_lens):
        if l / total > 0.01:
            probs[i] = l / residual_total * residual
    return probs


def pack_datasets(
    ds_dict: dict[str, Dataset],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_packed_size: int = -1,
    seed: int = 42,
) -> Dataset:
    """Interleave and pack multiple datasets."""
    ds_lens = [len(ds) for ds in ds_dict.values()]
    total = sum(ds_lens)
    logger.info(f"Total samples before packing: {total}")

    train_ds = interleave_datasets(
        list(ds_dict.values()),
        probabilities=get_ds_prob(ds_lens, total),
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    packed_ds = train_ds.map(
        pack_batch,
        fn_kwargs={
            "max_packed_inp_len": max_packed_inp_len,
            "max_packed_ctx_len": max_packed_ctx_len,
            "max_packed_size": max_packed_size,
        },
        batched=True,
        batch_size=125_000,
        num_proc=0,
        remove_columns=train_ds.column_names,
    )
    logger.info(f"Packed dataset: {len(packed_ds)} samples (from {total} original)")
    return packed_ds


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

_flattener = DataCollatorWithFlattening()


def flatten_if_not_packed(inp_list):
    """Collator that handles both packed (train) and unpacked (eval) data."""
    sample = inp_list[0]

    # Packed training data (batch_size=1)
    if "position_ids" in sample:
        assert len(inp_list) == 1, "Use batch_size=1 with packed data"
        n_queries = sample.pop("n_queries")
        n_ctx_chunks = sample.pop("n_ctx_chunks")
        batch = default_data_collator(inp_list, return_tensors="pt")
        batch["n_queries"] = torch.tensor(n_queries)
        batch["n_ctx_chunks"] = torch.tensor(n_ctx_chunks)
        return batch

    # Eval data (not packed)
    n_queries = torch.ones(len(inp_list), dtype=torch.int32)
    n_ctx_chunks = torch.tensor(
        [len(ex["ctx_ids"]) for ex in inp_list], dtype=torch.int32
    )
    packed = _flattener(inp_list, return_tensors="pt")
    packed["n_queries"] = n_queries
    packed["n_ctx_chunks"] = n_ctx_chunks

    if "ctx_ids" in sample:
        ctx_ids = _concat_list([ex.pop("ctx_ids") for ex in inp_list])
        ctx_position_ids = torch.cat([torch.arange(len(ids)) for ids in ctx_ids])
        ctx_ids = torch.tensor(_concat_list(ctx_ids))
        packed["ctx_ids"] = ctx_ids.unsqueeze(0)
        packed["ctx_position_ids"] = ctx_position_ids.unsqueeze(0)

    return packed
