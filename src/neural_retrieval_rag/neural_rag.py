#!/usr/bin/env python3
"""
SPLADE passage ranker over a GLOBAL corpus.

Input format:
    parquet / HF dataset rows with columns:
      - context: string document text (the gold supporting doc for that query)
      - prompts: [question]
      - responses: [answer]

Pipeline:
1) Split every example's context into chunks and combine all chunks into one global corpus.
2) SPLADE-encode every chunk once.
3) SPLADE-encode the query, and retrieve a ranked list of similar chunks from the corpus.
4) Use top-ranked chunks to build bridge queries, then re-rank the list of retrieved chunks.
5) Evaluate Recall@k / MRR@k against the gold-context origin, plus token-F1 of top-k chunks against the gold-context string.
6) Run the generation pipeline on the retrieved context and score EM/F1/containment with latency and peak-memory stats.

Run (retrieval only):
    python neural_rag.py --input ./data/raw_datasets/hotpotQA_compact/test/ds.parquet

Run (with generation):
    python neural_rag.py --pipeline regular --model_path <hf_model_dir>
    python neural_rag.py --pipeline doc2lora --model_path <hypernet.bin> --context_mode per_chunk
"""

import argparse
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import spacy
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.hypernetwork.inference import (
    load_baseline,
    load_hypernet,
    run_baseline,
    run_hypernet,
)
from src.evaluation.retrieval_aware import (
    apply_refusal_credit,
    compute_containment,
    compute_em,
    compute_f1,
    compute_rouge_l,
    gold_in_retrieved,
    normalize_answer,
)

load_dotenv()

DEFAULT_INPUT = "./data/raw_datasets/hotpotQA_compact/test/ds.parquet"
DEFAULT_OUTPUT = "./data/retrieved/passage_ranker_2_outputs.json"
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_CHUNK_SIZE = 480
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 20
DEFAULT_SEED_PASSAGES = 3

# k values evaluated for retrieval and generation
K_VALUES = (10,)

# Keep tokenizer parallelism off to avoid noisy thread contention in HF tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DATA LOADING
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Normalize raw text fields into a consistent single-line format so downstream matching and scoring behave predictably
def normalize_text(text: Any) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Load dataset from disk, handling format differences so the rest of the pipeline can assume a uniform HuggingFace dataset object
def load_processed_dataset(input_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = Path(input_path).suffix.lower()
    if ext == ".parquet":
        ds = load_dataset("parquet", data_files=input_path)["train"]
    elif ext in {".json", ".jsonl"}:
        ds = load_dataset("json", data_files=input_path)["train"]
    else:
        ds = load_dataset("parquet", data_files=input_path)["train"]
    return ds


# Break the gold context into sentence-like units before normalization so we can later check whether any retrieved chunk contains a gold sentence substring
def _split_gold_sentences(raw) -> List[str]:
    """Extract non-empty sentence strings from a raw gold_context field.

    Normalizes whitespace within each sentence so substring matches line up with retrieved chunks
    """
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = str(raw or "").split("\n")
    return [normalize_text(s) for s in items if s and str(s).strip()]


# Extract and normalize the core fields from the dataset, handling quirks like nested lists and alternate field names
def get_examples_from_dataset(ds) -> Tuple[List[str], List[str], List[str], List[str], List[List[str]]]:
    required = {"context", "prompts", "responses"}
    if not required.issubset(ds.column_names):
        raise ValueError(
            "Expected columns: context, prompts, responses (and gold_context or needle_text). "
            f"Found columns: {ds.column_names}"
        )

    if "gold_context" in ds.column_names:
        gold_field = "gold_context"
    elif "needle_text" in ds.column_names:
        gold_field = "needle_text"
    else:
        gold_field = "context"

    contexts = []
    gold_contexts = []
    gold_sentences_list: List[List[str]] = []
    for c in ds["context"]:
        # Flatten list-valued contexts into a single passage before normalization
        if isinstance(c, (list, tuple)):
            c = " ".join(str(x) for x in c)
        contexts.append(normalize_text(c))

    for gc in ds[gold_field]:
        # Preserve sentence boundaries from the raw field before whitespace is collapsed
        gold_sentences_list.append(_split_gold_sentences(gc))
        if isinstance(gc, (list, tuple)):
            gc = " ".join(str(x) for x in gc)
        gold_contexts.append(normalize_text(gc))

    questions = []
    answers = []

    for q, a in zip(ds["prompts"], ds["responses"]):
        # Prompts and responses are sometimes nested lists; use the first item when that happens
        if isinstance(q, (list, tuple)) and q:
            q = q[0]
        if isinstance(a, (list, tuple)) and a:
            a = a[0]
        questions.append(normalize_text(q))
        answers.append(normalize_text(a))

    return contexts, questions, answers, gold_contexts, gold_sentences_list


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RETRIEVAL HELPERS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Split a document into overlapping token-based chunks so retrieval operates over manageable passage units instead of full documents
def chunk_text(tokenizer, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
    if not tokens:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# Deduplicate text items while preserving order so bridge terms stay focused instead of repeating near-duplicates
def ordered_unique(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        cleaned = normalize_text(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


# Extract salient terms from text to build bridge queries that expand the original question using retrieved context
def extract_bridge_terms(text: str, nlp, max_terms: int = 24) -> List[str]:
    doc = nlp(text)
    candidates: List[str] = []

    for ent in doc.ents:
        if ent.text.strip():
            candidates.append(ent.text)

    try:
        for chunk in doc.noun_chunks:
            if chunk.text.strip():
                candidates.append(chunk.text)
    except Exception:
        pass

    for tok in doc:
        if tok.is_space or tok.is_punct or tok.is_stop:
            continue
        if tok.pos_ in {"NOUN", "PROPN", "ADJ", "NUM"}:
            lemma = tok.lemma_.strip()
            if lemma and lemma != "-PRON-":
                candidates.append(lemma)

    return ordered_unique(candidates)[:max_terms]


# Combine the original question with important terms from seed passages to form an expanded query for second-stage retrieval
def build_bridge_query(question: str, seed_texts: List[str], nlp, max_terms: int = 24) -> str:
    bridge_source = " ".join(seed_texts).strip()
    bridge_terms = extract_bridge_terms(bridge_source, nlp, max_terms=max_terms)
    parts = [normalize_text(question)] + bridge_terms
    return " ".join(ordered_unique(parts))


# Rescale scores to [0,1] so different scoring signals can be combined fairly
def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@torch.no_grad()
# Encode text into sparse SPLADE representations using MLM logits and max pooling over tokens
def splade_encode(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 16,
    show_progress: bool = False,
) -> List[torch.Tensor]:
    model.eval()
    outputs: List[torch.Tensor] = []

    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="SPLADE encode", total=(len(texts) + batch_size - 1) // batch_size)

    for start in iterator:
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # Encode the batch once, then use max pooling over tokens to get one sparse passage vector per text
        out = model(**enc)
        logits = out.logits
        activations = torch.log1p(torch.relu(logits))

        if "attention_mask" in enc:
            mask = enc["attention_mask"].unsqueeze(-1).to(dtype=activations.dtype)
            activations = activations * mask

        sparse_vec = activations.max(dim=1).values
        outputs.extend([vec.detach().cpu() for vec in sparse_vec])

    return outputs


# Compute similarity scores between a query vector and all passage vectors using a fast matrix-vector product
def score_query_against_matrix(query_vec: torch.Tensor, matrix: torch.Tensor) -> List[float]:
    q = query_vec.to(dtype=matrix.dtype, device=matrix.device)
    return torch.mv(matrix, q).tolist()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# METRICS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Heuristically extract a short answer span from model output for EM and F1 scoring
def extract_answer_span(text: str, gold: str) -> str:
    if not text:
        return ""

    gold_norm = normalize_answer(gold)

    first_word = re.split(r"[\s,.!?:;]+", text.strip().lower(), maxsplit=1)[0]
    if gold_norm in {"yes", "no"} and first_word in {"yes", "no"}:
        return first_word

    t = text.strip()
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = t.replace("**", "").replace("*", "").replace("`", "")

    for line in t.split("\n"):
        line = re.sub(r"^[\-\*#\d\.\)]+\s*", "", line.strip())
        if line:
            t = line
            break

    m = re.match(r"^(.+?[.!?])(?:\s|$)", t)
    if m:
        t = m.group(1).rstrip(".!?")

    return t.strip()


# Check whether any of the top-k retrieved chunks came from the correct document
def compute_recall_index(
    scored_pids: List[Tuple[int, float]],
    gold_example_id: int,
    chunk_id_to_original: List[int],
    k: int,
) -> int:
    """Index-based recall: any top-k chunk's source doc id matches query's example id."""
    for pid, _ in scored_pids[:k]:
        if chunk_id_to_original[pid] == gold_example_id:
            return 1
    return 0


# Compute reciprocal rank of the first correctly sourced chunk
def compute_mrr_index(
    scored_pids: List[Tuple[int, float]],
    gold_example_id: int,
    chunk_id_to_original: List[int],
    k: int,
) -> float:
    """Index-based MRR: rank of first top-k chunk whose source doc id matches."""
    for i, (pid, _) in enumerate(scored_pids[:k]):
        if chunk_id_to_original[pid] == gold_example_id:
            return 1.0 / (i + 1)
    return 0.0


# Check whether any retrieved chunk contains a gold sentence substring
def compute_recall_text(retrieved_texts: List[str], gold_sentences: List[str], k: int) -> int:
    """Text-based recall: any gold sentence appears as substring of any top-k chunk."""
    top_k = retrieved_texts[:k]
    return int(any(
        any(gold.lower() in ctx.lower() for ctx in top_k)
        for gold in gold_sentences
    ))


# Compute reciprocal rank of the first chunk that contains a gold sentence
def compute_mrr_text(retrieved_texts: List[str], gold_sentences: List[str], k: int) -> float:
    """Text-based MRR: rank of first chunk containing any gold sentence."""
    for i, ctx in enumerate(retrieved_texts[:k]):
        if any(gold.lower() in ctx.lower() for gold in gold_sentences):
            return 1.0 / (i + 1)
    return 0.0


# Measure how well retrieved chunks match the full gold context using token-level F1
def compute_f1_gold(retrieved_texts: List[str], gold_context: str, k: int) -> float:
    """Best token-F1 between any top-k chunk and the full gold context string."""
    best_f1 = 0.0
    for ctx in retrieved_texts[:k]:
        best_f1 = max(best_f1, compute_f1(ctx, gold_context))
    return best_f1


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL CORPUS BUILD + QUERY RETRIEVAL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Build a single global corpus of chunks from all contexts and track which original example each chunk came from
def build_global_corpus(
    contexts: List[str],
    tokenizer: AutoTokenizer,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[str], List[int]]:
    all_chunks: List[str] = []
    chunk_id_to_original: List[int] = []
    for i, ctx in enumerate(contexts):
        chunks = chunk_text(tokenizer, ctx, chunk_size=chunk_size, overlap=chunk_overlap)
        if not chunks:
            chunks = [ctx] if ctx else []
        all_chunks.extend(chunks)
        chunk_id_to_original.extend([i] * len(chunks))
    return all_chunks, chunk_id_to_original


# Rank all corpus chunks for a query using base SPLADE retrieval plus bridge-query reranking
def rank_query_global(
    question: str,
    all_chunks: List[str],
    passage_matrix: torch.Tensor,
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    nlp,
    device: torch.device,
    max_length: int,
    batch_size: int,
    seed_passages: int = DEFAULT_SEED_PASSAGES,
    bridge_weight: float = 0.70,
    max_bridge_terms: int = 24,
) -> List[Tuple[int, float]]:
    q_vec = splade_encode([question], tokenizer, model, device, max_length=max_length, batch_size=1)[0]

    base_raw = score_query_against_matrix(q_vec, passage_matrix)
    base_scores = normalize_scores(base_raw)

    # Use the top seed passages to build the bridge queries
    seed_ids = sorted(range(len(base_scores)), key=lambda i: base_scores[i], reverse=True)[: max(1, seed_passages)]
    bridge_queries: List[str] = []
    for pid in seed_ids:
        bq = build_bridge_query(
            question=question,
            seed_texts=[all_chunks[pid]],
            nlp=nlp,
            max_terms=max_bridge_terms,
        )
        if bq.strip():
            bridge_queries.append(bq)

    bridge_scores = [0.0 for _ in all_chunks]
    if bridge_queries:
        bridge_vecs = splade_encode(
            bridge_queries,
            tokenizer,
            model,
            device,
            max_length=max_length,
            batch_size=batch_size,
        )
        for bv in bridge_vecs:
            raw = score_query_against_matrix(bv, passage_matrix)
            norm = normalize_scores(raw)
            for i, sc in enumerate(norm):
                # Keep the best score seen for a passage across all bridge passes
                if sc > bridge_scores[i]:
                    bridge_scores[i] = sc

    final_scores = [b + bridge_weight * br for b, br in zip(base_scores, bridge_scores)]
    scored = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    return scored


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GENERATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Estimate token length of the formatted prompt so we can check context-window fit before generation
def _prompt_token_count(tokenizer, context_str, query, answer_style):
    """Return the number of tokens in the chat-formatted prompt (no generation tokens)."""
    if answer_style == "full":
        user_content = (
            f"Answer the question fully and completely based on the given passages. "
            f"Your answer should cover all relevant aspects and may be multiple sentences, but keep it concise. "
            f"If the passages do not contain enough information to answer, reply with exactly: answer not in context.\n\n"
            f"Passages:\n{context_str}\n\n"
            f"Question: {query}"
        )
    else:
        user_content = (
            f"Answer the question using only the given passages. "
            f"If the passages do not contain the answer, reply with exactly: answer not in context. "
            f"Otherwise give only the answer and do not output any other words.\n\n"
            f"Passages:\n{context_str}\n\n"
            f"Question: {query}"
        )
    # Measure the prompt after chat templating, since that is what actually hits the model
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return tokens.shape[1]


# Check whether the prompt plus generation budget stays within model limits
def context_fits_in_window(tokenizer, context_str, query, answer_style, max_new_tokens, model_max_len):
    """Return True if prompt + generation budget fits within the model's context window."""
    prompt_len = _prompt_token_count(tokenizer, context_str, query, answer_style)
    return (prompt_len + max_new_tokens) <= model_max_len


# Load the appropriate generation pipeline depending on the chosen model path
def load_generator(pipeline: str, model_path: str | None):
    if pipeline == "doc2lora":
        return load_hypernet(model_path) if model_path else load_hypernet()
    if pipeline == "regular":
        resolved = model_path
        if resolved and os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        return load_baseline(resolved) if resolved else load_baseline()
    raise ValueError(f"Unknown pipeline: {pipeline}")


# Run the generation model on retrieved context and return the answer with latency and peak GPU memory
def generate_answer(pipeline, model, tokenizer, context, query, max_new_tokens, answer_style="short"):
    example = {
        "context": context,
        "prompts": [query],
        "responses": [""],
    }
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None

    # Clear CUDA state so latency and peak-memory numbers reflect this example only
    if cuda_device is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)

    t0 = time.time()
    if pipeline == "doc2lora":
        outputs = run_hypernet(model, tokenizer, example, max_new_tokens=max_new_tokens, answer_style=answer_style)
    else:
        outputs = run_baseline(model, tokenizer, example, max_new_tokens=max_new_tokens,
                               answer_style=answer_style)

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    latency = time.time() - t0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 2)
        if cuda_device is not None
        else 0.0
    )
    return outputs[0].strip(), latency, peak_mem_mb


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# End-to-end pipeline:
# - load data
# - build and encode global corpus
# - retrieve chunks for each query
# - compute retrieval metrics
# - optionally run generation and evaluate answers
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--spacy_model", type=str, default=DEFAULT_SPACY_MODEL)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_examples", type=int, default=-1)
    parser.add_argument("--top_k_passages", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed_passages", type=int, default=DEFAULT_SEED_PASSAGES)
    parser.add_argument("--bridge_weight", type=float, default=0.70)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--corpus_device",
        type=str,
        default="cpu",
        help="Device to hold the passage matrix on (cpu or cuda). CPU is safer for large corpora.",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pipeline",
        choices=["doc2lora", "regular", "none"],
        default="none",
        help="Generation pipeline: doc2lora (hypernetwork), regular (LLM), or none (skip).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Checkpoint/model directory. For doc2lora: hypernet .bin. For regular: HF model dir.",
    )
    parser.add_argument(
        "--context_mode",
        choices=["joined", "per_chunk"],
        default="joined",
        help="joined: concat top-K into one context. per_chunk: pass each retrieved chunk as list (doc2lora only).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--gen_output",
        type=str,
        default="./data/retrieved/neural_rag_gen_outputs.json",
    )
    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Path to save a JSON of final retrieval (and generation) metrics only.",
    )
    parser.add_argument(
        "--answer_style",
        choices=["short", "full"],
        default="short",
        help=(
            "short: terse answer + span extraction + EM/F1 (HotpotQA). "
            "full: long answer + no span extraction + ROUGE-L (ASQA)."
        ),
    )
    args = parser.parse_args()

    print(f"Loading data: {args.input}")
    ds = load_processed_dataset(args.input)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    contexts, questions, answers, gold_contexts, gold_sentences_list = get_examples_from_dataset(ds)
    print(f"Loaded {len(questions)} examples")
    print(f"Columns: {ds.column_names}")
    print(f"Loading model: {args.model}")
    print(f"Loading spaCy model: {args.spacy_model}")

    # Put the retrieval model on the requested device, but keep the corpus matrix on the cheaper storage device
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    nlp = spacy.load(args.spacy_model)

    # Build the shared chunk corpus once instead of per query for efficiency
    print("\nBuilding global corpus...")
    t_chunk = time.time()
    all_chunks, chunk_id_to_original = build_global_corpus(
        contexts,
        tokenizer,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunk_build_sec = time.time() - t_chunk
    print(f"Total chunks: {len(all_chunks)} (from {len(contexts)} contexts) in {chunk_build_sec:.2f}s")

    # Encode all chunks into SPLADE vectors once; retrieval reuses these for every query
    print("\nEncoding corpus with SPLADE...")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t_encode = time.time()
    passage_vecs = splade_encode(
        all_chunks,
        tokenizer,
        model,
        device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        show_progress=True,
    )
    # Keep the passage matrix in a simple dense tensor so scoring stays fast and easy to batch
    passage_matrix = torch.stack([v.to(dtype=torch.float32) for v in passage_vecs], dim=0)
    corpus_device = torch.device(args.corpus_device)
    passage_matrix = passage_matrix.to(corpus_device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    corpus_encode_sec = time.time() - t_encode
    corpus_encode_peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda"
        else 0.0
    )
    corpus_matrix_mb = (passage_matrix.element_size() * passage_matrix.nelement()) / (1024 ** 2)
    print(
        f"Passage matrix shape: {tuple(passage_matrix.shape)} on {corpus_device} "
        f"(~{corpus_matrix_mb:.1f} MB); SPLADE encode took {corpus_encode_sec:.2f}s"
    )

    recall_text = {k: 0 for k in K_VALUES}
    recall_index = {k: 0 for k in K_VALUES}
    mrr_text = {k: 0.0 for k in K_VALUES}
    mrr_index = {k: 0.0 for k in K_VALUES}
    f1_k = {k: 0.0 for k in K_VALUES}

    # Only run generation when the user asked for it
    run_generation = args.pipeline != "none"
    gen_em_total = {k: 0.0 for k in K_VALUES}
    gen_f1_total = {k: 0.0 for k in K_VALUES}
    gen_contain_total = {k: 0.0 for k in K_VALUES}
    gen_rouge_l_total = {k: 0.0 for k in K_VALUES}
    gen_em_aware_total = {k: 0.0 for k in K_VALUES}
    gen_f1_aware_total = {k: 0.0 for k in K_VALUES}
    gen_contain_aware_total = {k: 0.0 for k in K_VALUES}
    gen_rouge_l_aware_total = {k: 0.0 for k in K_VALUES}
    gen_refused_total = {k: 0 for k in K_VALUES}
    gen_gold_found_total = {k: 0 for k in K_VALUES}
    gen_refused_correct_total = {k: 0 for k in K_VALUES}
    gen_latencies: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    gen_peak_mems: Dict[int, List[float]] = {k: [] for k in K_VALUES}
    gen_skipped: Dict[int, int] = {k: 0 for k in K_VALUES}
    gen_model = None
    gen_tokenizer = None
    model_max_len = None

    # Set up the generator model and tokenizer once before the evaluation loop starts
    if run_generation:
        print(f"\nLoading generator: pipeline={args.pipeline}, model_path={args.model_path}")
        gen_model, gen_tokenizer = load_generator(args.pipeline, args.model_path)

        if args.pipeline == "regular":
            try:
                model_max_len = gen_model.config.max_position_embeddings
            except Exception:
                model_max_len = 8192

            print(f"Model context window: {model_max_len} tokens")

    retrieved_records: List[Dict[str, Any]] = []
    retrieval_latencies: List[float] = []
    retrieval_peak_mems: List[float] = []

    # Loop over each example: retrieve top chunks, compute retrieval metrics, and optionally run generation
    print("\nRunning evaluation...")
    for i in tqdm(range(len(questions)), desc="Eval Progress"):
        query = questions[i]
        true_answer = answers[i]
        # context = contexts[i]
        gold_context = gold_contexts[i]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)

        t_q = time.time()
        # Rank all chunks for this question using the base SPLADE score plus bridge-query reranking
        scored_pids = rank_query_global(
            question=query,
            all_chunks=all_chunks,
            passage_matrix=passage_matrix,
            tokenizer=tokenizer,
            model=model,
            nlp=nlp,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            seed_passages=args.seed_passages,
            bridge_weight=args.bridge_weight,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        query_latency = time.time() - t_q
        query_peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        )
        retrieval_latencies.append(query_latency)
        retrieval_peak_mems.append(query_peak_mem_mb)

        # Only the top retrieved texts are used for the retrieval metrics below
        metric_texts = [all_chunks[pid] for pid, _ in scored_pids[:max(K_VALUES)]]
        gold_sentences = gold_sentences_list[i]

        for k in K_VALUES:
            recall_text[k] += compute_recall_text(metric_texts, gold_sentences, k=k)
            recall_index[k] += compute_recall_index(scored_pids, i, chunk_id_to_original, k=k)
            mrr_text[k] += compute_mrr_text(metric_texts, gold_sentences, k=k)
            mrr_index[k] += compute_mrr_index(scored_pids, i, chunk_id_to_original, k=k)
            f1_k[k] += compute_f1_gold(metric_texts, gold_context, k=k)

        top_texts = [all_chunks[pid] for pid, _ in scored_pids[:args.top_k_passages]]
        top_origins = [chunk_id_to_original[pid] for pid, _ in scored_pids[:args.top_k_passages]]

        # Save both the retrieved text and the original example index for later inspection
        record = {
            "id": i,
            "prompt": query,
            "gold_context": gold_context,
            "retrieved_context": top_texts,
            "retrieved_origins": top_origins,
            "answer": normalize_answer(true_answer),
            "scored_pids": scored_pids[:args.top_k_passages],
        }

        if run_generation:
            gen_results: Dict[int, Any] = {}

            # For each k, build context from retrieved chunks and run generation plus scoring
            for k in K_VALUES:
                top_k_texts = top_texts[:k]

                if args.pipeline == "doc2lora":
                    gen_context = list(top_k_texts)
                else:
                    gen_context = "\n\n".join(top_k_texts)

                    # Do not waste time generating if the prompt would exceed the model's context window
                    if model_max_len is not None and not context_fits_in_window(
                        gen_tokenizer,
                        gen_context,
                        query,
                        args.answer_style,
                        args.max_new_tokens,
                        model_max_len,
                    ):
                        gen_skipped[k] += 1
                        gen_em_total[k] += 0.0
                        gen_f1_total[k] += 0.0
                        gen_contain_total[k] += 0.0
                        gen_rouge_l_total[k] += 0.0

                        if gold_in_retrieved(true_answer, top_k_texts):
                            gen_gold_found_total[k] += 1

                        gen_results[k] = {"skipped": "context_too_long"}
                        continue

                try:
                    # Generate the answer and record latency and peak memory for this one example
                    raw_prediction, latency, peak_mem_mb = generate_answer(
                        args.pipeline,
                        gen_model,
                        gen_tokenizer,
                        gen_context,
                        query,
                        args.max_new_tokens,
                        answer_style=args.answer_style,
                    )
                except RuntimeError as e:
                    # These runtime errors are usually shape or CUDA issues, so skip the sample and keep going
                    if "aligned" in str(e).lower() or "cuda" in str(e).lower():
                        print(f"[WARN] Skipping sample {i} k={k} due to error: {e}")
                        gen_skipped[k] += 1
                        gen_em_total[k] += 0.0
                        gen_f1_total[k] += 0.0
                        gen_contain_total[k] += 0.0
                        gen_rouge_l_total[k] += 0.0

                        if gold_in_retrieved(true_answer, top_k_texts):
                            gen_gold_found_total[k] += 1

                        gen_results[k] = {"skipped": "cuda_error"}
                        continue

                    raise

                # Short answers get span extraction; full answers stay untouched for ROUGE-style scoring
                if args.answer_style == "full":
                    prediction = raw_prediction
                else:
                    prediction = extract_answer_span(raw_prediction, true_answer)

                em = compute_em(prediction, true_answer)
                f1 = compute_f1(prediction, true_answer)
                contain = compute_containment(raw_prediction, true_answer)
                rouge_l = compute_rouge_l(prediction, true_answer)

                pred_for_refusal_check = prediction if args.answer_style != "full" else raw_prediction
                # Apply refusal-aware scoring so explicit refusals are not treated the same as wrong answers
                em_a, f1_a, rouge_l_a, contain_a, refused_correct, refused, gold_found = apply_refusal_credit(
                    em,
                    f1,
                    rouge_l,
                    contain,
                    pred_for_refusal_check,
                    true_answer,
                    top_k_texts,
                )

                gen_em_total[k] += em
                gen_f1_total[k] += f1
                gen_contain_total[k] += contain
                gen_rouge_l_total[k] += rouge_l
                gen_em_aware_total[k] += em_a
                gen_f1_aware_total[k] += f1_a
                gen_contain_aware_total[k] += contain_a
                gen_rouge_l_aware_total[k] += rouge_l_a
                gen_refused_total[k] += int(refused)
                gen_gold_found_total[k] += int(gold_found)
                gen_refused_correct_total[k] += int(refused_correct)
                gen_latencies[k].append(latency)
                gen_peak_mems[k].append(peak_mem_mb)

                gen_results[k] = {
                    "prediction_raw": raw_prediction,
                    "prediction": normalize_answer(prediction),
                    "gen_em": em,
                    "gen_f1": f1,
                    "gen_contain": contain,
                    "gen_rouge_l": rouge_l,
                    "gen_em_aware": em_a,
                    "gen_f1_aware": f1_a,
                    "gen_contain_aware": contain_a,
                    "gen_rouge_l_aware": rouge_l_a,
                    "is_refusal": refused,
                    "gold_in_retrieved": gold_found,
                    "refused_correctly": refused_correct,
                    "gen_latency_sec": latency,
                    "gen_peak_mem_mb": peak_mem_mb,
                }

            record["generation"] = gen_results

        # Persist the per-example retrieval trace so the results are easy to audit later
        retrieved_records.append(record)

    num_samples = len(questions)
    # Average each metric over the full dataset once the loop is done
    for k in K_VALUES:
        recall_text[k] /= num_samples
        recall_index[k] /= num_samples
        mrr_text[k] /= num_samples
        mrr_index[k] /= num_samples
        f1_k[k] /= num_samples

    print("\n===== RETRIEVAL (TEXT: gold substring in retrieved chunk) =====")
    for k in K_VALUES:
        print(f"Recall@{k}: {recall_text[k]:.4f}   MRR@{k}: {mrr_text[k]:.4f}")

    print("\n===== RETRIEVAL (INDEX: retrieved chunk's source doc == query's doc) =====")
    for k in K_VALUES:
        print(f"Recall@{k}: {recall_index[k]:.4f}   MRR@{k}: {mrr_index[k]:.4f}")

    print("\n===== RETRIEVAL F1 (vs full gold context) =====")
    for k in K_VALUES:
        print(f"F1@{k}: {f1_k[k]:.4f}")

    sorted_latencies = sorted(retrieval_latencies)
    avg_query_sec = sum(sorted_latencies) / len(sorted_latencies) if sorted_latencies else 0.0
    p95_idx = max(0, int(0.95 * (len(sorted_latencies) - 1))) if sorted_latencies else 0
    p95_query_sec = sorted_latencies[p95_idx] if sorted_latencies else 0.0
    avg_peak_mem_mb_retrieval = (
        sum(retrieval_peak_mems) / len(retrieval_peak_mems) if retrieval_peak_mems else 0.0
    )
    max_peak_mem_mb_retrieval = max(retrieval_peak_mems) if retrieval_peak_mems else 0.0

    print("\n===== RETRIEVAL LATENCY =====")
    print(f"Corpus build (s): {chunk_build_sec:.2f}")
    print(f"Corpus SPLADE enc (s): {corpus_encode_sec:.2f}")
    print(f"Corpus matrix size MB: {corpus_matrix_mb:.1f}")
    print(f"Corpus enc peak mem MB: {corpus_encode_peak_mb:.1f}")
    print(f"Avg query time (s): {avg_query_sec:.4f}")
    print(f"p95 query time (s): {p95_query_sec:.4f}")
    print(f"Avg peak mem MB: {avg_peak_mem_mb_retrieval:.1f}")
    print(f"Max peak mem MB: {max_peak_mem_mb_retrieval:.1f}")

    if run_generation:
        print(f"\n===== GENERATION ({args.pipeline}, answer_style={args.answer_style}) =====")
        gen_summary: Dict[str, Any] = {}

        # Summarize generation metrics separately for each top-k setting
        for k in K_VALUES:
            g_em = gen_em_total[k] / num_samples
            g_f1 = gen_f1_total[k] / num_samples
            g_contain = gen_contain_total[k] / num_samples
            g_rouge_l = gen_rouge_l_total[k] / num_samples

            g_em_a = gen_em_aware_total[k] / num_samples
            g_f1_a = gen_f1_aware_total[k] / num_samples
            g_contain_a = gen_contain_aware_total[k] / num_samples
            g_rouge_l_a = gen_rouge_l_aware_total[k] / num_samples

            refusal_rate = gen_refused_total[k] / num_samples
            retrieval_fail_rate = (num_samples - gen_gold_found_total[k]) / num_samples
            refused_correct_rate = gen_refused_correct_total[k] / num_samples

            lats = gen_latencies[k]
            mems = gen_peak_mems[k]
            avg_lat = sum(lats) / len(lats) if lats else 0.0
            avg_mem = sum(mems) / len(mems) if mems else 0.0
            max_mem = max(mems) if mems else 0.0
            skipped = gen_skipped[k]

            print(f"\n  --- top-{k} context ---")
            print(f"Answer EM: {g_em:.4f} (aware: {g_em_a:.4f})")
            print(f"Answer F1: {g_f1:.4f} (aware: {g_f1_a:.4f})")
            print(f"ROUGE-L: {g_rouge_l:.4f} (aware: {g_rouge_l_a:.4f})")
            print(f"Containment: {g_contain:.4f} (aware: {g_contain_a:.4f})")
            print(
                f"Refusal rate: {refusal_rate:.4f} "
                f"Retrieval-fail: {retrieval_fail_rate:.4f} "
                f"Correct refusals: {refused_correct_rate:.4f}"
            )

            if skipped:
                print(f"Skipped: {skipped}/{num_samples} (context too long or CUDA error)")

            print(f"Avg latency (s): {avg_lat:.4f}")
            print(f"Avg peak mem MB: {avg_mem:.1f}")
            print(f"Max peak mem MB: {max_mem:.1f}")

            gen_summary[str(k)] = {
                "answer_em": g_em,
                "answer_f1": g_f1,
                "answer_rouge_l": g_rouge_l,
                "answer_contain": g_contain,
                "answer_em_aware": g_em_a,
                "answer_f1_aware": g_f1_a,
                "answer_rouge_l_aware": g_rouge_l_a,
                "answer_contain_aware": g_contain_a,
                "refusal_rate": refusal_rate,
                "retrieval_fail_rate": retrieval_fail_rate,
                "refused_correct_rate": refused_correct_rate,
                "avg_latency_sec": avg_lat,
                "avg_peak_mem_mb": avg_mem,
                "max_peak_mem_mb": max_mem,
                "num_skipped": skipped,
                "num_samples": num_samples,
            }

        # Write the generation run output before the retrieval summary so both artifacts are available
        Path(args.gen_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.gen_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pipeline": args.pipeline,
                    "answer_style": args.answer_style,
                    "model_path": args.model_path,
                    "summary": gen_summary,
                    "records": retrieved_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Saved generation results to {args.gen_output}")

    # Save the retrieval-only results in a separate file for easier downstream use
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "input": args.input,
                "summary": {
                    "recall_text": {str(k): recall_text[k] for k in K_VALUES},
                    "recall_index": {str(k): recall_index[k] for k in K_VALUES},
                    "mrr_text": {str(k): mrr_text[k] for k in K_VALUES},
                    "mrr_index": {str(k): mrr_index[k] for k in K_VALUES},
                    "f1_vs_gold_context": {str(k): f1_k[k] for k in K_VALUES},
                    "num_samples": num_samples,
                    "num_chunks": len(all_chunks),
                    "corpus_build_sec": chunk_build_sec,
                    "corpus_encode_sec": corpus_encode_sec,
                    "corpus_matrix_mb": corpus_matrix_mb,
                    "corpus_encode_peak_mb": corpus_encode_peak_mb,
                    "avg_query_sec": avg_query_sec,
                    "p95_query_sec": p95_query_sec,
                    "avg_peak_mem_mb_retrieval": avg_peak_mem_mb_retrieval,
                    "max_peak_mem_mb_retrieval": max_peak_mem_mb_retrieval,
                },
                "records": retrieved_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved retrieval results to {args.output}")

    # Keep the metrics-only JSON small so it is easy to compare runs programmatically
    if args.metrics_output:
        metrics: Dict[str, Any] = {
            "model": args.model,
            "input": args.input,
            "num_samples": num_samples,
            "retrieval": {
                "recall_text": {str(k): recall_text[k] for k in K_VALUES},
                "recall_index": {str(k): recall_index[k] for k in K_VALUES},
                "mrr_text": {str(k): mrr_text[k] for k in K_VALUES},
                "mrr_index": {str(k): mrr_index[k] for k in K_VALUES},
                "f1_vs_gold_context": {str(k): f1_k[k] for k in K_VALUES},
                "avg_query_sec": avg_query_sec,
                "p95_query_sec": p95_query_sec,
                "corpus_build_sec": chunk_build_sec,
                "corpus_encode_sec": corpus_encode_sec,
            },
        }

        if run_generation:
            metrics["generation"] = {
                "answer_style": args.answer_style,
                "summary": gen_summary,
            }

        Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()

