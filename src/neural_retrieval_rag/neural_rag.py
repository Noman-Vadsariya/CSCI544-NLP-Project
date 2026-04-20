#!/usr/bin/env python3
"""
neural_rag.py

Multi-hop SPLADE passage ranker over a GLOBAL corpus.

Input format:
    parquet / HF dataset rows with columns:
      - context: string document text (the gold supporting doc for that query)
      - prompts: [question]
      - responses: [answer]

Pipeline:
1) Chunk every example's context and build one global corpus.
2) SPLADE-encode every chunk once.
3) For each question, score chunks against the full corpus.
4) Use top seeds to build bridge queries, then re-rank.
5) Evaluate Recall@k / MRR@k against the gold-context origin
   (chunk_id_to_original[pid] == query's example id), plus token-F1
   of top-k chunks against the gold-context string.
6) Optionally run a generation pipeline (doc2lora or regular LLM) on
   the retrieved context and score EM/F1/containment with latency and
   peak-memory stats (matches src/standard_rag/rag_colbert_reranker.py).

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
import string
import time
from collections import Counter
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

load_dotenv()

DEFAULT_INPUT = "./data/raw_datasets/hotpotQA_compact/test/ds.parquet"
DEFAULT_OUTPUT = "./data/retrieved/passage_ranker_2_outputs.json"
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"
DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_CHUNK_SIZE = 480
DEFAULT_CHUNK_OVERLAP = 80
DEFAULT_TOP_K = 10
DEFAULT_SEED_PASSAGES = 3

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

def normalize_text(text: Any) -> str:
    text = html.unescape(str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer(text: str) -> str:
    def remove_articles(t: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    def lower(t: str) -> str:
        return t.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


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


def get_examples_from_dataset(ds) -> Tuple[List[str], List[str], List[str], List[str]]:
    if "context" not in ds.column_names or "prompts" not in ds.column_names or "responses" not in ds.column_names or "gold_context" not in ds.column_names:
        raise ValueError(
            "Expected columns: context, prompts, responses, gold_context. "
            f"Found columns: {ds.column_names}"
        )

    contexts = []
    gold_contexts = []
    for c in ds["context"]:
        if isinstance(c, (list, tuple)):
            c = " ".join(str(x) for x in c)
        contexts.append(normalize_text(c))

    for gc in ds["gold_context"]:
        if isinstance(gc, (list, tuple)):
            gc = " ".join(str(x) for x in gc)
        gold_contexts.append(normalize_text(gc))

    questions = []
    answers = []

    for q, a in zip(ds["prompts"], ds["responses"]):
        if isinstance(q, (list, tuple)) and q:
            q = q[0]
        if isinstance(a, (list, tuple)) and a:
            a = a[0]
        questions.append(normalize_text(q))
        answers.append(normalize_text(a))

    return contexts, questions, answers, gold_contexts


# -------------------------------------------------------------------
# RETRIEVAL HELPERS
# -------------------------------------------------------------------

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


def build_bridge_query(question: str, seed_texts: List[str], nlp, max_terms: int = 24) -> str:
    bridge_source = " ".join(seed_texts).strip()
    bridge_terms = extract_bridge_terms(bridge_source, nlp, max_terms=max_terms)
    parts = [normalize_text(question)] + bridge_terms
    return " ".join(ordered_unique(parts))


def normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@torch.no_grad()
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

        out = model(**enc)
        logits = out.logits
        activations = torch.log1p(torch.relu(logits))

        if "attention_mask" in enc:
            mask = enc["attention_mask"].unsqueeze(-1).to(dtype=activations.dtype)
            activations = activations * mask

        sparse_vec = activations.max(dim=1).values
        outputs.extend([vec.detach().cpu() for vec in sparse_vec])

    return outputs


def score_query_against_matrix(query_vec: torch.Tensor, matrix: torch.Tensor) -> List[float]:
    q = query_vec.to(dtype=matrix.dtype, device=matrix.device)
    return torch.mv(matrix, q).tolist()


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_containment(prediction: str, ground_truth: str) -> float:
    gold_norm = normalize_answer(ground_truth)
    if not gold_norm:
        return 0.0
    return float(gold_norm in normalize_answer(prediction))


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


def compute_recall_at_k_origin(
    scored_pids: List[Tuple[int, float]],
    gold_example_id: int,
    chunk_id_to_original: List[int],
    k: int,
) -> int:
    for pid, _ in scored_pids[:k]:
        if chunk_id_to_original[pid] == gold_example_id:
            return 1
    return 0


def compute_mrr_at_k_origin(
    scored_pids: List[Tuple[int, float]],
    gold_example_id: int,
    chunk_id_to_original: List[int],
    k: int,
) -> float:
    for i, (pid, _) in enumerate(scored_pids[:k]):
        if chunk_id_to_original[pid] == gold_example_id:
            return 1.0 / (i + 1)
    return 0.0


# -------------------------------------------------------------------
# GLOBAL CORPUS BUILD + QUERY RETRIEVAL
# -------------------------------------------------------------------

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
                if sc > bridge_scores[i]:
                    bridge_scores[i] = sc

    final_scores = [b + bridge_weight * br for b, br in zip(base_scores, bridge_scores)]
    scored = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    return scored


# -------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------

HYPERNET_QUERY_PREFIX = (
    "Answer the question in as few words as possible. "
    "Only output the answer itself, no explanation or extra text.\n\n"
)


def load_generator(pipeline: str, model_path: str | None):
    if pipeline == "doc2lora":
        return load_hypernet(model_path) if model_path else load_hypernet()
    if pipeline == "regular":
        resolved = model_path
        if resolved and os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        return load_baseline(resolved) if resolved else load_baseline()
    raise ValueError(f"Unknown pipeline: {pipeline}")


def generate_answer(pipeline, model, tokenizer, context, query, max_new_tokens):
    example = {
        "context": context,
        "prompts": [query],
        "responses": [""],
    }
    cuda_device = torch.cuda.current_device() if torch.cuda.is_available() else None

    if cuda_device is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        torch.cuda.synchronize(cuda_device)

    t0 = time.time()
    if pipeline == "doc2lora":
        outputs = run_hypernet(model, tokenizer, example, max_new_tokens=max_new_tokens)
    else:
        outputs = run_baseline(model, tokenizer, example, max_new_tokens=max_new_tokens)

    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    latency = time.time() - t0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 2)
        if cuda_device is not None
        else 0.0
    )
    return outputs[0].strip(), latency, peak_mem_mb


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> None:
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
    parser.add_argument("--corpus_device", type=str, default="cpu",
                        help="Device to hold the passage matrix on (cpu or cuda). CPU is safer for large corpora.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pipeline",
        choices=["doc2lora", "regular", "none"],
        default="none",
        help="Generation pipeline: doc2lora (hypernetwork), regular (LLM), or none (skip).",
    )
    parser.add_argument("--model_path", type=str, default=None,
                        help="Checkpoint/model directory. For doc2lora: hypernet .bin. For regular: HF model dir.")
    parser.add_argument(
        "--context_mode",
        choices=["joined", "per_chunk"],
        default="joined",
        help="joined: concat top-K into one context. per_chunk: pass each retrieved chunk as list (doc2lora only).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--gen_output", type=str,
                        default="./data/retrieved/neural_rag_gen_outputs.json")
    args = parser.parse_args()

    print(f"Loading data: {args.input}")
    ds = load_processed_dataset(args.input)
    if args.max_examples > 0:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    contexts, questions, answers, gold_contexts = get_examples_from_dataset(ds)
    print(f"Loaded {len(questions)} examples")
    print(f"Columns: {ds.column_names}")
    print(f"Loading model: {args.model}")
    print(f"Loading spaCy model: {args.spacy_model}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    nlp = spacy.load(args.spacy_model)

    print("\nBuilding global corpus...")
    t_chunk = time.time()
    all_chunks, chunk_id_to_original = build_global_corpus(
        contexts, tokenizer, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    chunk_build_sec = time.time() - t_chunk
    print(f"Total chunks: {len(all_chunks)} (from {len(contexts)} contexts) in {chunk_build_sec:.2f}s")

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
    passage_matrix = torch.stack([v.to(dtype=torch.float32) for v in passage_vecs], dim=0)
    corpus_device = torch.device(args.corpus_device)
    passage_matrix = passage_matrix.to(corpus_device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    corpus_encode_sec = time.time() - t_encode
    corpus_encode_peak_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.type == "cuda" else 0.0
    )
    corpus_matrix_mb = (passage_matrix.element_size() * passage_matrix.nelement()) / (1024 ** 2)
    print(
        f"Passage matrix shape: {tuple(passage_matrix.shape)} on {corpus_device} "
        f"(~{corpus_matrix_mb:.1f} MB); SPLADE encode took {corpus_encode_sec:.2f}s"
    )

    recall_2 = 0
    recall_5 = 0
    recall_10 = 0
    mrr_2 = 0.0
    mrr_5 = 0.0
    mrr_10 = 0.0
    f1_total = 0.0

    run_generation = args.pipeline != "none"
    gen_em_total = 0.0
    gen_f1_total = 0.0
    gen_contain_total = 0.0
    gen_latencies: List[float] = []
    gen_peak_mems: List[float] = []
    gen_model = None
    gen_tokenizer = None

    if run_generation:
        print(f"\nLoading generator: pipeline={args.pipeline}, model_path={args.model_path}")
        gen_model, gen_tokenizer = load_generator(args.pipeline, args.model_path)

    retrieved_records: List[Dict[str, Any]] = []
    retrieval_latencies: List[float] = []
    retrieval_peak_mems: List[float] = []

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
            if device.type == "cuda" else 0.0
        )
        retrieval_latencies.append(query_latency)
        retrieval_peak_mems.append(query_peak_mem_mb)

        recall_2 += compute_recall_at_k_origin(scored_pids, i, chunk_id_to_original, k=2)
        recall_5 += compute_recall_at_k_origin(scored_pids, i, chunk_id_to_original, k=5)
        recall_10 += compute_recall_at_k_origin(scored_pids, i, chunk_id_to_original, k=10)
        mrr_2 += compute_mrr_at_k_origin(scored_pids, i, chunk_id_to_original, k=2)
        mrr_5 += compute_mrr_at_k_origin(scored_pids, i, chunk_id_to_original, k=5)
        mrr_10 += compute_mrr_at_k_origin(scored_pids, i, chunk_id_to_original, k=10)

        top_texts = [all_chunks[pid] for pid, _ in scored_pids[: args.top_k_passages]]
        top_origins = [chunk_id_to_original[pid] for pid, _ in scored_pids[: args.top_k_passages]]

        best_f1 = 0.0
        for ctx in top_texts:
            best_f1 = max(best_f1, compute_f1(ctx, gold_context))
        f1_total += best_f1

        record = {
            "id": i,
            "prompt": query,
            "gold_context": gold_context,
            "retrieved_context": top_texts,
            "retrieved_origins": top_origins,
            "answer": normalize_answer(true_answer),
            "scored_pids": scored_pids[: args.top_k_passages],
        }

        if run_generation:
            if args.context_mode == "per_chunk" and args.pipeline == "doc2lora":
                gen_context = list(top_texts)
            else:
                gen_context = "\n\n".join(top_texts)

            gen_query = (
                HYPERNET_QUERY_PREFIX + query
                if args.pipeline == "doc2lora"
                else query
            )

            raw_prediction, latency, peak_mem_mb = generate_answer(
                args.pipeline,
                gen_model,
                gen_tokenizer,
                gen_context,
                gen_query,
                args.max_new_tokens,
            )
            prediction = extract_answer_span(raw_prediction, true_answer)

            em = compute_em(prediction, true_answer)
            f1 = compute_f1(prediction, true_answer)
            contain = compute_containment(raw_prediction, true_answer)
            gen_em_total += em
            gen_f1_total += f1
            gen_contain_total += contain
            gen_latencies.append(latency)
            gen_peak_mems.append(peak_mem_mb)

            record["prediction_raw"] = raw_prediction
            record["prediction"] = normalize_answer(prediction)
            record["gen_em"] = em
            record["gen_f1"] = f1
            record["gen_contain"] = contain
            record["gen_latency_sec"] = latency
            record["gen_peak_mem_mb"] = peak_mem_mb

        retrieved_records.append(record)

    num_samples = len(questions)
    recall_2 /= num_samples
    recall_5 /= num_samples
    recall_10 /= num_samples
    mrr_2 = mrr_2 / num_samples
    mrr_5 = mrr_5 / num_samples
    mrr_10 = mrr_10 / num_samples
    f1_score = f1_total / num_samples

    print("\n===== RETRIEVAL (gold = originating context) =====")
    print(f"Recall@2       : {recall_2:.4f}")
    print(f"Recall@5       : {recall_5:.4f}")
    print(f"Recall@10      : {recall_10:.4f}")
    print(f"MRR@2         : {mrr_2:.4f}")
    print(f"MRR@5         : {mrr_5:.4f}")
    print(f"MRR@10        : {mrr_10:.4f}")
    print(f"F1 vs gold ctx : {f1_score:.4f}")

    sorted_latencies = sorted(retrieval_latencies)
    avg_query_sec = sum(sorted_latencies) / len(sorted_latencies) if sorted_latencies else 0.0
    p95_idx = max(0, int(0.95 * (len(sorted_latencies) - 1))) if sorted_latencies else 0
    p95_query_sec = sorted_latencies[p95_idx] if sorted_latencies else 0.0
    avg_peak_mem_mb_retrieval = (
        sum(retrieval_peak_mems) / len(retrieval_peak_mems) if retrieval_peak_mems else 0.0
    )
    max_peak_mem_mb_retrieval = max(retrieval_peak_mems) if retrieval_peak_mems else 0.0

    print("\n===== RETRIEVAL LATENCY =====")
    print(f"Corpus build (s)       : {chunk_build_sec:.2f}")
    print(f"Corpus SPLADE enc (s)  : {corpus_encode_sec:.2f}")
    print(f"Corpus matrix size MB  : {corpus_matrix_mb:.1f}")
    print(f"Corpus enc peak mem MB : {corpus_encode_peak_mb:.1f}")
    print(f"Avg query time (s)     : {avg_query_sec:.4f}")
    print(f"p95 query time (s)     : {p95_query_sec:.4f}")
    print(f"Avg peak mem MB        : {avg_peak_mem_mb_retrieval:.1f}")
    print(f"Max peak mem MB        : {max_peak_mem_mb_retrieval:.1f}")

    if run_generation:
        gen_em = gen_em_total / num_samples
        gen_f1 = gen_f1_total / num_samples
        gen_contain = gen_contain_total / num_samples
        avg_latency = sum(gen_latencies) / len(gen_latencies)
        avg_peak_mem = sum(gen_peak_mems) / len(gen_peak_mems)
        max_peak_mem = max(gen_peak_mems)

        print(f"\n===== GENERATION ({args.pipeline}) =====")
        print(f"Answer EM      : {gen_em:.4f}  (on cleaned prediction)")
        print(f"Answer F1      : {gen_f1:.4f}  (on cleaned prediction)")
        print(f"Containment    : {gen_contain:.4f}  (gold in raw prediction)")
        print(f"Avg latency (s): {avg_latency:.4f}")
        print(f"Avg peak mem MB: {avg_peak_mem:.1f}")
        print(f"Max peak mem MB: {max_peak_mem:.1f}")

        Path(args.gen_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.gen_output, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pipeline": args.pipeline,
                    "model_path": args.model_path,
                    "summary": {
                        "answer_em": gen_em,
                        "answer_f1": gen_f1,
                        "answer_contain": gen_contain,
                        "avg_latency_sec": avg_latency,
                        "avg_peak_mem_mb": avg_peak_mem,
                        "max_peak_mem_mb": max_peak_mem,
                        "num_samples": num_samples,
                    },
                    "records": retrieved_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved generation results to {args.gen_output}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "input": args.input,
                "summary": {
                    "recall_at_2": recall_2,
                    "recall_at_5": recall_5,
                    "recall_at_10": recall_10,
                    "mrr_at_2": mrr_2,
                    "mrr_at_5": mrr_5,
                    "mrr_at_10": mrr_10,
                    "f1_vs_gold_context": f1_score,
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


if __name__ == "__main__":
    main()
