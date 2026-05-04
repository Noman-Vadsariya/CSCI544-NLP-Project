"""
Generation-only script that reads pre-retrieved contexts from a JSON file (produced by rag_colbert_reranker.py or neural_rag.py) and runs generation
with the top-K retrieved passages, for K in (2, 5).

Usage:
    python -m src.standard_rag.gen_from_retrieved \
        --retrieved_input  data/retrieved/hotpotQA_compact_colbert_d2l_gemma.json \
        --pipeline         doc2lora \
        --model_path       checkpoints/... \
        --gen_output       data/retrieved/hotpotQA_compact_colbert_d2l_gemma_topk_gen.json
"""

import argparse
import json
import os
import time

import torch
from pathlib import Path
import re
from tqdm import tqdm

from src.hypernetwork.inference import (
    load_baseline,
    load_hypernet,
    run_baseline,
    run_hypernet,
)
from src.evaluation.retrieval_aware import (
    apply_refusal_credit,
    compute_contain,
    compute_em,
    compute_f1,
    compute_rouge_l,
    gold_in_retrieved,
    normalize_answer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TOP_KS = (2, 5, 10)

HYPERNET_QUERY_PREFIX = (
    "Answer the question in as few words as possible. "
    "Only output the answer itself, no explanation or extra text.\n\n"
)

# -------------------------------------------------------------------
# ARGS
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieved_input",
        type=str,
        required=True,
        help="Path to JSON file with pre-retrieved contexts (list of record dicts).",
    )
    parser.add_argument(
        "--pipeline",
        choices=["doc2lora", "regular"],
        required=True,
        help="Generation pipeline: doc2lora (hypernetwork) or regular (LLM).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Checkpoint/model directory.",
    )
    parser.add_argument(
        "--context_mode",
        choices=["joined", "per_chunk"],
        default="joined",
        help="joined: concat top-K into one string. per_chunk: pass list (doc2lora only).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit number of records to evaluate.",
    )
    parser.add_argument(
        "--sample_pct",
        type=float,
        default=None,
        help="Percentage of records to evaluate, e.g. 0.5 for 50%%. Overridden by --num_samples if both given.",
    )
    parser.add_argument(
        "--gen_output",
        type=str,
        default="./data/retrieved/gen_from_retrieved_output.json",
    )
    parser.add_argument(
        "--answer_style",
        choices=["short", "full"],
        default="short",
        help="short: terse answer (default). full: long-form (use for ASQA).",
    )
    parser.add_argument(
        "--no_query_prefix",
        action="store_true",
        help="Disable the short-answer prefix prepended to doc2lora queries. Use for long-form datasets like ASQA.",
    )
    parser.add_argument(
        "--top_ks",
        type=str,
        default=None,
        help="Comma-separated K values to evaluate (e.g. '2,5,10' or '20'). Defaults to TOP_KS.",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------


def extract_answer_span(text, gold):
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


# -------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------

def load_generator(pipeline, model_path):
    if pipeline == "doc2lora":
        return load_hypernet(model_path) if model_path else load_hypernet()
    if pipeline == "regular":
        resolved = model_path
        if resolved and os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        return load_baseline(resolved) if resolved else load_baseline()
    raise ValueError(f"Unknown pipeline: {pipeline}")


def generate_answer(pipeline, model, tokenizer, context, query, max_new_tokens, answer_style="short"):
    example = {
        "context": context,
        "prompts": [query],
        "responses": [""],
    }
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    if device is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    t0 = time.time()
    if pipeline == "doc2lora":
        outputs = run_hypernet(model, tokenizer, example, max_new_tokens=max_new_tokens, answer_style=answer_style)
    else:
        outputs = run_baseline(model, tokenizer, example, max_new_tokens=max_new_tokens, answer_style=answer_style)

    if device is not None:
        torch.cuda.synchronize(device)
    latency = time.time() - t0
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device is not None else 0.0
    )
    return outputs[0].strip(), latency, peak_mem_mb


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.retrieved_input) as f:
        data = json.load(f)
    records = data["records"] if isinstance(data, dict) else data

    if args.num_samples is not None:
        records = records[:args.num_samples]
    elif args.sample_pct is not None:
        n = max(1, int(len(records) * args.sample_pct))
        records = records[:n]

    num_samples = len(records)
    print(f"Loaded {num_samples} records from {args.retrieved_input}")
    print(f"Loading generator: pipeline={args.pipeline}")
    model, tokenizer = load_generator(args.pipeline, args.model_path)

    if args.top_ks:
        top_ks = tuple(int(x) for x in args.top_ks.split(",") if x.strip())
    else:
        top_ks = TOP_KS
    print(f"Evaluating top-K values: {top_ks}")

    totals = {
        k: {
            "em": 0.0,
            "f1": 0.0,
            "contain": 0.0,
            "rouge_l": 0.0,
            "em_aware": 0.0,
            "f1_aware": 0.0,
            "contain_aware": 0.0,
            "rouge_l_aware": 0.0,
            "refused": 0,
            "gold_found": 0,
            "refused_correct": 0,
            "latencies": [],
            "peak_mems": [],
            "skipped": 0,
        }
        for k in top_ks
    }

    output_records = []

    for record in tqdm(records, desc="Generating"):
        true_answer = record["answer"]
        query = record["prompt"]
        all_retrieved = record["retrieved_context"]

        gen_query = (
            HYPERNET_QUERY_PREFIX + query
            if args.pipeline == "doc2lora" and not args.no_query_prefix
            else query
        )

        out_record = {
            k: v
            for k, v in record.items()
            if k not in ("prediction", "gen_em", "gen_f1", "gen_contain", "latency", "mem")
        }

        for top_k in top_ks:
            passages = all_retrieved[:top_k]

            if args.context_mode == "per_chunk" and args.pipeline == "doc2lora":
                gen_context = list(passages)
            else:
                gen_context = "\n\n".join(passages)

            try:
                raw_pred, latency, peak_mem_mb = generate_answer(
                    args.pipeline,
                    model,
                    tokenizer,
                    gen_context,
                    gen_query,
                    args.max_new_tokens,
                    answer_style=args.answer_style,
                )
            except RuntimeError as e:
                if "aligned" in str(e).lower():
                    print(f"[WARN] Skipping record id={record.get('id')} top_k={top_k}: {e}")
                    totals[top_k]["skipped"] += 1

                    if gold_in_retrieved(true_answer, passages):
                        totals[top_k]["gold_found"] += 1

                    out_record[f"top{top_k}_prediction_raw"] = None
                    out_record[f"top{top_k}_prediction"] = None
                    out_record[f"top{top_k}_gen_em"] = None
                    out_record[f"top{top_k}_gen_f1"] = None
                    out_record[f"top{top_k}_gen_contain"] = None
                    out_record[f"top{top_k}_gen_rouge_l"] = None
                    out_record[f"top{top_k}_latency"] = None
                    out_record[f"top{top_k}_mem"] = None
                    continue
                raise

            if args.answer_style == "full":
                prediction = raw_pred
            else:
                prediction = extract_answer_span(raw_pred, true_answer)

            em = compute_em(prediction, true_answer)
            f1 = compute_f1(prediction, true_answer)
            contain = compute_contain(raw_pred, true_answer)
            rouge_l = compute_rouge_l(prediction, true_answer)

            pred_for_refusal_check = prediction if args.answer_style != "full" else raw_pred
            em_a, f1_a, rouge_l_a, contain_a, refused_correct, refused, gold_found = apply_refusal_credit(
                em,
                f1,
                rouge_l,
                contain,
                pred_for_refusal_check,
                true_answer,
                passages,
            )

            totals[top_k]["em"] += em
            totals[top_k]["f1"] += f1
            totals[top_k]["contain"] += contain
            totals[top_k]["rouge_l"] += rouge_l
            totals[top_k]["em_aware"] += em_a
            totals[top_k]["f1_aware"] += f1_a
            totals[top_k]["contain_aware"] += contain_a
            totals[top_k]["rouge_l_aware"] += rouge_l_a
            totals[top_k]["refused"] += int(refused)
            totals[top_k]["gold_found"] += int(gold_found)
            totals[top_k]["refused_correct"] += int(refused_correct)
            totals[top_k]["latencies"].append(latency)
            totals[top_k]["peak_mems"].append(peak_mem_mb)

            out_record[f"top{top_k}_prediction_raw"] = raw_pred
            out_record[f"top{top_k}_prediction"] = prediction
            out_record[f"top{top_k}_gen_em"] = em
            out_record[f"top{top_k}_gen_f1"] = f1
            out_record[f"top{top_k}_gen_contain"] = contain
            out_record[f"top{top_k}_gen_rouge_l"] = rouge_l
            out_record[f"top{top_k}_gen_em_aware"] = em_a
            out_record[f"top{top_k}_gen_f1_aware"] = f1_a
            out_record[f"top{top_k}_gen_contain_aware"] = contain_a
            out_record[f"top{top_k}_gen_rouge_l_aware"] = rouge_l_a
            out_record[f"top{top_k}_is_refusal"] = refused
            out_record[f"top{top_k}_gold_in_retrieved"] = gold_found
            out_record[f"top{top_k}_refused_correctly"] = refused_correct
            out_record[f"top{top_k}_latency"] = latency
            out_record[f"top{top_k}_mem"] = peak_mem_mb

        output_records.append(out_record)

    print(f"\n===== GENERATION SUMMARY ({args.pipeline}, answer_style={args.answer_style}) =====")
    summaries = {}

    for top_k in top_ks:
        t = totals[top_k]
        n_valid = num_samples - t["skipped"]

        if n_valid == 0:
            print(f"\nTop-{top_k}: all samples skipped")
            continue

        em = t["em"] / num_samples
        f1 = t["f1"] / num_samples
        contain = t["contain"] / num_samples
        rouge_l = t["rouge_l"] / num_samples
        em_a = t["em_aware"] / num_samples
        f1_a = t["f1_aware"] / num_samples
        contain_a = t["contain_aware"] / num_samples
        rouge_l_a = t["rouge_l_aware"] / num_samples

        avg_lat = sum(t["latencies"]) / len(t["latencies"]) if t["latencies"] else 0
        avg_mem = sum(t["peak_mems"]) / len(t["peak_mems"]) if t["peak_mems"] else 0
        max_mem = max(t["peak_mems"]) if t["peak_mems"] else 0

        print(f"\nTop-{top_k} passages:")
        print(f"EM: {em:.4f}   EM (aware): {em_a:.4f}")
        print(f"F1: {f1:.4f}   F1 (aware): {f1_a:.4f}")
        print(f"ROUGE-L: {rouge_l:.4f}   ROUGE-L (aware): {rouge_l_a:.4f}")
        print(f"Containment: {contain:.4f}   Contain (aware): {contain_a:.4f}")
        print(
            f"Refusal rate: {t['refused'] / num_samples:.4f}   "
            f"Retrieval-fail: {(num_samples - t['gold_found']) / num_samples:.4f}   "
            f"Correct refusals: {t['refused_correct'] / num_samples:.4f}"
        )
        print(f"Avg latency: {avg_lat:.4f}s")
        print(f"Avg peak mem: {avg_mem:.1f} MB")
        print(f"Max peak mem: {max_mem:.1f} MB")
        print(f"Skipped: {t['skipped']}")

        summaries[f"top{top_k}"] = {
            "answer_em": em,
            "answer_f1": f1,
            "answer_rouge_l": rouge_l,
            "answer_contain": contain,
            "answer_em_aware": em_a,
            "answer_f1_aware": f1_a,
            "answer_rouge_l_aware": rouge_l_a,
            "answer_contain_aware": contain_a,
            "refusal_rate": t["refused"] / num_samples,
            "retrieval_fail_rate": (num_samples - t["gold_found"]) / num_samples,
            "refused_correct_rate": t["refused_correct"] / num_samples,
            "avg_latency_sec": avg_lat,
            "avg_peak_mem_mb": avg_mem,
            "max_peak_mem_mb": max_mem,
            "num_skipped": t["skipped"],
            "num_samples": num_samples,
        }

    Path(args.gen_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.gen_output, "w") as f:
        json.dump(
            {
                "pipeline": args.pipeline,
                "answer_style": args.answer_style,
                "model_path": args.model_path,
                "retrieved_input": args.retrieved_input,
                "num_samples": num_samples,
                "summary": summaries,
                "records": output_records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved output to {args.gen_output}")


if __name__ == "__main__":
    main()