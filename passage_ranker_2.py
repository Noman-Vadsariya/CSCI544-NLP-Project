

# #!/usr/bin/env python3
# """
# passage_ranker_2.py

# Script Description:
#     This script loads the SPLADE model and uses it to
#     predict the highest-to-lowest ranked passage indices
#     for each example in the validation.json file.

# Accuracy:
#     In >84% of cases, the model's top 2 passage
#     indices overlap with the top-1 ground-truth passage
#     index.

# Dependencies:
#     - First, run download_data.py to get this file created:
#         hotpotqa_json/validation.json
#     - Pip install the below libs
#     - It may work on CPU or may require GPU...idk yet

# Run this script like:
#     python passage_ranker_2.py --input hotpotqa_json/validation.json

# Authored by:
#     Ayush Saha
# """

# import argparse
# import html
# import json
# import os
# import re
# from collections import defaultdict
# from typing import Any, Dict, List, Tuple

# import torch
# from tqdm import tqdm
# from transformers import AutoModelForMaskedLM, AutoTokenizer


# DEFAULT_INPUT = "hotpotqa_json/validation.json"
# DEFAULT_OUTPUT = "predictions.json"
# DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


# def load_json_flexible(path: str) -> List[Dict[str, Any]]:
#     """
#     Loads standard JSON arrays, single objects, JSONL-like files,
#     or concatenated JSON objects.
#     Returns a list of dict records.
#     """
#     with open(path, "r", encoding="utf-8") as f:
#         raw = f.read().strip()

#     if not raw:
#         return []

#     try:
#         parsed = json.loads(raw)
#         if isinstance(parsed, list):
#             return parsed
#         if isinstance(parsed, dict):
#             return [parsed]
#     except json.JSONDecodeError:
#         pass

#     records = []
#     lines = [line.strip() for line in raw.splitlines() if line.strip()]
#     jsonl_ok = True
#     for line in lines:
#         try:
#             obj = json.loads(line)
#             if isinstance(obj, dict):
#                 records.append(obj)
#             else:
#                 jsonl_ok = False
#                 break
#         except json.JSONDecodeError:
#             jsonl_ok = False
#             break
#     if jsonl_ok and records:
#         return records

#     decoder = json.JSONDecoder()
#     idx = 0
#     n = len(raw)
#     out = []
#     while idx < n:
#         while idx < n and raw[idx].isspace():
#             idx += 1
#         if idx >= n:
#             break
#         obj, next_idx = decoder.raw_decode(raw, idx)
#         if isinstance(obj, dict):
#             out.append(obj)
#         idx = next_idx
#     return out


# def normalize_text(text: str) -> str:
#     text = html.unescape(text or "")
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def _normalize_title_to_key(title: str) -> str:
#     return normalize_text(title).lower()


# def get_context_titles_and_sentences(record: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
#     context = record.get("context", {})
#     titles = context.get("title", []) or []
#     sentences = context.get("sentences", []) or []
#     return titles, sentences


# def get_supporting_facts(record: Dict[str, Any]) -> List[Tuple[str, int]]:
#     """
#     Returns list of (title, sentence_id) gold supporting facts.

#     Supports two common formats:
#     1) HotpotQA official:
#        supporting_facts = [["Title", 3], ["Title2", 1], ...]
#     2) Dict-like:
#        supporting_facts = {"title": [...], "sent_id": [...]}
#     """
#     sf = record.get("supporting_facts", [])

#     facts: List[Tuple[str, int]] = []

#     if isinstance(sf, dict):
#         titles = sf.get("title", []) or []
#         sent_ids = sf.get("sent_id", []) or []
#         for t, s in zip(titles, sent_ids):
#             try:
#                 facts.append((normalize_text(t), int(s)))
#             except Exception:
#                 continue
#         return facts

#     if isinstance(sf, list):
#         for item in sf:
#             if isinstance(item, (list, tuple)) and len(item) >= 2:
#                 try:
#                     facts.append((normalize_text(item[0]), int(item[1])))
#                 except Exception:
#                     continue
#         return facts

#     return facts


# def map_gold_to_context_indices(
#     record: Dict[str, Any]
# ) -> Tuple[List[int], List[Tuple[int, int]]]:
#     """
#     Returns:
#       gold_passage_indices: sorted unique passage indices
#       gold_sentence_indices: list of (passage_idx, sent_idx)
#     """
#     titles, _ = get_context_titles_and_sentences(record)
#     title_to_indices: Dict[str, List[int]] = defaultdict(list)
#     for i, title in enumerate(titles):
#         title_to_indices[_normalize_title_to_key(title)].append(i)

#     supporting_facts = get_supporting_facts(record)

#     gold_passage_indices: List[int] = []
#     gold_sentence_indices: List[Tuple[int, int]] = []

#     used_title_positions = set()

#     for gold_title, sent_id in supporting_facts:
#         key = _normalize_title_to_key(gold_title)
#         if key not in title_to_indices:
#             continue

#         chosen_passage_idx = None
#         for idx in title_to_indices[key]:
#             if idx not in used_title_positions:
#                 chosen_passage_idx = idx
#                 used_title_positions.add(idx)
#                 break
#         if chosen_passage_idx is None:
#             chosen_passage_idx = title_to_indices[key][0]

#         gold_passage_indices.append(chosen_passage_idx)
#         gold_sentence_indices.append((chosen_passage_idx, sent_id))

#     # Deduplicate passage indices while preserving order
#     seen = set()
#     deduped_passages = []
#     for idx in gold_passage_indices:
#         if idx not in seen:
#             seen.add(idx)
#             deduped_passages.append(idx)

#     return deduped_passages, gold_sentence_indices


# def build_sentence_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
#     """
#     Build a flat list of sentence candidates.

#     Each item:
#       {
#         "passage_idx": int,
#         "passage_title": str,
#         "sent_idx": int,
#         "text": str
#       }

#     Sentence text is encoded as: "TITLE. sentence"
#     """
#     titles, sentences = get_context_titles_and_sentences(record)
#     sentence_records: List[Dict[str, Any]] = []

#     for p_idx, title in enumerate(titles):
#         title = normalize_text(title)
#         sent_list = sentences[p_idx] if p_idx < len(sentences) and isinstance(sentences[p_idx], list) else []

#         if not sent_list:
#             # Fallback pseudo-sentence if a passage has no sentence list
#             sentence_records.append(
#                 {
#                     "passage_idx": p_idx,
#                     "passage_title": title,
#                     "sent_idx": 0,
#                     "text": title,
#                 }
#             )
#             continue

#         for s_idx, sent in enumerate(sent_list):
#             sent = normalize_text(sent)
#             text = f"{title}. {sent}" if title else sent
#             sentence_records.append(
#                 {
#                     "passage_idx": p_idx,
#                     "passage_title": title,
#                     "sent_idx": s_idx,
#                     "text": text,
#                 }
#             )

#     return sentence_records


# @torch.no_grad()
# def splade_encode(
#     texts: List[str],
#     tokenizer: AutoTokenizer,
#     model: AutoModelForMaskedLM,
#     device: torch.device,
#     max_length: int = 256,
#     batch_size: int = 16,
# ) -> List[torch.Tensor]:
#     """
#     Encode texts into sparse SPLADE vectors.
#     Returns a list of 1D CPU tensors, one per input text.
#     """
#     model.eval()
#     outputs: List[torch.Tensor] = []

#     for start in range(0, len(texts), batch_size):
#         batch_texts = texts[start:start + batch_size]
#         enc = tokenizer(
#             batch_texts,
#             padding=True,
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt",
#         ).to(device)

#         out = model(**enc)
#         logits = out.logits  # [B, T, V]

#         activations = torch.log1p(torch.relu(logits))
#         sparse_vec = activations.max(dim=1).values  # [B, V]

#         outputs.extend([vec.detach().cpu() for vec in sparse_vec])

#     return outputs


# def score_query_against_vectors(
#     query_vec: torch.Tensor,
#     vecs: List[torch.Tensor],
# ) -> List[float]:
#     scores = []
#     q = query_vec.to(dtype=torch.float32)
#     for v in vecs:
#         scores.append(float(torch.dot(q, v.to(dtype=torch.float32)).item()))
#     return scores


# def rank_example(
#     question: str,
#     record: Dict[str, Any],
#     tokenizer: AutoTokenizer,
#     model: AutoModelForMaskedLM,
#     device: torch.device,
#     max_length: int,
#     batch_size: int,
#     top_k_passages: int = 2,
#     top_k_sentences: int = 2,
# ) -> Dict[str, Any]:
#     """
#     Rank passages and sentence evidence for one record.
#     """
#     sentence_records = build_sentence_records(record)
#     if not sentence_records:
#         return {
#             "question": question,
#             "predicted_passage_indices": [],
#             "predicted_top_passages": [],
#         }

#     sentence_texts = [r["text"] for r in sentence_records]

#     # Encode question and candidate sentences
#     encoded = splade_encode(
#         [question] + sentence_texts,
#         tokenizer=tokenizer,
#         model=model,
#         device=device,
#         max_length=max_length,
#         batch_size=batch_size,
#     )

#     query_vec = encoded[0]
#     sentence_vecs = encoded[1:]

#     sentence_scores = score_query_against_vectors(query_vec, sentence_vecs)

#     # Group sentence scores by passage
#     grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
#     passage_best_score: Dict[int, float] = {}

#     for i, rec in enumerate(sentence_records):
#         p_idx = rec["passage_idx"]
#         s_idx = rec["sent_idx"]
#         score = sentence_scores[i]

#         grouped[p_idx].append(
#             {
#                 "sent_idx": s_idx,
#                 "score": score,
#                 "text": rec["text"],
#             }
#         )

#     for p_idx, items in grouped.items():
#         passage_best_score[p_idx] = max(x["score"] for x in items)

#     predicted_passage_indices = sorted(
#         passage_best_score.keys(),
#         key=lambda idx: passage_best_score[idx],
#         reverse=True,
#     )

#     predicted_top_passages = []
#     for p_idx in predicted_passage_indices[:top_k_passages]:
#         top_sentences = sorted(
#             grouped[p_idx],
#             key=lambda x: x["score"],
#             reverse=True,
#         )[:top_k_sentences]

#         predicted_top_passages.append(
#             {
#                 "passage_idx": p_idx,
#                 "passage_score": passage_best_score[p_idx],
#                 "top_sentences": top_sentences,
#             }
#         )

#     return {
#         "question": question,
#         "predicted_passage_indices": predicted_passage_indices,
#         "predicted_top_passages": predicted_top_passages,
#     }


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--input",
#         type=str,
#         default=DEFAULT_INPUT,
#         help="Path to HotpotQA validation JSON file",
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default=DEFAULT_OUTPUT,
#         help="Path to save prediction JSON",
#     )
#     parser.add_argument(
#         "--model",
#         type=str,
#         default=DEFAULT_MODEL,
#         help="Hugging Face SPLADE model name",
#     )
#     parser.add_argument(
#         "--max_length",
#         type=int,
#         default=256,
#         help="Tokenizer max length",
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=16,
#         help="Batch size for SPLADE encoding",
#     )
#     parser.add_argument(
#         "--max_examples",
#         type=int,
#         default=-1,
#         help="Optional limit on number of examples to process (-1 means all)",
#     )
#     parser.add_argument(
#         "--device",
#         type=str,
#         default="cuda" if torch.cuda.is_available() else "cpu",
#         help="Device to run on",
#     )
#     args = parser.parse_args()

#     if not os.path.exists(args.input):
#         raise FileNotFoundError(f"Input file not found: {args.input}")

#     print(f"Loading data: {args.input}")
#     records = load_json_flexible(args.input)
#     if args.max_examples > 0:
#         records = records[:args.max_examples]

#     print(f"Loaded {len(records)} examples")
#     print(f"Loading model: {args.model}")
#     device = torch.device(args.device)

#     tokenizer = AutoTokenizer.from_pretrained(args.model)
#     model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)

#     all_predictions: List[Dict[str, Any]] = []

#     for idx, record in enumerate(tqdm(records, desc="Processing")):
#         question = normalize_text(record.get("question", ""))
#         if not question:
#             continue

#         gold_passage_indices, gold_sentence_indices = map_gold_to_context_indices(record)

#         result = rank_example(
#             question=question,
#             record=record,
#             tokenizer=tokenizer,
#             model=model,
#             device=device,
#             max_length=args.max_length,
#             batch_size=args.batch_size,
#             top_k_passages=2,
#             top_k_sentences=2,
#         )

#         # Continuous printout
#         print("\n" + "=" * 110)
#         print(f"example idx: {idx}")
#         print(f"query: {question}")
#         print(f"gold passage indices: {gold_passage_indices}")
#         print(f"gold sentence indices: {gold_sentence_indices}")
#         print(f"predicted passage indices (highest -> lowest): {result['predicted_passage_indices']}")

#         print("top 2 predicted passages with top sentence indices (highest -> lowest):")
#         for p_rank, passage_info in enumerate(result["predicted_top_passages"], start=1):
#             passage_idx = passage_info["passage_idx"]
#             top_sent_indices = [x["sent_idx"] for x in passage_info["top_sentences"]]
#             print(
#                 f"  passage rank {p_rank}: passage_idx={passage_idx}, "
#                 f"passage_score={passage_info['passage_score']:.4f}, "
#                 f"sentence_indices={top_sent_indices}"
#             )

#         all_predictions.append(
#             {
#                 "idx": idx,
#                 "question": question,
#                 "gold_passage_indices": gold_passage_indices,
#                 "gold_sentence_indices": [
#                     {"passage_idx": p_idx, "sent_idx": s_idx}
#                     for p_idx, s_idx in gold_sentence_indices
#                 ],
#                 "predicted_passage_indices": result["predicted_passage_indices"],
#                 "predicted_top_passages": result["predicted_top_passages"],
#             }
#         )

#     # Save final JSON
#     with open(args.output, "w", encoding="utf-8") as f:
#         json.dump(all_predictions, f, indent=2, ensure_ascii=False)

#     print("\n" + "=" * 110)
#     print(f"Saved predictions to: {args.output}")


# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
passage_ranker_2.py

Script Description:
    This script loads the SPLADE model and uses it to
    predict ranked passage indices and ranked sentence indices
    for each example in the validation.json file.

Run this script like:
    python passage_ranker_2.py --input hotpotqa_json/validation.json
"""

import argparse
import html
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


DEFAULT_INPUT = "validation_subset.json"
DEFAULT_OUTPUT = "predictions.json"
DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


def load_json_flexible(path: str) -> List[Dict[str, Any]]:
    """
    Loads standard JSON arrays, single objects, JSONL-like files,
    or concatenated JSON objects.
    Returns a list of dict records.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    records = []
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    jsonl_ok = True
    for line in lines:
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
            else:
                jsonl_ok = False
                break
        except json.JSONDecodeError:
            jsonl_ok = False
            break
    if jsonl_ok and records:
        return records

    decoder = json.JSONDecoder()
    idx = 0
    n = len(raw)
    out = []
    while idx < n:
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, next_idx = decoder.raw_decode(raw, idx)
        if isinstance(obj, dict):
            out.append(obj)
        idx = next_idx
    return out


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_title_to_key(title: str) -> str:
    return normalize_text(title).lower()


def get_context_titles_and_sentences(record: Dict[str, Any]) -> Tuple[List[str], List[List[str]]]:
    context = record.get("context", {})
    titles = context.get("title", []) or []
    sentences = context.get("sentences", []) or []
    return titles, sentences


def get_supporting_facts(record: Dict[str, Any]) -> List[Tuple[str, int]]:
    """
    Returns list of (title, sentence_id) gold supporting facts.
    """
    sf = record.get("supporting_facts", [])

    facts: List[Tuple[str, int]] = []

    if isinstance(sf, dict):
        titles = sf.get("title", []) or []
        sent_ids = sf.get("sent_id", []) or []
        for t, s in zip(titles, sent_ids):
            try:
                facts.append((normalize_text(t), int(s)))
            except Exception:
                continue
        return facts

    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    facts.append((normalize_text(item[0]), int(item[1])))
                except Exception:
                    continue
        return facts

    return facts


def map_gold_to_context_indices(
    record: Dict[str, Any]
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Returns:
      gold_passage_indices: sorted unique passage indices
      gold_sentence_indices: list of (passage_idx, sent_idx)
    """
    titles, _ = get_context_titles_and_sentences(record)
    title_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, title in enumerate(titles):
        title_to_indices[_normalize_title_to_key(title)].append(i)

    supporting_facts = get_supporting_facts(record)

    gold_passage_indices: List[int] = []
    gold_sentence_indices: List[Tuple[int, int]] = []

    used_title_positions = set()

    for gold_title, sent_id in supporting_facts:
        key = _normalize_title_to_key(gold_title)
        if key not in title_to_indices:
            continue

        chosen_passage_idx = None
        for idx in title_to_indices[key]:
            if idx not in used_title_positions:
                chosen_passage_idx = idx
                used_title_positions.add(idx)
                break
        if chosen_passage_idx is None:
            chosen_passage_idx = title_to_indices[key][0]

        gold_passage_indices.append(chosen_passage_idx)
        gold_sentence_indices.append((chosen_passage_idx, sent_id))

    seen = set()
    deduped_passages = []
    for idx in gold_passage_indices:
        if idx not in seen:
            seen.add(idx)
            deduped_passages.append(idx)

    return deduped_passages, gold_sentence_indices


def build_sentence_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a flat list of sentence candidates.
    """
    titles, sentences = get_context_titles_and_sentences(record)
    sentence_records: List[Dict[str, Any]] = []

    for p_idx, title in enumerate(titles):
        title = normalize_text(title)
        sent_list = sentences[p_idx] if p_idx < len(sentences) and isinstance(sentences[p_idx], list) else []

        if not sent_list:
            sentence_records.append(
                {
                    "passage_idx": p_idx,
                    "passage_title": title,
                    "sent_idx": 0,
                    "text": title,
                }
            )
            continue

        for s_idx, sent in enumerate(sent_list):
            sent = normalize_text(sent)
            text = f"{title}. {sent}" if title else sent
            sentence_records.append(
                {
                    "passage_idx": p_idx,
                    "passage_title": title,
                    "sent_idx": s_idx,
                    "text": text,
                }
            )

    return sentence_records


@torch.no_grad()
def splade_encode(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 16,
) -> List[torch.Tensor]:
    """
    Encode texts into sparse SPLADE vectors.
    Returns a list of 1D CPU tensors, one per input text.
    """
    model.eval()
    outputs: List[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
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
        sparse_vec = activations.max(dim=1).values

        outputs.extend([vec.detach().cpu() for vec in sparse_vec])

    return outputs


def score_query_against_vectors(
    query_vec: torch.Tensor,
    vecs: List[torch.Tensor],
) -> List[float]:
    scores = []
    q = query_vec.to(dtype=torch.float32)
    for v in vecs:
        scores.append(float(torch.dot(q, v.to(dtype=torch.float32)).item()))
    return scores


def group_gold_sentence_indices_by_passage(
    gold_sentence_indices: List[Tuple[int, int]]
) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for p_idx, s_idx in gold_sentence_indices:
        grouped[p_idx].append(s_idx)
    return {p_idx: sorted(set(sents)) for p_idx, sents in grouped.items()}


def format_index_list(values: List[int]) -> str:
    return "[]" if not values else "[" + ", ".join(str(v) for v in values) + "]"


def print_example_table(
    example_idx: int,
    question: str,
    gold_passage_indices: List[int],
    gold_sentence_indices: List[Tuple[int, int]],
    predicted_top_passages: List[Dict[str, Any]],
    top_k_passages: int = 5,
) -> None:
    gold_by_passage = group_gold_sentence_indices_by_passage(gold_sentence_indices)

    rows: List[List[str]] = []
    for rank in range(top_k_passages):
        pred_entry = predicted_top_passages[rank] if rank < len(predicted_top_passages) else None

        if pred_entry is not None:
            pred_p_idx = str(pred_entry["passage_idx"])
            pred_s_idxs = [x["sent_idx"] for x in pred_entry["top_sentences"]]
            pred_s_idxs_str = format_index_list(pred_s_idxs)
        else:
            pred_p_idx = ""
            pred_s_idxs_str = "[]"

        if rank < len(gold_passage_indices):
            gold_p_idx = str(gold_passage_indices[rank])
            gold_s_idxs = gold_by_passage.get(gold_passage_indices[rank], [])
            gold_s_idxs_str = format_index_list(gold_s_idxs)
        else:
            gold_p_idx = ""
            gold_s_idxs_str = "[]"

        rows.append([
            str(rank + 1),
            pred_p_idx,
            pred_s_idxs_str,
            gold_p_idx,
            gold_s_idxs_str,
        ])

    headers = [
        "rank",
        "predicted psg index",
        "predicted sentence indices",
        "grndtruth psg index",
        "grnd truth sentence indices",
    ]

    widths = []
    for col_idx, header in enumerate(headers):
        max_row_len = max((len(row[col_idx]) for row in rows), default=0)
        widths.append(max(len(header), max_row_len))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(row[i].ljust(widths[i]) for i in range(len(headers)))

    print("\n" + "=" * 110)
    print(f"example idx: {example_idx}")
    print(f"query: {question}")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def rank_example(
    question: str,
    record: Dict[str, Any],
    tokenizer: AutoTokenizer,
    model: AutoModelForMaskedLM,
    device: torch.device,
    max_length: int,
    batch_size: int,
    top_k_passages: int = 5,
    top_k_sentences: int = 2,
) -> Dict[str, Any]:
    """
    Rank passages and sentence evidence for one record.
    """
    sentence_records = build_sentence_records(record)
    if not sentence_records:
        return {
            "question": question,
            "predicted_passage_indices": [],
            "predicted_top_passages": [],
        }

    sentence_texts = [r["text"] for r in sentence_records]

    encoded = splade_encode(
        [question] + sentence_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    query_vec = encoded[0]
    sentence_vecs = encoded[1:]

    sentence_scores = score_query_against_vectors(query_vec, sentence_vecs)

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    passage_best_score: Dict[int, float] = {}

    for i, rec in enumerate(sentence_records):
        p_idx = rec["passage_idx"]
        s_idx = rec["sent_idx"]
        score = sentence_scores[i]

        grouped[p_idx].append(
            {
                "sent_idx": s_idx,
                "score": score,
                "text": rec["text"],
            }
        )

    for p_idx, items in grouped.items():
        passage_best_score[p_idx] = max(x["score"] for x in items)

    predicted_passage_indices = sorted(
        passage_best_score.keys(),
        key=lambda idx: passage_best_score[idx],
        reverse=True,
    )

    predicted_top_passages = []
    for p_idx in predicted_passage_indices[:top_k_passages]:
        top_sentences = sorted(
            grouped[p_idx],
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k_sentences]

        predicted_top_passages.append(
            {
                "passage_idx": p_idx,
                "passage_score": passage_best_score[p_idx],
                "top_sentences": top_sentences,
            }
        )

    return {
        "question": question,
        "predicted_passage_indices": predicted_passage_indices,
        "predicted_top_passages": predicted_top_passages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Path to HotpotQA validation JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to save prediction JSON",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face SPLADE model name",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Tokenizer max length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for SPLADE encoding",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=-1,
        help="Optional limit on number of examples to process (-1 means all)",
    )
    parser.add_argument(
        "--top_k_passages",
        type=int,
        default=5,
        help="Number of top predicted passages to print per example",
    )
    parser.add_argument(
        "--top_k_sentences",
        type=int,
        default=2,
        help="Number of top predicted sentences to print per predicted passage",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading data: {args.input}")
    records = load_json_flexible(args.input)
    if args.max_examples > 0:
        records = records[:args.max_examples]

    print(f"Loaded {len(records)} examples")
    print(f"Loading model: {args.model}")
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)

    all_predictions: List[Dict[str, Any]] = []

    for idx, record in enumerate(tqdm(records, desc="Processing")):
        question = normalize_text(record.get("question", ""))
        if not question:
            continue

        gold_passage_indices, gold_sentence_indices = map_gold_to_context_indices(record)

        result = rank_example(
            question=question,
            record=record,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            top_k_passages=args.top_k_passages,
            top_k_sentences=args.top_k_sentences,
        )

        print_example_table(
            example_idx=idx,
            question=question,
            gold_passage_indices=gold_passage_indices,
            gold_sentence_indices=gold_sentence_indices,
            predicted_top_passages=result["predicted_top_passages"],
            top_k_passages=args.top_k_passages,
        )

        all_predictions.append(
            {
                "idx": idx,
                "question": question,
                "gold_passage_indices": gold_passage_indices,
                "gold_sentence_indices": [
                    {"passage_idx": p_idx, "sent_idx": s_idx}
                    for p_idx, s_idx in gold_sentence_indices
                ],
                "predicted_passage_indices": result["predicted_passage_indices"],
                "predicted_top_passages": result["predicted_top_passages"],
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 110)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
