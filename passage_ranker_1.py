
#!/usr/bin/env python3
"""
passage_ranker_1.py

Script Description:
    This script loads the e5 model and uses it to
    predict the highest-to-lowest ranked passage indices
    for each example in the validation.json file.

Accuracy:
    In >84% of cases, the model's top 2 passage
    indices overlap with the top-1 ground-truth passage
    index.

Dependencies:
    - First, run download_data.py to get this file created:
        hotpotqa_json/validation.json
    - Pip install the below libs
    - It may work on CPU or may require GPU...idk yet

Run this script like:
    python passage_ranker_1.py --data_path hotpotqa_json/validation.json

Authored by:
    Ayush Saha
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_examples(path: str) -> List[Dict[str, Any]]:
    """
    Load either:
      - a JSON array file
      - a JSONL file (one object per line)
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Try JSON array/object first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Sometimes people store {"data": [...]}.
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL
    examples = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            examples.append(json.loads(line))
    return examples


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pool token embeddings using the attention mask.
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    prefix: str,
    batch_size: int = 32,
    max_length: int = 512,
) -> torch.Tensor:
    """
    Encode a list of texts into normalized embeddings.
    """
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = [prefix + t for t in texts[start:start + batch_size]]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = model(**batch)
        emb = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def extract_passages(example: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Return titles and passage texts.
    In HotpotQA, each title corresponds to one passage; the passage text is the
    concatenation of its sentences.
    """
    context = example["context"]
    titles = context["title"]
    sentences = context["sentences"]

    passages = []
    for sent_list in sentences:
        if isinstance(sent_list, list):
            passage = " ".join(s.strip() for s in sent_list if s and s.strip())
        else:
            passage = str(sent_list).strip()
        passages.append(passage)

    return titles, passages


def gold_passage_indices(example: Dict[str, Any], titles: List[str]) -> List[int]:
    """
    Convert supporting_facts titles into unique passage indices.
    """
    sf = example.get("supporting_facts", {})
    gold_titles = sf.get("title", [])
    gold_set = set(gold_titles)

    indices = []
    for i, title in enumerate(titles):
        if title in gold_set:
            indices.append(i)

    return indices


def rank_passages_for_example(
    example: Dict[str, Any],
    tokenizer,
    model,
    device: torch.device,
    top_k: int = 5,
) -> Tuple[List[int], List[int], List[float]]:
    """
    Rank all passages in the example against the question.
    Returns:
        retrieved_indices, gold_indices, retrieved_scores
    """
    question = example["question"]
    titles, passages = extract_passages(example)
    gold_indices = gold_passage_indices(example, titles)

    if len(passages) == 0:
        return [], gold_indices, []

    q_emb = encode_texts([question], tokenizer, model, device, prefix="query: ", batch_size=1)
    p_emb = encode_texts(passages, tokenizer, model, device, prefix="passage: ", batch_size=32)

    # cosine similarity because embeddings are normalized
    scores = torch.matmul(q_emb, p_emb.T).squeeze(0)  # shape: [num_passages]

    k = min(top_k, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=k, largest=True, sorted=True)

    return top_indices.tolist(), gold_indices, top_scores.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HotpotQA validation JSON or JSONL file.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="intfloat/e5-base-v2",
        help="HF model name for E5.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top passages to print per example.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=0,
        help="Optional limit for debugging; 0 means all examples.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    print(f"Loading data: {args.data_path}")
    examples = load_examples(args.data_path)
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]

    print(f"Loaded {len(examples)} examples\n")

    total = 0
    hit_at_1 = 0
    hit_at_k = 0
    any_gold = 0

    for ex in examples:
        example_id = ex.get("id", "unknown")
        retrieved, gold, _ = rank_passages_for_example(
            ex,
            tokenizer=tokenizer,
            model=model,
            device=device,
            top_k=args.top_k,
        )

        print(f"{example_id} | {retrieved} | {gold}")

        total += 1
        if gold:
            any_gold += 1
            if retrieved:
                if retrieved[0] in gold:
                    hit_at_1 += 1
                if any(idx in gold for idx in retrieved):
                    hit_at_k += 1

    if any_gold > 0:
        print("\nSummary")
        print(f"Examples: {total}")
        print(f"Hit@1: {hit_at_1 / any_gold:.4f}")
        print(f"Hit@{args.top_k}: {hit_at_k / any_gold:.4f}")


if __name__ == "__main__":
    main()

