# Scoring helpers shared by the RAG eval scripts.
# Refusal credit: if gold isn't in the retrieved context and the model says so, treat that as a correct answer instead of a 0.

from __future__ import annotations
import re
import string
from collections import Counter
from typing import Tuple


_PUNCT = set(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_UNDERSCORE_GAP = re.compile(r"\s*_\s*")


def normalize_answer(s: str):
    if not s:
        return ""
    s = s.lower()
    # Some tokenizers split foo_bar into "foo _ bar"; rejoin first.
    s = _UNDERSCORE_GAP.sub("_", s)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def compute_em(prediction: str, ground_truth: str):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str):
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def compute_contain(prediction: str, ground_truth: str):
    g = normalize_answer(ground_truth)
    if not g:
        return 0.0
    return float(g in normalize_answer(prediction))



def compute_rouge_l(prediction: str, ground_truth: str):
    p = normalize_answer(prediction).split()
    g = normalize_answer(ground_truth).split()
    if not p or not g:
        return 0.0

    m, n = len(p), len(g)
    row = [0] * (n + 1)
    for i in range(1, m + 1):
        next = [0] * (n + 1)
        for j in range(1, n + 1):
            if p[i - 1] == g[j - 1]:
                next[j] = row[j - 1] + 1
            else:
                next[j] = max(row[j], next[j - 1])
        row = next

    lcs = row[n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)


# Must match the prompt in src/hypernetwork/inference.py.
REFUSAL_TRIGGER = "answer not in context"
_TRIGGER_NORM = normalize_answer(REFUSAL_TRIGGER)


def is_refusal(pred: str):
    if not pred:
        return False
    n = normalize_answer(pred)
    return n == _TRIGGER_NORM or n.startswith(_TRIGGER_NORM + " ")


# Refusal phrasings observed in pre-prompt-fix Qwen runs
_LEGACY_PREFIXES = tuple(normalize_answer(p) for p in [
    "answer not in context",
    "passages do not mention",
    "passages dont mention",
    "passages do not contain",
    "passages do not provide",
    "passages do not state",
    "passages do not specify",
    "passages do not indicate",
    "passages do not directly",
    "passage does not mention",
    "passage does not provide",
    "passage does not contain",
    "passage does not state",
    "passage does not specify",
    "provided passages do not",
    "provided context does not",
    "provided information does not",
    "context does not mention",
    "context does not contain",
    "context does not provide",
    "there is no information",
    "there is no mention",
    "there is no indication",
    "there are no mentions",
    "there are no passages",
    "theres no information",
    "no information in passage",
    "no information in passages",
    "no information in provided",
    "no information in given",
    "no information is provided",
    "no mention of",
    "it is not mentioned",
    "it is not stated",
    "it is not specified",
    "it is not clear",
    "it is not possible to",
    "cannot be determined",
    "cannot determine",
    "i cannot determine",
    "i cannot find",
    "i dont have enough",
    "i do not have enough",
    "i dont have information",
    "i do not have information",
    "i am sorry i dont",
    "im sorry i dont",
    "unfortunately passage",
    "unfortunately provided",
    "unfortunately passages",
    "not enough information",
    "not mentioned in passages",
    "not mentioned in passage",
    "not mentioned in provided",
    "not mentioned in given",
    "not mentioned in context",
    "not provided in passages",
    "not provided in passage",
    "not provided in context",
])

# just checks if the prediction is one of these 
def is_refusal_lenient(pred: str):
    if not pred:
        return False
    n = normalize_answer(pred)
    for pfx in _LEGACY_PREFIXES:
        if n == pfx or n.startswith(pfx + " "):
            return True
    return False

# checks if the gold answer is even found in the retrieved to confirm the model concern that the retirved context doesnt provide the actual answer
def gold_in_retrieved(gold: str, retrieved_context):
    if not gold or not retrieved_context:
        return False
    if isinstance(retrieved_context, (list, tuple)):
        joined = " ".join(str(c) for c in retrieved_context if c)
    else:
        joined = str(retrieved_context)

    gold_norm = normalize_answer(gold)
    ctx = normalize_answer(joined)
    if not gold_norm or not ctx:
        return False
    if gold_norm in ctx:
        return True

    gold_toks = gold_norm.split()

    # Long-form ASQA golds rarely match exactly so set a threahold of the number of matches
    if len(gold_toks) < 3:
        return False
    ctx_toks = set(ctx.split())
    hits = sum(1 for t in gold_toks if t in ctx_toks)
    return hits / len(gold_toks) >= 0.5


def apply_refusal_credit(em, f1, rouge_l, contain, pred, gold, retrieved_context, refusal_fn=is_refusal):
    refused = refusal_fn(pred)
    found = gold_in_retrieved(gold, retrieved_context)
    if refused and not found:
        return 1.0, 1.0, 1.0, 1.0, True, True, False
    return em, f1, rouge_l, contain, False, refused, found
