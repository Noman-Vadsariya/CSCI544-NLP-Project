import argparse
import os
import json
import time
import torch
import torch.nn.functional as F
import re
import string

from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from collections import Counter

from src.hypernetwork.inference import (
    load_baseline,
    load_hypernet,
    run_baseline,
    run_hypernet,
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

DATA_PATH = "./data/raw_datasets/hotpotQA_compact/test/ds.parquet"

# Backbone used for both dense retrieval and ColBERT-style token embeddings
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Chunking
CHUNK_SIZE = 480
CHUNK_OVERLAP = 80

# Pinecone / dense retrieval
USE_PINECONE = True
PINECONE_INDEX_NAME = "rag-index-v2"
PINECONE_NAMESPACE = "default"

# Candidate generation
BM25_CANDIDATE_K = 30
DENSE_CANDIDATE_K = 30

# Final rerank cutoff
RERANK_TOP_K = 10

# Late-interaction encoder limits
# Keep these aligned with your chunking for the best retrieval signal.
MAX_QUERY_LEN = 64
MAX_PASSAGE_LEN = 480

# Evaluation
NUM_SAMPLES = 2

UPSERT_PINECONE = False  # set to False to skip Pinecone upload (use existing index)

# -------------------------------------------------------------------
# CLI ARGS (parsed at module level so setup can use --embedding_model)
# -------------------------------------------------------------------

def parse_gen_args():
    parser = argparse.ArgumentParser()
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
        "--embedding_model",
        type=str,
        default=None,
        help=f"Local path or HF repo for the retrieval/ColBERT backbone. Defaults to {BGE_MODEL_NAME}.",
    )
    parser.add_argument(
        "--context_mode",
        choices=["joined", "per_chunk"],
        default="joined",
        help="joined: concat top-K into one context. per_chunk: internalize each retrieved chunk separately (doc2lora only).",
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
        help="Number of eval samples. Defaults to all available.",
    )
    parser.add_argument(
        "--gen_output",
        type=str,
        default="./data/retrieved/hotpotQA_gen_outputs.json",
    )
    parser.add_argument(
        "--retrieved_output",
        type=str,
        default="./data/retrieved/hotpotQA_colbert_retrieved.json",
        help="Path to save retrieved contexts JSON.",
    )
    return parser.parse_args()


args = parse_gen_args()
embedding_model_name_or_path = args.embedding_model or BGE_MODEL_NAME
print(f"Embedding model: {embedding_model_name_or_path}")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

ds = load_dataset("parquet", data_files=DATA_PATH)["train"]

print("num rows:", len(ds))
print("columns:", ds.column_names)

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]
golden_contexts = ds["gold_context"]

print("Total QA pairs:", len(queries))
print("Total contexts:", len(contexts))

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dense embedding model for Pinecone retrieval
dense_model = SentenceTransformer(embedding_model_name_or_path, device=device)

# Raw transformer backbone for token-level late interaction scoring
# Important: this is NOT SentenceTransformer pooling; we use last_hidden_state.
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name_or_path)
encoder = AutoModel.from_pretrained(embedding_model_name_or_path).to(device)
encoder.eval()

# -------------------------------------------------------------------
# CHUNKING
# -------------------------------------------------------------------

def chunk_text(text, chunk_size=480, overlap=80):
    """
    Split a document into overlapping token chunks.
    """
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        if chunk.strip():
            chunks.append(chunk)

    return chunks

chunked_contexts = []
chunk_id_to_original = []

for i, c in enumerate(contexts):
    chunks = chunk_text(c, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunked_contexts.extend(chunks)
    chunk_id_to_original.extend([i] * len(chunks))

print("Total chunked contexts:", len(chunked_contexts))

# Stable mapping: pid -> text
pid_to_text = {pid: text for pid, text in enumerate(chunked_contexts)}

# -------------------------------------------------------------------
# BM25 INDEX
# -------------------------------------------------------------------

tokenized_corpus = [c.lower().split() for c in chunked_contexts]
bm25 = BM25Okapi(tokenized_corpus)

# -------------------------------------------------------------------
# PINECONE SETUP
# -------------------------------------------------------------------

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
existing_indexes = [idx.name for idx in pc.list_indexes()]

if USE_PINECONE:
    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1024,  # BAAI/bge-large-en-v1.5 hidden size
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    if UPSERT_PINECONE:
        # Build dense embeddings for all chunks
        context_inputs = ["passage: " + c for c in chunked_contexts]
        context_embeddings = dense_model.encode(
            context_inputs,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device,
        ).astype("float32")

        print("Uploading embeddings to Pinecone...")

        vectors = [
            (
                str(i),  # use chunk pid as vector id
                context_embeddings[i].tolist(),
                {
                    "text": chunked_contexts[i],
                    "original_id": chunk_id_to_original[i],
                    "pid": i,
                },
            )
            for i in range(len(chunked_contexts))
        ]

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            pinecone_index.upsert(
                vectors=vectors[i:i + batch_size],
                namespace=PINECONE_NAMESPACE,
            )

        print("Pinecone upload complete!")
else:
    pinecone_index = None

# -------------------------------------------------------------------
# COLBERT-STYLE LATE INTERACTION SCORER
# -------------------------------------------------------------------

class ColBERTStyleScorer:
    """
    Lightweight ColBERT-style reranker.

    Score(q, p) = sum_i max_j <q_i, p_j>

    Where:
      - q_i are query token embeddings
      - p_j are passage token embeddings
      - embeddings are L2-normalized
      - special tokens and padding are removed

    This uses the same transformer backbone for query and passage.
    """

    def __init__(
        self,
        tokenizer,
        encoder,
        passage_texts,
        device="cpu",
        max_query_len=64,
        max_passage_len=480,
    ):
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.passage_texts = passage_texts
        self.device = device
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len

        # Cache passage token embeddings by pid.
        # Each value is a CPU tensor of shape [num_tokens, hidden_dim].
        self.passage_cache = {}

    def _prefix_texts(self, texts, is_query):
        prefix = "query: " if is_query else "passage: "
        return [prefix + t for t in texts]

    @torch.inference_mode()
    def _encode_batch(self, texts, is_query):
        """
        Encode a batch of texts into token-level embeddings.

        Returns a list of tensors, one per input text.
        Each tensor has shape [seq_len, hidden_dim].
        """
        if not texts:
            return []

        max_len = self.max_query_len if is_query else self.max_passage_len
        prefixed_texts = self._prefix_texts(texts, is_query=is_query)

        inputs = self.tokenizer(
            prefixed_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding=True,
        ).to(self.device)

        outputs = self.encoder(**inputs)
        hidden = F.normalize(outputs.last_hidden_state, p=2, dim=-1)

        seqs = []
        attention_mask = inputs["attention_mask"]

        for i in range(len(prefixed_texts)):
            # Valid token span includes [CLS] ... [SEP] with padding after SEP.
            valid_len = int(attention_mask[i].sum().item())

            # Remove [CLS] at position 0 and [SEP] at position valid_len - 1.
            # This leaves only real text tokens.
            token_emb = hidden[i, 1:valid_len - 1, :]

            # Fallback for very short inputs, though this should rarely happen.
            if token_emb.shape[0] == 0:
                token_emb = hidden[i, :1, :]

            seqs.append(token_emb.detach().cpu())

        return seqs

    def encode_query(self, query):
        """
        Encode one query into token embeddings on the current device.
        """
        emb = self._encode_batch([query], is_query=True)[0]
        return emb.to(self.device)

    def ensure_passage_cache(self, pids):
        """
        Encode and cache any missing candidate passages.
        """
        missing_pids = [pid for pid in pids if pid not in self.passage_cache]
        if not missing_pids:
            return

        missing_texts = [self.passage_texts[pid] for pid in missing_pids]
        missing_embs = self._encode_batch(missing_texts, is_query=False)

        for pid, emb in zip(missing_pids, missing_embs):
            self.passage_cache[pid] = emb

    def score_query_passage(self, query_emb, passage_emb):
        """
        Compute the exact ColBERT-style MaxSim score.
        query_emb:  [q_len, dim]
        passage_emb:[p_len, dim]
        """
        passage_emb = passage_emb.to(self.device)

        # Similarity matrix: [q_len, p_len]
        sim = query_emb @ passage_emb.T

        # Max over passage tokens for each query token: [q_len]
        max_per_query_token = sim.max(dim=1).values

        # Sum over query tokens => scalar score
        return float(max_per_query_token.sum().item())

    @torch.inference_mode()
    def rerank(self, query, candidate_pids, top_k=10):
        """
        Rerank a candidate list of passage ids.
        """
        # Deduplicate while preserving order
        candidate_pids = list(dict.fromkeys(int(pid) for pid in candidate_pids))

        if not candidate_pids:
            return []

        self.ensure_passage_cache(candidate_pids)
        query_emb = self.encode_query(query)

        scored = []
        for pid in candidate_pids:
            score = self.score_query_passage(query_emb, self.passage_cache[pid])
            scored.append((pid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# Initialize reranker
colbert_scorer = ColBERTStyleScorer(
    tokenizer=tokenizer,
    encoder=encoder,
    passage_texts=pid_to_text,
    device=device,
    max_query_len=MAX_QUERY_LEN,
    max_passage_len=MAX_PASSAGE_LEN,
)

# -------------------------------------------------------------------
# RETRIEVAL HELPERS
# -------------------------------------------------------------------

def retrieve_bm25_pids(query, top_k=30):
    """
    BM25 candidate generation, returning passage IDs.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    return top_indices

def retrieve_dense_pids(query, top_k=30):
    """
    Dense candidate generation via Pinecone, returning passage IDs.
    """
    if not USE_PINECONE or pinecone_index is None:
        return []

    query_embedding = dense_model.encode(
        ["query: " + query + " passage:"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )[0].astype("float32").tolist()

    results = pinecone_index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )

    pids = []
    for match in results["matches"]:
        try:
            pids.append(int(match["id"]))
        except Exception:
            md = match.get("metadata", {})
            if "pid" in md:
                pids.append(int(md["pid"]))

    return pids

def combine_candidate_pids(*pid_lists):
    """
    Merge lists of pids while preserving order and removing duplicates.
    """
    merged = []
    seen = set()

    for pid_list in pid_lists:
        for pid in pid_list:
            pid = int(pid)
            if pid not in seen:
                seen.add(pid)
                merged.append(pid)

    return merged

def retrieve_with_colbert_rerank(query, top_k=10, candidate_k=50):
    """
    1) Generate candidates from BM25 + dense retrieval
    2) Rerank candidates with ColBERT-style MaxSim

    Returns:
        retrieved_texts: list[str]
        scored_pids: list[tuple[int, float]]
    """
    bm25_pids = retrieve_bm25_pids(query, top_k=candidate_k)
    dense_pids = retrieve_dense_pids(query, top_k=candidate_k)

    candidate_pids = combine_candidate_pids(bm25_pids, dense_pids)

    scored = colbert_scorer.rerank(
        query=query,
        candidate_pids=candidate_pids,
        top_k=top_k,
    )

    retrieved_texts = [pid_to_text[pid] for pid, _ in scored]
    return retrieved_texts, scored


# -------------------------------------------------------------------
#  EVAL HELPERS  
# -------------------------------------------------------------------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_em(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_containment(prediction, ground_truth):
    """Gold answer appears as a substring of the (normalized) prediction."""
    gold_norm = normalize_answer(ground_truth)
    if not gold_norm:
        return 0.0
    return float(gold_norm in normalize_answer(prediction))


def extract_answer_span(text, gold):
    """Post-process free-form LLM output into a short answer span."""
    if not text:
        return ""

    gold_norm = normalize_answer(gold)

    # Yes/no shortcut: look at the first word of the raw prediction.
    first_word = re.split(r"[\s,.!?:;]+", text.strip().lower(), maxsplit=1)[0]
    if gold_norm in {"yes", "no"} and first_word in {"yes", "no"}:
        return first_word

    # Strip common markdown.
    t = text.strip()
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = t.replace("**", "").replace("*", "").replace("`", "")

    # Take the first non-empty line and drop list/heading markers.
    for line in t.split("\n"):
        line = re.sub(r"^[\-\*#\d\.\)]+\s*", "", line.strip())
        if line:
            t = line
            break

    # Take the first sentence.
    m = re.match(r"^(.+?[.!?])(?:\s|$)", t)
    if m:
        t = m.group(1).rstrip(".!?")

    return t.strip()


HYPERNET_QUERY_PREFIX = (
    "Answer the question in as few words as possible. "
    "Only output the answer itself, no explanation or extra text.\n\n"
)


# -------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------

def load_generator(pipeline: str, model_path: str | None):
    if pipeline == "doc2lora":
        return load_hypernet(model_path) if model_path else load_hypernet()
    if pipeline == "regular":
        return load_baseline(model_path) if model_path else load_baseline()
    raise ValueError(f"Unknown pipeline: {pipeline}")


def generate_answer(pipeline, model, tokenizer, context, query, max_new_tokens):
    """Run one generation, return (prediction, latency_sec, peak_mem_mb)."""
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
        outputs = run_hypernet(model, tokenizer, example, max_new_tokens=max_new_tokens)
    else:
        outputs = run_baseline(model, tokenizer, example, max_new_tokens=max_new_tokens)

    if device is not None:
        torch.cuda.synchronize(device)
    latency = time.time() - t0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device is not None
        else 0.0
    )
    return outputs[0].strip(), latency, peak_mem_mb


# -------------------------------------------------------------------
# GOLD METRICS
# -------------------------------------------------------------------

def compute_recall_gold(retrieved_texts, gold_sentences, k):
    top_k = retrieved_texts[:k]
    return int(any(
        any(gold.lower() in ctx.lower() for ctx in top_k)
        for gold in gold_sentences
    ))


def compute_mrr_gold(retrieved_texts, gold_sentences, k):
    for i, ctx in enumerate(retrieved_texts[:k]):
        if any(gold.lower() in ctx.lower() for gold in gold_sentences):
            return 1.0 / (i + 1)
    return 0.0


def compute_f1_gold(retrieved_texts, gold_sentences, k):
    best_f1 = 0.0
    for ctx in retrieved_texts[:k]:
        for gold in gold_sentences:
            f1 = compute_f1(ctx, gold)
            best_f1 = max(best_f1, f1)
    return best_f1
    

def extract_gold_sentences(text):
    # Simple sentence splitter (can be replaced with nltk.sent_tokenize for better accuracy)
    sentences = text.strip().split('\n')
    return sentences

# -------------------------------------------------------------------
# MAIN EXECUTION 
# -------------------------------------------------------------------

if __name__ == "__main__":

    num_samples = len(queries) if args.num_samples is None else min(args.num_samples, len(queries))

    recall_2 = 0
    recall_5 = 0
    mrr_2_total = 0
    mrr_5_total = 0
    f1_2_total = 0
    f1_5_total = 0

    # generation setup
    run_generation = args.pipeline != "none"
    gen_em_total = 0.0
    gen_f1_total = 0.0
    gen_contain_total = 0.0
    gen_latencies = []
    gen_peak_mems = []

    gen_model = None
    gen_tokenizer = None

    if run_generation:
        print(f"\nLoading generator: pipeline={args.pipeline}")
        gen_model, gen_tokenizer = load_generator(args.pipeline, args.model_path)

    retrieved_records = []

    print("\nRunning evaluation...")

    for i in tqdm(range(num_samples), desc="Eval Progress"):

        query = queries[i]
        true_answer = answers[i]

        retrieved_texts, scored = retrieve_with_colbert_rerank(
            query,
            top_k=RERANK_TOP_K,
            candidate_k=max(BM25_CANDIDATE_K, DENSE_CANDIDATE_K),
        )

        record = {
            "id": i,
            "prompt": query,
            "full_context": contexts[i],
            "retrieved_context": retrieved_texts,
            "answer": true_answer,
        }

        # ---------------- GOLD SENTENCES ----------------
        gold_sentences = extract_gold_sentences(golden_contexts[i])

        # ---------------- GOLD RECALL ----------------
        recall_2 += compute_recall_gold(retrieved_texts, gold_sentences, k=2)
        recall_5 += compute_recall_gold(retrieved_texts, gold_sentences, k=5)

        # ---------------- GOLD MRR ----------------
        mrr_2_total += compute_mrr_gold(retrieved_texts, gold_sentences, k=2)
        mrr_5_total += compute_mrr_gold(retrieved_texts, gold_sentences, k=5)

        # ---------------- GOLD F1 ----------------
        f1_2_total += compute_f1_gold(retrieved_texts, gold_sentences, k=2)
        f1_5_total += compute_f1_gold(retrieved_texts, gold_sentences, k=5)

        # ---------------- GENERATION ----------------
        if run_generation:

            if args.context_mode == "per_chunk" and args.pipeline == "doc2lora":
                gen_context = list(retrieved_texts)
            else:
                gen_context = "\n\n".join(retrieved_texts)

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

            record.update({
                "prediction": prediction,
                "gen_em": em,
                "gen_f1": f1,
                "gen_contain": contain,
                "latency": latency,
                "mem": peak_mem_mb,
            })

        retrieved_records.append(record)

    # ---------------- FINAL METRICS ----------------

    recall_2 /= num_samples
    recall_5 /= num_samples
    mrr_2 = mrr_2_total / num_samples
    mrr_5 = mrr_5_total / num_samples
    f1_2 = f1_2_total / num_samples
    f1_5 = f1_5_total / num_samples

    print("\n===== RETRIEVAL (GOLD-BASED) =====")
    print(f"Recall@2    : {recall_2:.4f}")
    print(f"Recall@5    : {recall_5:.4f}")
    print(f"MRR@2       : {mrr_2:.4f}")
    print(f"MRR@5       : {mrr_5:.4f}")
    print(f"F1@2       : {f1_2:.4f}")
    print(f"F1@5       : {f1_5:.4f}")

    if run_generation:
        gen_em = gen_em_total / num_samples
        gen_f1 = gen_f1_total / num_samples
        gen_contain = gen_contain_total / num_samples
        avg_latency = sum(gen_latencies) / len(gen_latencies)
        avg_peak_mem = sum(gen_peak_mems) / len(gen_peak_mems)
        max_peak_mem = max(gen_peak_mems)

        print(f"\n===== GENERATION ({args.pipeline}) =====")
        print(f"Answer EM       : {gen_em:.4f}  (on cleaned prediction)")
        print(f"Answer F1       : {gen_f1:.4f}  (on cleaned prediction)")
        print(f"Containment     : {gen_contain:.4f}  (gold in raw prediction)")
        print(f"Avg latency (s) : {avg_latency:.4f}")
        print(f"Avg peak mem MB : {avg_peak_mem:.1f}")
        print(f"Max peak mem MB : {max_peak_mem:.1f}")

        Path(args.gen_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.gen_output, "w") as f:
            json.dump({
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
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved generation results to {args.gen_output}")

    Path(args.retrieved_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.retrieved_output, "w") as f:
        json.dump(retrieved_records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(retrieved_records)} retrieved records to {args.retrieved_output}")