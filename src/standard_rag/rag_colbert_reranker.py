import argparse
import os
import json
import time
import torch
import torch.nn.functional as F
import re

from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec

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

# Final rerank cutoff — must be >= max(K_VALUES) below
RERANK_TOP_K = 10

# k values evaluated for retrieval and generation
K_VALUES = (10,)

# Late-interaction encoder limits, keep these aligned with chunking 
MAX_QUERY_LEN = 64
MAX_PASSAGE_LEN = 480

# Evaluation
NUM_SAMPLES = 2

UPSERT_PINECONE = True  # set to False to skip Pinecone upload (use existing index)

# -------------------------------------------------------------------
# CLI ARGS 
# -------------------------------------------------------------------

def parse_gen_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "--data_path",
        dest="input",
        type=str,
        default=DATA_PATH,
        help="Parquet dataset path (overrides the module-level DATA_PATH default).",
    )
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
    parser.add_argument(
        "--pinecone_namespace",
        type=str,
        default=None,
        help="Pinecone namespace to use. Defaults to the dataset filename stem (e.g. 'asqa_gold_subset').",
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
        help="short: terse answer + span extraction + EM/F1 (HotpotQA). "
             "full: long answer + no span extraction + ROUGE-L (ASQA).",
    )
    return parser.parse_args()


args = parse_gen_args()
embedding_model_name_or_path = args.embedding_model or BGE_MODEL_NAME
print(f"Embedding model: {embedding_model_name_or_path}")

PINECONE_NAMESPACE = args.pinecone_namespace or Path(args.input).stem
print(f"Pinecone namespace: {PINECONE_NAMESPACE}")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

ds = load_dataset("parquet", data_files=args.input)["train"]

print("num rows:", len(ds))
print("columns:", ds.column_names)

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]
if "gold_context" in ds.column_names:
    gold_contexts = ds["gold_context"]
elif "needle_text" in ds.column_names:
    gold_contexts = ds["needle_text"]
else:
    gold_contexts = ds["context"]

print("Total QA pairs:", len(queries))
print("Total contexts:", len(contexts))

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dense embedding model for Pinecone retrieval
dense_model = SentenceTransformer(embedding_model_name_or_path, device=device)

# transformer backbone for token-level late interaction scoring
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
        self.passage_cache = {} [num_tokens, hidden_dim]

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
        query_emb: [q_len, dim]
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

    Returns retrieved_texts and scored_pids
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

def extract_answer_span(text, gold):
    """Post-process free-form LLM output into a short answer span."""
    if not text:
        return ""

    gold_norm = normalize_answer(gold)

    # look at the first word of the raw prediction to get a yes or no answer
    first_word = re.split(r"[\s,.!?:;]+", text.strip().lower(), maxsplit=1)[0]
    if gold_norm in {"yes", "no"} and first_word in {"yes", "no"}:
        return first_word

    # Strip common markdowns
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


# -------------------------------------------------------------------
# GENERATION
# -------------------------------------------------------------------

def _prompt_token_count(tokenizer, context_str, query, answer_style):
    # for the longer answer
    if answer_style == "full":
        user_content = (
            f"Answer the question fully and completely based on the given passages. "
            f"Your answer should cover all relevant aspects and may be multiple sentences, but keep it concise. "
            f"If the passages do not contain enough information to answer, reply with exactly: answer not in context.\n\n"
            f"Passages:\n{context_str}\n\n"
            f"Question: {query}"
        )
    # shorter answer to match hotpotqa
    else:
        user_content = (
            f"Answer the question using only the given passages. "
            f"If the passages do not contain the answer, reply with exactly: answer not in context. "
            f"Otherwise give only the answer and do not output any other words.\n\n"
            f"Passages:\n{context_str}\n\n"
            f"Question: {query}"
        )
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return tokens.shape[1]

# helper 
def context_fits_in_window(tokenizer, context_str, query, answer_style, max_new_tokens, model_max_len):
    prompt_len = _prompt_token_count(tokenizer, context_str, query, answer_style)
    return (prompt_len + max_new_tokens) <= model_max_len


def load_generator(pipeline: str, model_path: str | None):
    if pipeline == "doc2lora":
        return load_hypernet(model_path) if model_path else load_hypernet()
    if pipeline == "regular":
        resolved = model_path
        if resolved and os.path.isfile(resolved):
            resolved = os.path.dirname(resolved)
        return load_baseline(resolved) if resolved else load_baseline()
    raise ValueError(f"Unknown pipeline: {pipeline}")


def generate_answer(pipeline, model, tokenizer, context, query, max_new_tokens, answer_style="short"):
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
        outputs = run_hypernet(model, tokenizer, example, max_new_tokens=max_new_tokens, answer_style=answer_style)
    else:
        outputs = run_baseline(model, tokenizer, example, max_new_tokens=max_new_tokens, answer_style=answer_style)

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

def compute_recall_text(retrieved_texts, gold_sentences, k):
    top_k = retrieved_texts[:k]
    return int(
        any(any(gold.lower() in ctx.lower() for ctx in top_k) for gold in gold_sentences
    ))


def compute_mrr_text(retrieved_texts, gold_sentences, k):
    """Text-based MRR: rank of first chunk containing any gold sentence."""
    for i, ctx in enumerate(retrieved_texts[:k]):
        if any(gold.lower() in ctx.lower() for gold in gold_sentences):
            return 1.0 / (i + 1)
    return 0.0


def compute_recall_index(scored_pids, gold_example_id, chunk_id_to_original, k):
    """Index-based recall: any top-k chunk's source document id matches the query's example id."""
    for pid, _ in scored_pids[:k]:
        if chunk_id_to_original[pid] == gold_example_id:
            return 1
    return 0


def compute_mrr_index(scored_pids, gold_example_id, chunk_id_to_original, k):
    """Index-based MRR: rank of first chunk whose source doc id matches."""
    for i, (pid, _) in enumerate(scored_pids[:k]):
        if chunk_id_to_original[pid] == gold_example_id:
            return 1.0 / (i + 1)
    return 0.0


def compute_f1_gold(retrieved_texts, gold_context, k):
    """Best token-F1 between any top-k chunk and the full gold context string."""
    best_f1 = 0.0
    for ctx in retrieved_texts[:k]:
        best_f1 = max(best_f1, compute_f1(ctx, gold_context))
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

    recall_text = {k: 0 for k in K_VALUES}
    recall_index = {k: 0 for k in K_VALUES}
    mrr_text = {k: 0.0 for k in K_VALUES}
    mrr_index = {k: 0.0 for k in K_VALUES}
    f1_k = {k: 0.0 for k in K_VALUES}

    # generation setup
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
    gen_latencies = {k: [] for k in K_VALUES}
    gen_peak_mems = {k: [] for k in K_VALUES}
    gen_skipped = {k: 0 for k in K_VALUES}

    gen_model = None
    gen_tokenizer = None
    model_max_len = None

    if run_generation:
        print(f"\nLoading generator: pipeline={args.pipeline}")
        gen_model, gen_tokenizer = load_generator(args.pipeline, args.model_path)
        if args.pipeline == "regular":
            try:
                model_max_len = gen_model.config.max_position_embeddings
            except Exception:
                model_max_len = 8192
            print(f"Model context window: {model_max_len} tokens")

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

        # GOLD SENTENCES 
        gold_sentences = extract_gold_sentences(gold_contexts[i])

        # calculate RECALL/MRR/F1 @ k 
        for k in K_VALUES:
            recall_text[k] += compute_recall_text(retrieved_texts, gold_sentences, k=k)
            recall_index[k] += compute_recall_index(scored, i, chunk_id_to_original, k=k)
            mrr_text[k] += compute_mrr_text(retrieved_texts, gold_sentences, k=k)
            mrr_index[k] += compute_mrr_index(scored, i, chunk_id_to_original, k=k)
            f1_k[k] += compute_f1_gold(retrieved_texts, gold_contexts[i], k=k)

        # perform GENERATION for top-k
        if run_generation:
            gen_results = {}

            for k in K_VALUES:
                top_k_texts = retrieved_texts[:k]

                if args.pipeline == "doc2lora":
                    # Always internalize each chunk as a separate LoRA adapter.
                    gen_context = list(top_k_texts)
                else:
                    gen_context = "\n\n".join(top_k_texts)
                    # Skip generation if context + generation budget exceeds the window.
                    if model_max_len is not None and not context_fits_in_window(
                        gen_tokenizer, gen_context, query,
                        args.answer_style, args.max_new_tokens, model_max_len,
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

                # try and except for samples that get a cuda error
                try:
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
                    if "aligned" in str(e).lower() or "cuda" in str(e).lower():
                        print(f"[WARN] Skipping sample {i} k={k} due to CUDA error: {e}")
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

                if args.answer_style == "full":
                    prediction = raw_prediction
                else:
                    prediction = extract_answer_span(raw_prediction, true_answer)

                em = compute_em(prediction, true_answer)
                f1 = compute_f1(prediction, true_answer)
                contain = compute_containment(raw_prediction, true_answer)
                rouge_l = compute_rouge_l(prediction, true_answer)

                # Retrieval-aware: credit correct refusals when gold isn't in context.
                pred_for_refusal_check = prediction if args.answer_style != "full" else raw_prediction
                em_a, f1_a, rouge_l_a, contain_a, refused_correct, refused, gold_found = apply_refusal_credit(
                    em, f1, rouge_l, contain, pred_for_refusal_check, true_answer, top_k_texts,
                )

                # store the evals
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
                    "prediction": prediction,
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
                    "latency": latency,
                    "mem": peak_mem_mb,
                }

            record["generation"] = gen_results

        retrieved_records.append(record)

    # Report the final metrics
    for k in K_VALUES:
        recall_text[k] /= num_samples
        recall_index[k] /= num_samples
        mrr_text[k] /= num_samples
        mrr_index[k] /= num_samples
        f1_k[k] /= num_samples

    print("\n===== RETRIEVAL (TEXT: gold substring in retrieved chunk) =====")
    for k in K_VALUES:
        print(f"Recall@{k:<2} : {recall_text[k]:.4f}   MRR@{k:<2} : {mrr_text[k]:.4f}")

    print("\n===== RETRIEVAL (INDEX: retrieved chunk's source doc == query's doc) =====")
    for k in K_VALUES:
        print(f"Recall@{k:<2} : {recall_index[k]:.4f}   MRR@{k:<2} : {mrr_index[k]:.4f}")

    print("\n===== RETRIEVAL F1 (vs full gold context) =====")
    for k in K_VALUES:
        print(f"F1@{k:<2} : {f1_k[k]:.4f}")

    if run_generation:
        print(f"\n===== GENERATION ({args.pipeline}, answer_style={args.answer_style}) =====")
        gen_summary = {}

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

        Path(args.gen_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.gen_output, "w") as f:
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

    Path(args.retrieved_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.retrieved_output, "w") as f:
        json.dump(retrieved_records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(retrieved_records)} retrieved records to {args.retrieved_output}")

    if args.metrics_output:
        metrics = {
            "pipeline": args.pipeline,
            "input": args.input,
            "num_samples": num_samples,
            "retrieval": {
                "recall_text": {str(k): recall_text[k] for k in K_VALUES},
                "recall_index": {str(k): recall_index[k] for k in K_VALUES},
                "mrr_text": {str(k): mrr_text[k] for k in K_VALUES},
                "mrr_index": {str(k): mrr_index[k] for k in K_VALUES},
                "f1_vs_gold": {str(k): f1_k[k] for k in K_VALUES},
            },
        }

        if run_generation:
            metrics["generation"] = {
                "answer_style": args.answer_style,
                **gen_summary,
            }

        Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"Saved metrics to {args.metrics_output}")