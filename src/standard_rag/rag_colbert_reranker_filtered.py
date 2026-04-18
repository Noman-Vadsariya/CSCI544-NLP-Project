import os
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
NUM_SAMPLES = 200

UPSERT_PINECONE = True  # set to False to skip Pinecone upload (use existing index)

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

ds = load_dataset("parquet", data_files=DATA_PATH)["train"]

print("num rows:", len(ds))
print("columns:", ds.column_names)

contexts = ds["context"]
queries = [x[0] for x in ds["prompts"]]
answers = [x[0] for x in ds["responses"]]

print("Total QA pairs:", len(queries))
print("Total contexts:", len(contexts))

# -------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Dense embedding model for Pinecone retrieval
dense_model = SentenceTransformer(BGE_MODEL_NAME, device=device)

# Raw transformer backbone for token-level late interaction scoring
# Important: this is NOT SentenceTransformer pooling; we use last_hidden_state.
tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)
encoder = AutoModel.from_pretrained(BGE_MODEL_NAME).to(device)
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
# SENTENCE FILTER 
# -------------------------------------------------------------------

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text)

def filter_sentences(query, contexts, top_n=2):
    q_emb = dense_model.encode(
        ["query: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        device=device,
    )[0]

    filtered_contexts = []

    for ctx in contexts:
        sentences = split_into_sentences(ctx)

        if len(sentences) == 0:
            filtered_contexts.append(ctx)
            continue

        sent_embs = dense_model.encode(
            ["passage: " + s for s in sentences],
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=device,
        )

        scores = sent_embs @ q_emb
        top_idx = scores.argsort()[::-1][:top_n]

        selected = [sentences[i] for i in top_idx]
        filtered_contexts.append(" ".join(selected))

    return filtered_contexts


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
    retrieved_texts = filter_sentences(query, retrieved_texts, top_n=2)

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


# -------------------------------------------------------------------
# EVALUATION 
# -------------------------------------------------------------------

def compute_recall_at_k(ranked_texts, answer, k):
    top_k = ranked_texts[:k]
    return int(any(answer.lower() in ctx.lower() for ctx in top_k))


def compute_mrr_at_k(scored_pids, answer, k):
    for i, (pid, _) in enumerate(scored_pids[:k]):
        text = pid_to_text[pid]
        if answer.lower() in text.lower():
            return 1.0 / (i + 1)
    return 0.0


num_samples = min(NUM_SAMPLES, len(queries))

accuracy = 0
recall_2 = 0
recall_5 = 0
mrr_2_total = 0
mrr_5_total = 0
f1_total = 0  

print("\nRunning evaluation...")

for i in tqdm(range(num_samples), desc="Eval Progress"):
    query = queries[i]
    true_answer = answers[i]

    retrieved_texts, scored = retrieve_with_colbert_rerank(
        query,
        top_k=RERANK_TOP_K,
        candidate_k=max(BM25_CANDIDATE_K, DENSE_CANDIDATE_K),
    )

    # ---------------- accuracy - EM ----------------
    found_ctx = any(
        true_answer.lower() in ctx.lower()
        or all(word in ctx.lower() for word in true_answer.lower().split())
        for ctx in retrieved_texts
    )

    if found_ctx:
        accuracy += 1

    # ---------------- recall ----------------
    recall_2 += compute_recall_at_k(retrieved_texts, true_answer, k=2)
    recall_5 += compute_recall_at_k(retrieved_texts, true_answer, k=5)

    # ---------------- mrr ----------------
    mrr_2_total += compute_mrr_at_k(scored, true_answer, k=2)
    mrr_5_total += compute_mrr_at_k(scored, true_answer, k=5)

    # ---------------- F1  ----------------
    best_f1 = 0.0
    for ctx in retrieved_texts:
        f1 = compute_f1(ctx, true_answer)
        best_f1 = max(best_f1, f1)

    f1_total += best_f1


# ---------------- final metrics ----------------

accuracy /= num_samples
recall_2 /= num_samples
recall_5 /= num_samples
mrr_2 = mrr_2_total / num_samples
mrr_5 = mrr_5_total / num_samples
f1_score = f1_total / num_samples  

print("\n===== RESULTS =====")
print(f"Accuracy@10 : {accuracy:.4f}")
print(f"Recall@2    : {recall_2:.4f}")
print(f"Recall@5    : {recall_5:.4f}")
print(f"MRR@2       : {mrr_2:.4f}")
print(f"MRR@5       : {mrr_5:.4f}")
print(f"F1 (word)   : {f1_score:.4f}")  