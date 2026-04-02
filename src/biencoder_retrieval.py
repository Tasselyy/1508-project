"""Bi-encoder retrieval: sentence-transformers encoding, FAISS indexing, top-k retrieval with Recall@k."""

import gc
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.profiler import Profiler


# ---------------------------------------------------------------------------
# 4.1  Encode corpus chunks
# ---------------------------------------------------------------------------

def encode_chunks(
    chunks: list[dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
) -> tuple[SentenceTransformer, np.ndarray]:
    """Encode all corpus chunks and return the model + embeddings matrix."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return model, np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4.2  Build FAISS flat index
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP (inner product / cosine on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# 4.3  Top-k query retrieval
# ---------------------------------------------------------------------------

def retrieve_top_k(
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    query: str,
    k: int = 20,
) -> tuple[list[str], float]:
    """Retrieve top-k chunk IDs for a query. Returns (chunk_ids, latency_ms)."""
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)

    start = time.perf_counter()
    scores, indices = index.search(q_emb, k)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_ids = [chunks[int(idx)]["chunk_id"] for idx in indices[0] if idx < len(chunks)]
    return retrieved_ids, latency_ms


# ---------------------------------------------------------------------------
# 4.4  Recall@k computation
# ---------------------------------------------------------------------------

def compute_recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Recall@k: fraction of ground-truth items found in top-k retrieved."""
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & ground_truth_ids) / len(ground_truth_ids)


# ---------------------------------------------------------------------------
# 4.5  Full bi-encoder pipeline with profiler
# ---------------------------------------------------------------------------

def run_biencoder_retrieval(
    chunks: list[dict],
    sampled_queries: list[dict],
    config: dict,
    profiler: Profiler,
) -> None:
    """Run bi-encoder encoding, indexing, and retrieval. Logs results via profiler."""
    model_name = config["models"]["biencoder"]
    k_values = config["retrieval"]["k_values"]
    max_k = max(k_values)

    # Encode corpus
    profiler.start_stage("biencoder_encoding")
    model, embeddings = encode_chunks(chunks, model_name=model_name)
    profiler.end_stage("biencoder_encoding")

    # Build FAISS index
    profiler.start_stage("faiss_indexing")
    index = build_faiss_index(embeddings)
    profiler.end_stage("faiss_indexing")

    # Save FAISS index to disk for size measurement
    index_path = config["paths"]["faiss_index_dir"]
    import os
    os.makedirs(index_path, exist_ok=True)
    faiss_file = os.path.join(index_path, "index.faiss")
    faiss.write_index(index, faiss_file)
    profiler.record_disk_size("faiss_index", index_path)

    # Retrieve for each query
    profiler.start_stage("biencoder_retrieval")
    for q in sampled_queries:
        gt_ids = set(q["ground_truth_chunk_ids"])
        retrieved_ids, latency_ms = retrieve_top_k(model, index, chunks, q["query"], k=max_k)

        recall_at_k = {str(k): compute_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values}

        q["biencoder_retrieved_ids"] = retrieved_ids
        q["biencoder_recall_at_k"] = recall_at_k
        q["biencoder_latency_ms"] = round(latency_ms, 4)
    profiler.end_stage("biencoder_retrieval")

    # Free bi-encoder resources to reclaim memory before ColBERT stage
    del model, embeddings, index
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
