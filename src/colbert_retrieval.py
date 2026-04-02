"""ColBERTv2 retrieval via RAGatouille: PLAID indexing, late interaction top-k retrieval with Recall@k."""

import gc
import logging
import time

import torch

# Monkey-patch ColBERT to skip C++ extension compilation on Windows (no MSVC needed).
# Provides a pure-Python fallback for segmented_maxsim.
import colbert.modeling.colbert as _cm

@classmethod
def _patched_try_load(cls, use_gpu):
    if hasattr(cls, "loaded_extensions") or use_gpu:
        return
    # Pure-Python fallback: ColBERT will use the torch-based path instead
    cls.loaded_extensions = True

_cm.ColBERT.try_load_torch_extensions = _patched_try_load

from ragatouille import RAGPretrainedModel

from src.profiler import Profiler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 5.1  Index corpus with ColBERTv2
# ---------------------------------------------------------------------------

def build_colbert_index(
    chunks: list[dict],
    index_path: str = "results/colbert_index",
    model_name: str = "colbert-ir/colbertv2.0",
) -> RAGPretrainedModel:
    """Create a ColBERTv2 PLAID index from corpus chunks via RAGatouille."""
    rag = RAGPretrainedModel.from_pretrained(model_name)

    texts = [c["text"] for c in chunks]
    doc_ids = [c["chunk_id"] for c in chunks]

    rag.index(
        collection=texts,
        document_ids=doc_ids,
        index_name="colbert_benchmark",
        split_documents=False,
        max_document_length=180,
    )
    return rag


# ---------------------------------------------------------------------------
# 5.2  Top-k retrieval with late interaction scoring
# ---------------------------------------------------------------------------

def retrieve_top_k_colbert(
    rag: RAGPretrainedModel,
    query: str,
    k: int = 20,
) -> tuple[list[str], float]:
    """Retrieve top-k chunk IDs using ColBERTv2 late interaction.

    Returns (chunk_ids, latency_ms).
    """
    start = time.perf_counter()
    results = rag.search(query=query, k=k)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieved_ids = [r["document_id"] for r in results]
    return retrieved_ids, latency_ms


# ---------------------------------------------------------------------------
# 5.3  Recall@k computation (reuse from biencoder module)
# ---------------------------------------------------------------------------

def compute_recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float:
    """Compute Recall@k: fraction of ground-truth items found in top-k retrieved."""
    if not ground_truth_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & ground_truth_ids) / len(ground_truth_ids)


# ---------------------------------------------------------------------------
# 5.4 + 5.5  Full ColBERT pipeline with VRAM error handling and profiler
# ---------------------------------------------------------------------------

def run_colbert_retrieval(
    chunks: list[dict],
    sampled_queries: list[dict],
    config: dict,
    profiler: Profiler,
) -> None:
    """Run ColBERTv2 indexing and retrieval. Logs results via profiler.

    If indexing fails due to VRAM overflow, logs the error and raises with
    guidance to reduce corpus size.
    """
    model_name = config["models"]["colbert"]
    k_values = config["retrieval"]["k_values"]
    max_k = max(k_values)
    index_path = config["paths"]["colbert_index_dir"]

    # Index corpus
    profiler.start_stage("colbert_indexing")
    try:
        rag = build_colbert_index(chunks, index_path=index_path, model_name=model_name)
    except (RuntimeError, MemoryError) as e:
        profiler.end_stage("colbert_indexing")
        msg = (
            f"ColBERTv2 indexing failed (likely VRAM overflow): {e}\n"
            f"Try reducing corpus.target_size in configs/experiment_config.yaml "
            f"(current: {config['corpus']['target_size']}). Suggested: 5000."
        )
        logger.error(msg)
        raise RuntimeError(msg) from e
    profiler.end_stage("colbert_indexing")

    # Record index disk size
    profiler.record_disk_size("colbert_index", index_path)

    # Retrieve for each query
    profiler.start_stage("colbert_retrieval")
    for q in sampled_queries:
        gt_ids = set(q["ground_truth_chunk_ids"])
        retrieved_ids, latency_ms = retrieve_top_k_colbert(rag, q["query"], k=max_k)

        recall_at_k = {str(k): compute_recall_at_k(retrieved_ids, gt_ids, k) for k in k_values}

        q["colbert_retrieved_ids"] = retrieved_ids
        q["colbert_recall_at_k"] = recall_at_k
        q["colbert_latency_ms"] = round(latency_ms, 4)
    profiler.end_stage("colbert_retrieval")

    # Free ColBERT resources
    del rag
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
