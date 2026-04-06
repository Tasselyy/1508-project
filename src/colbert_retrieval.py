"""ColBERTv2 retrieval via RAGatouille: PLAID indexing, late interaction top-k retrieval with Recall@k."""

import gc
import logging
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Monkey-patch ColBERT for Windows (no MSVC / CUDA_HOME).
#
# ColBERT JIT-compiles C++ extensions at runtime.  On Windows without a
# C++ toolchain this always fails.  We skip the JIT compilation and provide
# pure-Python / numpy fallbacks for the four affected classes:
#   ColBERT          – segmented_maxsim   (CPU path, has built-in torch fallback)
#   ResidualCodec    – packbits, decompress_residuals  (GPU path)
#   StridedTensor    – segmented_lookup   (CPU path)
#   IndexScorer      – filter_pids, decompress_residuals (CPU path)
#
# The GPU PLAID indexer OOMs on 8 GB VRAM; RAGatouille catches this and
# retries with CPU FAISS.  The retry still marks use_gpu=True (GPU exists)
# so ResidualCodec needs GPU-side fallbacks for packbits / decompress.
# ---------------------------------------------------------------------------
import numpy as np
import colbert.modeling.colbert as _cm
import colbert.indexing.codecs.residual as _cr
import colbert.search.strided_tensor as _st
import colbert.search.index_storage as _is

# 1. Skip JIT compilation on all four classes
@classmethod
def _patched_try_load(cls, use_gpu):
    if not hasattr(cls, "loaded_extensions"):
        cls.loaded_extensions = True

_cm.ColBERT.try_load_torch_extensions = _patched_try_load
_cr.ResidualCodec.try_load_torch_extensions = _patched_try_load
_st.StridedTensor.try_load_torch_extensions = _patched_try_load
_is.IndexScorer.try_load_torch_extensions = _patched_try_load

# 2. Provide pure-Python fallbacks for ResidualCodec GPU code-paths
#    These are only hit when GPU indexing OOMs and the FAISS retry runs
#    with use_gpu=True.

@staticmethod
def _fallback_packbits(tensor):
    """Pure-Python replacement for the CUDA packbits kernel."""
    packed = np.packbits(np.asarray(tensor.contiguous().cpu()))
    return torch.as_tensor(packed, dtype=torch.uint8)

@staticmethod
def _fallback_decompress_residuals(
    residuals, bucket_weights, reversed_bit_map,
    decompression_lookup_table, codes, centroids, dim, nbits,
):
    """Pure-Python replacement for the CUDA decompress_residuals kernel."""
    centroids_ = centroids[codes.long()]
    residuals_ = reversed_bit_map[residuals.long()]
    residuals_ = decompression_lookup_table[residuals_.long()]
    residuals_ = residuals_.reshape(residuals_.shape[0], -1)
    residuals_ = bucket_weights[residuals_.long()]
    centroids_ = centroids_ + residuals_
    return centroids_

if not hasattr(_cr.ResidualCodec, "packbits"):
    _cr.ResidualCodec.packbits = _fallback_packbits
if not hasattr(_cr.ResidualCodec, "decompress_residuals"):
    _cr.ResidualCodec.decompress_residuals = _fallback_decompress_residuals

from ragatouille import RAGPretrainedModel

from src.profiler import Profiler

logger = logging.getLogger(__name__)
COLBERT_INDEX_NAME = "colbert_benchmark"


# ---------------------------------------------------------------------------
# 5.1  Index corpus with ColBERTv2
# ---------------------------------------------------------------------------

def build_colbert_index(
    chunks: list[dict],
    index_path: str = "results/colbert_index",
    model_name: str = "colbert-ir/colbertv2.0",
    bsize: int = 32,
) -> RAGPretrainedModel:
    """Create a ColBERTv2 PLAID index from corpus chunks via RAGatouille.

    Args:
        bsize: Batch size for ColBERT encoding during indexing. Lower values
               reduce peak VRAM usage (default 32, try 16 for 8GB VRAM).
    """
    index_root = Path(index_path)
    index_root.mkdir(parents=True, exist_ok=True)
    rag = RAGPretrainedModel.from_pretrained(model_name, index_root=str(index_root))

    texts = [c["text"] for c in chunks]
    doc_ids = [c["chunk_id"] for c in chunks]

    rag.index(
        collection=texts,
        document_ids=doc_ids,
        index_name=COLBERT_INDEX_NAME,
        split_documents=False,
        max_document_length=180,
        bsize=bsize,
    )
    return rag


def resolve_colbert_index_path(index_path: str) -> Path:
    """Best-effort resolution of the on-disk ColBERT index directory."""
    configured = Path(index_path)
    candidates = [
        configured / COLBERT_INDEX_NAME,
        configured,
        Path(".ragatouille") / "colbert" / "indexes" / COLBERT_INDEX_NAME,
        Path(".ragatouille") / "indexes" / COLBERT_INDEX_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return configured


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
        bsize = config.get("colbert", {}).get("bsize", 32)
        rag = build_colbert_index(chunks, index_path=index_path, model_name=model_name, bsize=bsize)
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
    resolved_index_path = resolve_colbert_index_path(index_path)
    profiler.record_disk_size("colbert_index", str(resolved_index_path))
    profiler.data["metadata"]["colbert_index_path"] = str(resolved_index_path)

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
