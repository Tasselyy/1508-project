# ColBERTv2 vs Bi-Encoder Retrieval Benchmark

A standalone benchmark comparing ColBERTv2 (late interaction) against a bi-encoder baseline (MiniLM) on the KILT NaturalQuestions dev set. Part of the ECE1508 course project "Enhancing RAG with Adaptive Chunking."

## Setup

### Prerequisites

- Python 3.10+
- GPU recommended (NVIDIA with CUDA support) for ColBERTv2 indexing
- ~20 GB disk space for data and indexes

### Installation

```bash
# Create and activate virtual environment
uv venv .venv
.venv\Scripts\activate   # Windows PowerShell
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Data Download

Data is downloaded automatically on first run:

1. **KILT NQ dev set** (~50 MB) — downloaded via HuggingFace `datasets`
2. **KILT Wikipedia** (~35 GB streamed) — only ~10K pages are kept; cached locally at `data/corpus_cache.parquet` for subsequent runs

To skip streaming on repeat runs, ensure the cache file exists at the path specified in `configs/experiment_config.yaml` → `corpus.local_corpus_cache`.

## Running the Benchmark

Open and run the Jupyter notebook end-to-end:

```bash
jupyter notebook notebooks/run_benchmark.ipynb
```

The notebook executes:
1. **Data Pipeline** — loads NQ queries, builds reduced Wikipedia corpus, chunks paragraphs, classifies queries by NER entity count
2. **Bi-Encoder Retrieval** — encodes chunks with `all-MiniLM-L6-v2`, builds FAISS index, retrieves top-k
3. **ColBERTv2 Retrieval** — indexes chunks with RAGatouille ColBERTv2, retrieves top-k with late interaction
4. **JSON Log** — saves all profiling data and per-query results to `results/benchmark_log.json`
5. **Evaluation** — computes Recall@k from JSON log, grouped by entity complexity
6. **Visualization** — generates charts and CSV summary table

## Configuration

Edit `configs/experiment_config.yaml` to adjust:

| Parameter | Default | Description |
|---|---|---|
| `corpus.target_size` | 10000 | Number of Wikipedia pages in reduced corpus |
| `queries.sample_size_per_group` | 500 | Queries per entity group |
| `retrieval.k_values` | [1, 5, 10, 20] | Top-k values for retrieval |
| `spacy.entity_threshold` | 2 | NER entity count threshold for multi-entity classification |

## Hardware Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 4 GB (use 5K corpus) | 8+ GB (10K corpus) |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB | 20 GB |
| Time | ~30 min (GPU) | ~1 hour (CPU-only) |

If ColBERTv2 indexing fails due to VRAM overflow, reduce `corpus.target_size` to 5000 in the config.

## Ablation Experiments

Run systematic ablation studies across corpus scale, chunk granularity, and top-k depth:

```bash
python run_ablation.py
```

Configuration: `configs/ablation_config.yaml`. See `doc/experiment_guide.md` for details on each ablation dimension.

Results are written to `results/ablation/`.

## Output

All results are written to the `results/` directory:

- `benchmark_log.json` — Single source of truth: timing, GPU/RAM metrics, disk sizes, per-query results, run metadata
- `charts/` — PNG visualizations (Recall@k bar chart, histograms, latency comparison, index size comparison)
- `summary_statistics.csv` — Recall@k summary table with delta columns
- `faiss_index/` — Saved FAISS index
- `colbert_index/` — Saved ColBERT PLAID index

### JSON Log Structure

```json
{
  "metadata": {
    "timestamp": "...",
    "config": {...},
    "gpu_device": "NVIDIA ...",
    "gpu_total_vram_bytes": ...,
    "corpus_size": 10000,
    "total_chunks": ...,
    "total_queries": ...,
    "queries_per_group": {"single-entity": ..., "multi-entity": ...},
    "models": {...},
    "k_values": [1, 5, 10, 20]
  },
  "stages": {
    "data_loading": {"duration_seconds": ..., "peak_vram_bytes": ..., "rss_bytes": ...},
    ...
  },
  "queries": [
    {
      "query": "...",
      "entity_count": 1,
      "entity_list": ["..."],
      "entity_group": "single-entity",
      "ground_truth_chunk_ids": ["..."],
      "biencoder_retrieved_ids": ["..."],
      "biencoder_recall_at_k": {"1": ..., "5": ..., "10": ..., "20": ...},
      "biencoder_latency_ms": ...,
      "colbert_retrieved_ids": ["..."],
      "colbert_recall_at_k": {"1": ..., "5": ..., "10": ..., "20": ...},
      "colbert_latency_ms": ...
    }
  ],
  "disk_sizes": {"faiss_index": ..., "colbert_index": ...}
}
```

## Project Structure

```
├── configs/
│   ├── experiment_config.yaml   # Main benchmark configuration
│   └── ablation_config.yaml     # Ablation experiment configuration
├── doc/
│   ├── architecture.md          # System architecture overview
│   ├── experiment_guide.md      # Detailed experiment guide
│   └── data_format.md           # Data format reference
├── notebooks/
│   └── run_benchmark.ipynb      # Demo notebook (end-to-end pipeline)
├── src/
│   ├── profiler.py              # Stage timing, GPU/RAM/disk profiling
│   ├── data_pipeline.py         # KILT data loading, chunking, NER classification
│   ├── biencoder_retrieval.py   # MiniLM encoding, FAISS indexing, retrieval
│   ├── colbert_retrieval.py     # RAGatouille ColBERTv2 indexing, retrieval
│   ├── evaluation.py            # Recall@k computation from JSON log
│   ├── visualize.py             # Charts, histograms, CSV export
│   ├── ablation.py              # Ablation test set construction
│   └── ablation_visualize.py    # Ablation-specific visualizations
├── results/                     # Output directory (generated)
├── reports/                     # Experiment reports
├── final-report/                # LaTeX report
├── run_benchmark.py             # Main benchmark script
├── run_ablation.py              # Ablation experiment script
├── requirements.txt
└── README.md
```
