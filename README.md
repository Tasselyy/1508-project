# ColBERTv2 vs Bi-Encoder Retrieval Benchmark

A benchmark-driven retrieval project for RAG on the KILT NaturalQuestions dev set, extended with adaptive chunking, semantic chunking, a lightweight learned boundary scorer, and knowledge-graph-assisted hybrid retrieval.

## Setup

### Prerequisites

- Python 3.10+
- GPU recommended for ColBERTv2 indexing
- spaCy English model: `en_core_web_sm`
- Extra disk space for cached corpus files and retrieval indexes

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Data Download

Data is downloaded automatically on first run:

1. **KILT NQ dev set** via HuggingFace `datasets`
2. **KILT Wikipedia** streamed to build a reduced local corpus

The reduced corpus is cached locally so repeated runs do not need to rescan the full streamed source every time.

## Running the Benchmark

The main entry point is the notebook:

```bash
jupyter notebook notebooks/run_benchmark.ipynb
```

Run the notebook section by section rather than executing the full notebook blindly.

The notebook covers:

1. **Data Pipeline**
   - load KILT queries
   - build reduced Wikipedia corpus
   - chunk documents
   - classify queries by entity count
2. **Bi-Encoder Retrieval**
   - encode chunks with `all-MiniLM-L6-v2`
   - build FAISS index
   - retrieve top-k
3. **ColBERTv2 Retrieval**
   - build a ColBERT index with RAGatouille
   - retrieve top-k using late interaction
4. **Save JSON Log**
   - save profiling metadata and per-query results
5. **Evaluation**
   - compute Recall@k from saved outputs
6. **Visualization**
   - generate summary charts and CSV tables
7. **Chunking Variants**
   - sentence window
   - adaptive sentence
   - semantic similarity
8. **Error Analysis**
9. **Knowledge Graph Prototype**
10. **Hybrid Retrieval**
    - graph only
    - bi-encoder + graph
    - ColBERT + graph
    - selective ColBERT + graph fusion
11. **Final Comparison Charts**

## Configuration

Two main configs are provided:

- `configs/quick_ablation.yaml`
  - smaller and faster
  - useful for local validation and chunking comparison
- `configs/experiment_config.yaml`
  - larger benchmark setting
  - better for stronger machines or cloud GPUs

Important knobs include:

| Parameter | Description |
|---|---|
| `corpus.target_size` | number of Wikipedia pages kept in the reduced corpus |
| `queries.sample_size_per_group` | number of sampled queries per entity group |
| `retrieval.k_values` | values of k used for Recall@k |
| `chunking.strategy` | active chunking strategy |
| `chunking.adaptive_min_words` | minimum size for adaptive chunking |
| `chunking.adaptive_max_words` | maximum size for adaptive chunking |
| `chunking.semantic_similarity_threshold` | split threshold for semantic chunking |

## Chunking Strategies

The final notebook compares these retrieval units:

- `paragraph`
  - default benchmark baseline
- `sentence_window`
  - fixed local sentence grouping
- `adaptive_sentence`
  - enhanced adaptive chunker used as the main practical baseline
  - includes the lightweight learned boundary scorer in the final codebase
- `semantic_similarity`
  - embedding-based semantic chunker

### Adaptive Sentence

The adaptive chunker works sentence by sentence inside each paragraph.
It keeps adding sentences until the chunk reaches a target range, then decides whether to stop or continue a little longer.

Main parameters:

- `adaptive_min_words`
  - minimum chunk size before a boundary is allowed
- `adaptive_max_words`
  - maximum chunk size before the chunk must stop
- `adaptive_keyword_slack_words`
  - allows slight extension when adjacent sentences still look locally coherent
- `adaptive_keyword_min_overlap`
  - minimum keyword overlap needed before slack is allowed

In the final codebase, `adaptive_sentence` is the enhanced version used in the final experiments.

### Semantic Similarity

The semantic chunker encodes sentences with a sentence-transformer model and compares the next sentence against the current chunk representation.
If cosine similarity falls below a threshold, a new chunk is started.

Main parameters:

- `semantic_model`
- `semantic_similarity_threshold`
- `semantic_min_words`
- `semantic_max_words`

## Knowledge Graph and Hybrid Retrieval

The project also includes a lightweight knowledge graph prototype:

- entities are extracted from chunks
- entity-to-chunk links are created
- co-occurrence edges are added between entities

This graph can be used:

- as a standalone graph retriever
- as a signal fused with dense retrieval

Hybrid retrieval currently supports:

- `bi-encoder + graph`
- `ColBERT + graph`
- `ColBERT + graph selective`

The selective hybrid only applies graph fusion to multi-entity queries, which is the strongest hybrid story in the final project.

## Output

All outputs are written under `results/`.

Important outputs include:

- `results/benchmark_log.json`
  - main benchmark log for the default run
- `results/summary_statistics.csv`
  - summary table for the default run
- `results/charts/`
  - base charts for the default benchmark
- `results/notebook/<variant>/`
  - per-variant notebook outputs
- `results/notebook/colbert_graph_hybrid/`
  - graph and hybrid retrieval summaries
- `results/notebook/final_comparison_charts/`
  - final comparison figures used for the report

Useful files in the hybrid directory:

- `graph_recall_summary.csv`
- `biencoder_graph_hybrid_summary.csv`
- `colbert_graph_hybrid_summary.csv`
- `colbert_graph_selective_summary.csv`
- `hybrid_compare_overall.csv`

## Hardware Notes

- ColBERT indexing is the heaviest part of the pipeline.
- On smaller local GPUs, some indexing runs may fall back to CPU or require reduced settings.
- For larger runs, a stronger GPU machine is recommended.
- The notebook is designed to be run section by section to make this manageable.

## Project Structure

```text
configs/
  experiment_config.yaml
  quick_ablation.yaml

finalreport/
  final_report.tex
  neurips.sty

notebooks/
  run_benchmark.ipynb

reports/
  report_en.md
  report_zh.md

src/
  benchmark_runner.py
  biencoder_retrieval.py
  colbert_retrieval.py
  data_pipeline.py
  error_analysis.py
  evaluation.py
  hybrid_retrieval.py
  knowledge_graph.py
  learnable_boundary.py
  profiler.py
  visualize.py

results/
  ...

requirements.txt
README.md
```

## Notes

- The notebook is the main workflow and should be treated as the source of truth for the final experiments.
- Some earlier experimental variants were removed from the final workflow to keep the final comparison clean.
- The strongest final chunking baseline is `adaptive_sentence`.
- The strongest graph-based result is the selective ColBERT+graph hybrid.
