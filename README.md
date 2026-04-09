# Retrieval Benchmark Project

This project studies retrieval quality for RAG under different chunking strategies and retrieval architectures.
The current workflow is centered on a single notebook:

- [run_benchmark.ipynb](/C:/Users/Cosmo/Documents/GitHub/1508-project/notebooks/run_benchmark.ipynb)

The notebook runs the project step by step:

- data pipeline
- paragraph benchmark
- sentence-window benchmark
- adaptive-sentence benchmark
- semantic-similarity benchmark
- enhanced adaptive-sentence benchmark
- error analysis
- lightweight knowledge graph prototype
- summary comparison charts

## Recommended Workflow

Use the notebook as the main entry point and execute sections one at a time.
Do not run the whole notebook blindly on a small local machine.

Recommended order:

1. `Setup`
2. `Data Pipeline`
3. Paragraph benchmark
4. The additional strategy sections you actually want
5. Summary comparison
6. Error analysis / knowledge graph sections only when needed

## Config Files

- [quick_ablation.yaml](/C:/Users/Cosmo/Documents/GitHub/1508-project/configs/quick_ablation.yaml)
  Used for smaller, faster local runs.
- [experiment_config.yaml](/C:/Users/Cosmo/Documents/GitHub/1508-project/configs/experiment_config.yaml)
  Used for larger runs, especially on a stronger GPU machine such as Runpod.

## Environment Setup

Requirements:

- Python 3.10+
- NVIDIA GPU recommended for ColBERT indexing
- `en_core_web_sm` spaCy model

Install:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Data and Outputs

Data is downloaded automatically on first run:

- KILT NaturalQuestions validation set
- streamed KILT Wikipedia source

Important directories:

- `data/`
  local corpus caches
- `results/`
  generated logs, CSV summaries, charts, indexes

These are runtime artifacts and can be regenerated.

## Project Structure

```text
configs/
  experiment_config.yaml
  quick_ablation.yaml

notebooks/
  run_benchmark.ipynb

src/
  benchmark_runner.py
  biencoder_retrieval.py
  colbert_retrieval.py
  data_pipeline.py
  error_analysis.py
  evaluation.py
  knowledge_graph.py
  profiler.py
  visualize.py

reports/
  report_en.md
  report_zh.md

requirements.txt
README.md
```

## Notes

- Local machines can hit memory limits during ColBERT indexing for heavier variants.
- For large runs, prefer a stronger GPU machine and run notebook sections individually.
- The notebook writes each strategy run to its own folder under `results/notebook/<variant>/`.
