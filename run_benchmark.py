"""Run the full ColBERTv2 vs Bi-Encoder benchmark end-to-end."""

import gc
import json

import yaml

from src.profiler import Profiler
from src.data_pipeline import run_data_pipeline
from src.biencoder_retrieval import run_biencoder_retrieval
from src.colbert_retrieval import run_colbert_retrieval
from src.evaluation import run_evaluation
from src.visualize import run_visualization


def main():
    # Load config
    with open("configs/experiment_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded.")

    profiler = Profiler(config=config)

    # === 1. Data Pipeline ===
    print("\n=== 1. Data Pipeline ===")
    pipeline_out = run_data_pipeline(config, profiler)
    corpus_df = pipeline_out["corpus_df"]
    chunks = pipeline_out["chunks"]
    sampled_queries = pipeline_out["sampled_queries"]
    print(f"  Corpus pages: {len(corpus_df)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Sampled queries: {len(sampled_queries)}")
    print(f"    single-entity: {sum(1 for q in sampled_queries if q['entity_group'] == 'single-entity')}")
    print(f"    multi-entity:  {sum(1 for q in sampled_queries if q['entity_group'] == 'multi-entity')}")

    # corpus_df is no longer needed after chunking — free it
    del corpus_df, pipeline_out
    gc.collect()

    # === 2. Bi-Encoder Retrieval ===
    print("\n=== 2. Bi-Encoder Retrieval ===")
    run_biencoder_retrieval(chunks, sampled_queries, config, profiler)
    print("  Bi-encoder retrieval complete.")

    # === 3. ColBERTv2 Retrieval ===
    print("\n=== 3. ColBERTv2 Retrieval ===")
    run_colbert_retrieval(chunks, sampled_queries, config, profiler)
    print("  ColBERTv2 retrieval complete.")

    # === 4. Save JSON Log ===
    print("\n=== 4. Saving JSON Log ===")
    for q in sampled_queries:
        record = {
            "query": q["query"],
            "entity_count": q["entity_count"],
            "entity_list": q["entity_list"],
            "entity_group": q["entity_group"],
            "ground_truth_chunk_ids": q["ground_truth_chunk_ids"],
        }
        for key in ("biencoder_retrieved_ids", "biencoder_recall_at_k", "biencoder_latency_ms",
                     "colbert_retrieved_ids", "colbert_recall_at_k", "colbert_latency_ms"):
            if key in q:
                record[key] = q[key]
        profiler.log_query(record)

    profiler.data["metadata"]["models"] = config["models"]
    profiler.data["metadata"]["k_values"] = config["retrieval"]["k_values"]

    log_path = config["paths"]["json_log"]
    profiler.save(log_path)
    print(f"  JSON log saved to: {log_path}")

    # === 5. Evaluation ===
    print("\n=== 5. Evaluation ===")
    summary = run_evaluation(log_path)
    print(summary.to_string(index=False))

    # === 6. Visualization ===
    print("\n=== 6. Visualization ===")
    output_paths = run_visualization(
        log_path=log_path,
        charts_dir=config["paths"]["charts_dir"],
        csv_path=config["paths"]["csv_output"],
    )
    for name, path in output_paths.items():
        print(f"  {name}: {path}")

    # === 7. Profiling Summary ===
    print("\n=== 7. Profiling Summary ===")
    with open(log_path) as f:
        log = json.load(f)
    for stage, info in log["stages"].items():
        vram = f", VRAM peak: {info['peak_vram_bytes']/1e9:.2f} GB" if info.get("peak_vram_bytes") else ""
        print(f"  {stage}: {info['duration_seconds']:.2f}s, RSS: {info['rss_bytes']/1e9:.2f} GB{vram}")
    for label, size in log["disk_sizes"].items():
        print(f"  {label}: {size/1e6:.1f} MB")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
