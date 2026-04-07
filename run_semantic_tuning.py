"""Run a lightweight parameter sweep for the semantic-similarity chunker."""

import json
from pathlib import Path

import pandas as pd

from src.benchmark_runner import deep_update, load_config, run_full_benchmark


def build_variant_config(base_config: dict, variant_name: str, threshold: float, min_words: int, max_words: int) -> dict:
    config = deep_update(
        base_config,
        {
            "chunking": {
                "strategy": "semantic_similarity",
                "semantic_similarity_threshold": threshold,
                "semantic_min_words": min_words,
                "semantic_max_words": max_words,
            }
        },
    )

    result_root = Path(base_config["paths"]["results_dir"]) / "semantic_tuning" / variant_name
    result_root.mkdir(parents=True, exist_ok=True)

    config["paths"]["results_dir"] = str(result_root)
    config["paths"]["faiss_index_dir"] = str(result_root / "faiss_index")
    config["paths"]["colbert_index_dir"] = str(result_root / "colbert_index")
    config["paths"]["json_log"] = str(result_root / "benchmark_log.json")
    config["paths"]["charts_dir"] = str(result_root / "charts")
    config["paths"]["csv_output"] = str(result_root / "summary_statistics.csv")

    cache_file = Path(base_config["corpus"]["local_corpus_cache"])
    config["corpus"]["local_corpus_cache"] = str(
        cache_file.with_name(f"{cache_file.stem}_{variant_name}{cache_file.suffix}")
    )
    return config


def summarize_variant(result: dict, variant_name: str) -> dict:
    summary = result["summary"]
    overall = summary[summary["entity_group"] == "overall"].copy()
    row = {
        "variant": variant_name,
        "threshold": result["config"]["chunking"]["semantic_similarity_threshold"],
        "semantic_min_words": result["config"]["chunking"]["semantic_min_words"],
        "semantic_max_words": result["config"]["chunking"]["semantic_max_words"],
        "total_chunks": result["profiler_data"]["metadata"]["total_chunks"],
        "faiss_index_mb": round(result["profiler_data"]["disk_sizes"].get("faiss_index", 0) / (1024 * 1024), 3),
        "colbert_index_mb": round(result["profiler_data"]["disk_sizes"].get("colbert_index", 0) / (1024 * 1024), 3),
        "log_path": result["log_path"],
    }
    for _, metric_row in overall.iterrows():
        k = int(metric_row["k"])
        row[f"biencoder_recall_at_{k}"] = metric_row["biencoder_recall"]
        row[f"colbert_recall_at_{k}"] = metric_row["colbert_recall"]
        row[f"delta_at_{k}"] = metric_row["delta"]
    return row


def main():
    base_config = load_config("configs/quick_ablation.yaml")
    variants = [
        ("semantic_t072_w60_120", 0.72, 60, 120),
        ("semantic_t070_w60_140", 0.70, 60, 140),
        ("semantic_t068_w60_160", 0.68, 60, 160),
    ]

    rows = []
    manifest = {}
    for variant_name, threshold, min_words, max_words in variants:
        print(f"\n=== Running semantic variant: {variant_name} ===")
        config = build_variant_config(base_config, variant_name, threshold, min_words, max_words)
        result = run_full_benchmark(config)
        rows.append(summarize_variant(result, variant_name))
        manifest[variant_name] = {
            "config": result["config"],
            "log_path": result["log_path"],
            "charts": result["output_paths"],
        }

    output_dir = Path(base_config["paths"]["results_dir"]) / "semantic_tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(rows)
    summary_csv = output_dir / "semantic_tuning_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    manifest_path = output_dir / "semantic_tuning_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Semantic Tuning Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to: {summary_csv}")
    print(f"Saved manifest JSON to: {manifest_path}")


if __name__ == "__main__":
    main()
