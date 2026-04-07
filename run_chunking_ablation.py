"""Run a small chunking ablation over multiple chunking strategies."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.benchmark_runner import deep_update, load_config, run_full_benchmark


def build_variant_config(base_config: dict, variant_name: str, overrides: dict) -> dict:
    """Apply per-variant overrides and route outputs to a dedicated directory."""
    config = deep_update(base_config, overrides)
    result_root = Path(base_config["paths"]["results_dir"]) / "ablation" / variant_name
    result_root.mkdir(parents=True, exist_ok=True)

    config["paths"]["results_dir"] = str(result_root)
    config["paths"]["faiss_index_dir"] = str(result_root / "faiss_index")
    config["paths"]["colbert_index_dir"] = str(result_root / "colbert_index")
    config["paths"]["json_log"] = str(result_root / "benchmark_log.json")
    config["paths"]["charts_dir"] = str(result_root / "charts")
    config["paths"]["csv_output"] = str(result_root / "summary_statistics.csv")

    cache_file = Path(base_config["corpus"]["local_corpus_cache"])
    variant_cache = cache_file.with_name(f"{cache_file.stem}_{variant_name}{cache_file.suffix}")
    config["corpus"]["local_corpus_cache"] = str(variant_cache)
    return config


def summarize_variant(result: dict, variant_name: str) -> dict:
    """Extract the most important metrics from one benchmark run."""
    summary = result["summary"]
    overall = summary[summary["entity_group"] == "overall"].copy()
    profiler_data = result["profiler_data"]
    stages = profiler_data["stages"]
    disk_sizes = profiler_data["disk_sizes"]

    row = {
        "variant": variant_name,
        "chunking_strategy": result["config"]["chunking"]["strategy"],
        "total_chunks": profiler_data["metadata"]["total_chunks"],
        "total_queries": profiler_data["metadata"]["total_queries"],
        "biencoder_retrieval_seconds": stages.get("biencoder_retrieval", {}).get("duration_seconds"),
        "colbert_retrieval_seconds": stages.get("colbert_retrieval", {}).get("duration_seconds"),
        "faiss_index_mb": round(disk_sizes.get("faiss_index", 0) / (1024 * 1024), 3),
        "colbert_index_mb": round(disk_sizes.get("colbert_index", 0) / (1024 * 1024), 3),
        "log_path": result["log_path"],
    }

    for _, metric_row in overall.iterrows():
        k = int(metric_row["k"])
        row[f"biencoder_recall_at_{k}"] = metric_row["biencoder_recall"]
        row[f"colbert_recall_at_{k}"] = metric_row["colbert_recall"]
        row[f"delta_at_{k}"] = metric_row["delta"]
    return row


def main():
    parser = argparse.ArgumentParser(description="Run chunking ablations with a base YAML config.")
    parser.add_argument(
        "--config",
        default="configs/experiment_config.yaml",
        help="Path to the base YAML configuration file.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    variants = {
        "paragraph": {
            "chunking": {
                "strategy": "paragraph",
            }
        },
        "sentence_window": {
            "chunking": {
                "strategy": "sentence_window",
                "sentence_window_size": 3,
                "sentence_window_stride": 2,
            }
        },
        "adaptive_sentence": {
            "chunking": {
                "strategy": "adaptive_sentence",
                "adaptive_min_words": 80,
                "adaptive_max_words": 160,
            }
        },
        "semantic_similarity": {
            "chunking": {
                "strategy": "semantic_similarity",
                "semantic_model": "sentence-transformers/all-MiniLM-L6-v2",
                "semantic_similarity_threshold": 0.72,
                "semantic_min_words": 60,
                "semantic_max_words": 120,
                "semantic_min_sentences": 2,
                "semantic_max_sentences": 5,
            }
        },
    }

    rows = []
    manifest = {}

    for variant_name, overrides in variants.items():
        print(f"\n=== Running variant: {variant_name} ===")
        config = build_variant_config(base_config, variant_name, overrides)
        result = run_full_benchmark(config)
        rows.append(summarize_variant(result, variant_name))
        manifest[variant_name] = {
            "config": result["config"],
            "log_path": result["log_path"],
            "charts": result["output_paths"],
        }

    summary_df = pd.DataFrame(rows)
    output_dir = Path(base_config["paths"]["results_dir"]) / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "chunking_ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    manifest_path = output_dir / "chunking_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Ablation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV to: {summary_csv}")
    print(f"Saved manifest JSON to: {manifest_path}")


if __name__ == "__main__":
    main()
