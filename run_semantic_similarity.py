"""Run only the semantic-similarity chunking variant."""

import argparse
from pathlib import Path

from src.benchmark_runner import deep_update, load_config, run_full_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run only the semantic-similarity chunking benchmark.")
    parser.add_argument(
        "--config",
        default="configs/quick_ablation.yaml",
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional semantic similarity threshold override.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Optional minimum chunk word count override.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Optional maximum chunk word count override.",
    )
    parser.add_argument(
        "--label",
        default="semantic_similarity_single",
        help="Output directory label for this run.",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    overrides = {
        "chunking": {
            "strategy": "semantic_similarity",
        }
    }

    if args.threshold is not None:
        overrides["chunking"]["semantic_similarity_threshold"] = args.threshold
    if args.min_words is not None:
        overrides["chunking"]["semantic_min_words"] = args.min_words
    if args.max_words is not None:
        overrides["chunking"]["semantic_max_words"] = args.max_words

    config = deep_update(base_config, overrides)
    result_root = Path(base_config["paths"]["results_dir"]) / args.label
    result_root.mkdir(parents=True, exist_ok=True)

    config["paths"]["results_dir"] = str(result_root)
    config["paths"]["faiss_index_dir"] = str(result_root / "faiss_index")
    config["paths"]["colbert_index_dir"] = str(result_root / "colbert_index")
    config["paths"]["json_log"] = str(result_root / "benchmark_log.json")
    config["paths"]["charts_dir"] = str(result_root / "charts")
    config["paths"]["csv_output"] = str(result_root / "summary_statistics.csv")

    result = run_full_benchmark(config)
    print("\n=== Semantic Similarity Run Complete ===")
    print(result["summary"].to_string(index=False))
    print(f"\nJSON log: {result['log_path']}")
    for name, path in result["output_paths"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
