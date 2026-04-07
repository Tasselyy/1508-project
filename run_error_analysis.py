"""Compare paragraph vs adaptive chunking results and export query-level deltas."""

from pathlib import Path

from src.error_analysis import compare_variant_logs, save_comparison_outputs, summarize_comparison


def main():
    baseline_log = "results/ablation/paragraph/benchmark_log.json"
    candidate_log = "results/ablation/adaptive_sentence/benchmark_log.json"
    output_dir = Path("results") / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    for retriever in ("biencoder", "colbert"):
        comparison = compare_variant_logs(
            baseline_log_path=baseline_log,
            candidate_log_path=candidate_log,
            retriever=retriever,
            k=10,
        )
        saved = save_comparison_outputs(comparison, str(output_dir), f"{retriever}_paragraph_vs_adaptive_k10")
        print(f"\n=== {retriever.upper()} paragraph vs adaptive @10 ===")
        print(summarize_comparison(comparison).to_string(index=False))
        print("\nTop improvements:")
        print(comparison.head(5)[["query", "entity_group", "baseline_recall", "candidate_recall", "delta"]].to_string(index=False))
        print("\nTop regressions:")
        print(comparison.tail(5)[["query", "entity_group", "baseline_recall", "candidate_recall", "delta"]].to_string(index=False))
        print(f"\nSaved: {saved['details_csv']}")
        print(f"Saved: {saved['summary_csv']}")


if __name__ == "__main__":
    main()
