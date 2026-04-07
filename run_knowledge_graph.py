"""Build and evaluate a lightweight entity graph retrieval baseline."""

from pathlib import Path

from src.knowledge_graph import (
    build_graph_from_config,
    graph_summary_dataframe,
    run_graph_retrieval,
    save_graph,
)
from src.benchmark_runner import load_config


def main():
    config = load_config("configs/quick_ablation.yaml")
    output_dir = Path("results") / "knowledge_graph"
    output_dir.mkdir(parents=True, exist_ok=True)

    built = build_graph_from_config(config)
    graph = built["graph"]
    sampled_queries = built["sampled_queries"]

    save_graph(graph, str(output_dir / "knowledge_graph.json"))
    run_graph_retrieval(
        graph=graph,
        sampled_queries=sampled_queries,
        k_values=config["retrieval"]["k_values"],
        spacy_model=config["spacy"]["model"],
    )

    summary = graph_summary_dataframe(sampled_queries)
    summary_path = output_dir / "graph_recall_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("Knowledge graph built.")
    print(f"Entities: {graph['metadata']['num_entities']}")
    print(f"Chunks:   {graph['metadata']['num_chunks']}")
    print("\nGraph retrieval summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved graph to:   {output_dir / 'knowledge_graph.json'}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
