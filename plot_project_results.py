"""Create presentation-ready summary charts from current experiment outputs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def plot_chunking_colbert_recall(ablation_df: pd.DataFrame, output_dir: Path) -> str:
    chart_df = ablation_df.melt(
        id_vars=["variant"],
        value_vars=["colbert_recall_at_1", "colbert_recall_at_5", "colbert_recall_at_10", "colbert_recall_at_20"],
        var_name="metric",
        value_name="recall",
    )
    chart_df["k"] = chart_df["metric"].str.extract(r"(\d+)").astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=chart_df, x="k", y="recall", hue="variant", marker="o", ax=ax)
    ax.set_title("ColBERT Recall@k by Chunking Strategy")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, max(0.35, chart_df["recall"].max() + 0.03))
    fig.tight_layout()

    path = output_dir / "chunking_colbert_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_chunking_delta_at_10(ablation_df: pd.DataFrame, output_dir: Path) -> str:
    ranked = ablation_df.sort_values("delta_at_10", ascending=False).copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ranked, x="variant", y="delta_at_10", ax=ax, palette="Blues_d")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("ColBERT Advantage over Bi-Encoder at Recall@10")
    ax.set_xlabel("Chunking Strategy")
    ax.set_ylabel("Delta Recall@10")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    path = output_dir / "chunking_delta_at_10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_chunk_count_vs_recall(ablation_df: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=ablation_df,
        x="total_chunks",
        y="colbert_recall_at_10",
        hue="variant",
        s=120,
        ax=ax,
    )
    for _, row in ablation_df.iterrows():
        ax.text(row["total_chunks"], row["colbert_recall_at_10"], f"  {row['variant']}", va="center")
    ax.set_title("Chunk Count vs ColBERT Recall@10")
    ax.set_xlabel("Total Chunks")
    ax.set_ylabel("Recall@10")
    fig.tight_layout()

    path = output_dir / "chunk_count_vs_colbert_r10.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_semantic_tuning_colbert(semantic_df: pd.DataFrame, output_dir: Path) -> str:
    chart_df = semantic_df.melt(
        id_vars=["variant", "threshold", "semantic_max_words"],
        value_vars=["colbert_recall_at_1", "colbert_recall_at_5", "colbert_recall_at_10", "colbert_recall_at_20"],
        var_name="metric",
        value_name="recall",
    )
    chart_df["k"] = chart_df["metric"].str.extract(r"(\d+)").astype(int)
    chart_df["label"] = chart_df.apply(
        lambda row: f"t={row['threshold']:.2f}, max={int(row['semantic_max_words'])}",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=chart_df, x="k", y="recall", hue="label", marker="o", ax=ax)
    ax.set_title("Semantic Chunker Tuning: ColBERT Recall@k")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, max(0.35, chart_df["recall"].max() + 0.03))
    fig.tight_layout()

    path = output_dir / "semantic_tuning_colbert_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_semantic_tuning_tradeoff(semantic_df: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=semantic_df,
        x="total_chunks",
        y="colbert_recall_at_10",
        hue="threshold",
        size="semantic_max_words",
        sizes=(80, 180),
        palette="viridis",
        ax=ax,
    )
    for _, row in semantic_df.iterrows():
        label = f"t={row['threshold']:.2f}, max={int(row['semantic_max_words'])}"
        ax.text(row["total_chunks"], row["colbert_recall_at_10"], f"  {label}", va="center")
    ax.set_title("Semantic Tuning Trade-off")
    ax.set_xlabel("Total Chunks")
    ax.set_ylabel("ColBERT Recall@10")
    fig.tight_layout()

    path = output_dir / "semantic_tuning_tradeoff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_graph_summary(graph_df: pd.DataFrame, output_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=graph_df[graph_df["entity_group"] != "overall"],
        x="k",
        y="graph_recall",
        hue="entity_group",
        ax=ax,
    )
    ax.set_title("Knowledge Graph Recall by Query Group")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, max(0.2, graph_df["graph_recall"].max() + 0.03))
    fig.tight_layout()

    path = output_dir / "graph_recall_by_group.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_graph_overall(graph_df: pd.DataFrame, output_dir: Path) -> str:
    overall = graph_df[graph_df["entity_group"] == "overall"].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=overall, x="k", y="graph_recall", marker="o", ax=ax, color="#dd8452")
    ax.set_title("Knowledge Graph Overall Recall@k")
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_ylim(0, max(0.15, overall["graph_recall"].max() + 0.03))
    fig.tight_layout()

    path = output_dir / "graph_overall_recall.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main():
    output_dir = Path("results") / "presentation_charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_df = pd.read_csv("results/ablation/chunking_ablation_summary.csv")
    graph_df = pd.read_csv("results/knowledge_graph/graph_recall_summary.csv")
    semantic_df = pd.read_csv("results/semantic_tuning/semantic_tuning_summary.csv")

    paths = {
        "chunking_colbert_recall": plot_chunking_colbert_recall(ablation_df, output_dir),
        "chunking_delta_at_10": plot_chunking_delta_at_10(ablation_df, output_dir),
        "chunk_count_vs_colbert_r10": plot_chunk_count_vs_recall(ablation_df, output_dir),
        "semantic_tuning_colbert_recall": plot_semantic_tuning_colbert(semantic_df, output_dir),
        "semantic_tuning_tradeoff": plot_semantic_tuning_tradeoff(semantic_df, output_dir),
        "graph_recall_by_group": plot_graph_summary(graph_df, output_dir),
        "graph_overall_recall": plot_graph_overall(graph_df, output_dir),
    }

    print("Saved charts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
