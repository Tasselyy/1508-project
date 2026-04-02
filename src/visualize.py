"""Visualization: bar charts, histograms, latency plots, index size comparison, CSV export from JSON log."""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.evaluation import compute_grouped_recall, compute_overall_recall, generate_summary_table


def load_log(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 7.2  Grouped bar chart: Recall@k (retriever x entity group x k)
# ---------------------------------------------------------------------------

def plot_recall_bar_chart(log_data: dict, output_dir: str) -> str:
    """Create grouped bar chart of Recall@k by retriever and entity group."""
    grouped = compute_grouped_recall(log_data)
    overall = compute_overall_recall(log_data)
    df = pd.concat([grouped, overall], ignore_index=True)

    # Create a combined label for hue
    df["label"] = df["retriever"] + " / " + df["entity_group"]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x="k", y="mean_recall", hue="label", ax=ax)
    ax.set_xlabel("k")
    ax.set_ylabel("Mean Recall@k")
    ax.set_title("Recall@k: Bi-Encoder vs ColBERTv2 by Query Entity Group")
    ax.legend(title="Retriever / Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1)
    fig.tight_layout()

    path = os.path.join(output_dir, "recall_bar_chart.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7.3  Per-query Recall distribution histograms
# ---------------------------------------------------------------------------

def plot_recall_histograms(log_data: dict, output_dir: str) -> list[str]:
    """Create overlapping per-query Recall distribution histograms for each k."""
    queries = log_data["queries"]
    k_values = sorted({int(k) for q in queries for k in q.get("biencoder_recall_at_k", {})})
    paths = []

    for k in k_values:
        bi_recalls = [q["biencoder_recall_at_k"][str(k)] for q in queries if "biencoder_recall_at_k" in q]
        cb_recalls = [q["colbert_recall_at_k"][str(k)] for q in queries if "colbert_recall_at_k" in q]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(bi_recalls, bins=20, alpha=0.5, label="Bi-Encoder", edgecolor="black")
        ax.hist(cb_recalls, bins=20, alpha=0.5, label="ColBERTv2", edgecolor="black")
        ax.set_xlabel(f"Recall@{k}")
        ax.set_ylabel("Query Count")
        ax.set_title(f"Per-Query Recall@{k} Distribution")
        ax.legend()
        fig.tight_layout()

        path = os.path.join(output_dir, f"recall_histogram_k{k}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# 7.4  Latency comparison chart
# ---------------------------------------------------------------------------

def plot_latency_comparison(log_data: dict, output_dir: str) -> str:
    """Create latency comparison chart showing median + distribution per retriever."""
    queries = log_data["queries"]
    rows = []
    for q in queries:
        if "biencoder_latency_ms" in q:
            rows.append({"retriever": "Bi-Encoder", "latency_ms": q["biencoder_latency_ms"]})
        if "colbert_latency_ms" in q:
            rows.append({"retriever": "ColBERTv2", "latency_ms": q["colbert_latency_ms"]})

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="retriever", y="latency_ms", ax=ax)
    # Overlay median text
    medians = df.groupby("retriever")["latency_ms"].median()
    for i, retriever in enumerate(["Bi-Encoder", "ColBERTv2"]):
        if retriever in medians.index:
            ax.text(i, medians[retriever], f"  {medians[retriever]:.1f}ms", va="center", fontsize=9)

    ax.set_xlabel("Retriever")
    ax.set_ylabel("Per-Query Latency (ms)")
    ax.set_title("Retrieval Latency Comparison")
    fig.tight_layout()

    path = os.path.join(output_dir, "latency_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7.5  Index size comparison bar chart
# ---------------------------------------------------------------------------

def plot_index_size_comparison(log_data: dict, output_dir: str) -> str:
    """Create bar chart comparing FAISS vs ColBERT PLAID index sizes in MB."""
    disk = log_data.get("disk_sizes", {})
    labels = []
    sizes_mb = []
    for label, size_bytes in disk.items():
        labels.append(label.replace("_", " ").title())
        sizes_mb.append(size_bytes / (1024 * 1024))

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, sizes_mb, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("Index Size (MB)")
    ax.set_title("On-Disk Index Size Comparison")
    for bar, mb in zip(bars, sizes_mb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mb:.1f} MB", ha="center", va="bottom")
    fig.tight_layout()

    path = os.path.join(output_dir, "index_size_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7.6  Export summary statistics to CSV
# ---------------------------------------------------------------------------

def export_summary_csv(log_data: dict, output_path: str) -> str:
    """Export summary statistics table to CSV."""
    summary = generate_summary_table(log_data)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# 7.1  Main entry point
# ---------------------------------------------------------------------------

def run_visualization(log_path: str, charts_dir: str, csv_path: str) -> dict:
    """Generate all charts and CSV from the JSON log. Returns dict of output paths."""
    log_data = load_log(log_path)
    os.makedirs(charts_dir, exist_ok=True)

    paths = {
        "recall_bar_chart": plot_recall_bar_chart(log_data, charts_dir),
        "recall_histograms": plot_recall_histograms(log_data, charts_dir),
        "latency_comparison": plot_latency_comparison(log_data, charts_dir),
        "index_size_comparison": plot_index_size_comparison(log_data, charts_dir),
        "summary_csv": export_summary_csv(log_data, csv_path),
    }
    return paths
