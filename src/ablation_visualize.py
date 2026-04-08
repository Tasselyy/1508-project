"""Ablation visualization: line plots, grouped bar charts, summary CSV, and console table."""

import os

import matplotlib.pyplot as plt
import pandas as pd


def _extract_recall_rows(all_results: list[dict]) -> pd.DataFrame:
    """Extract per-retriever, per-k recall values from ablation results into a flat DataFrame."""
    rows = []
    for result in all_results:
        dimension = result["dimension"]
        condition = result["condition"]
        data = result["data"]

        for q in data.get("queries", []):
            for retriever in ("biencoder", "colbert"):
                recall_key = f"{retriever}_recall_at_k"
                if recall_key not in q:
                    continue
                for k_str, recall_val in q[recall_key].items():
                    rows.append({
                        "dimension": dimension,
                        "condition": condition,
                        "retriever": retriever,
                        "k": int(k_str),
                        "recall": recall_val,
                    })

    return pd.DataFrame(rows)


def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean recall per (dimension, condition, retriever, k) and add delta column."""
    summary = (
        df.groupby(["dimension", "condition", "retriever", "k"])["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "mean_recall"})
    )

    # Add delta (ColBERT - biencoder) per (dimension, condition, k)
    pivot = summary.pivot_table(
        index=["dimension", "condition", "k"],
        columns="retriever",
        values="mean_recall",
    ).reset_index()
    pivot.columns.name = None

    if "biencoder" in pivot.columns and "colbert" in pivot.columns:
        pivot["delta"] = pivot["colbert"] - pivot["biencoder"]
    else:
        pivot["delta"] = 0.0

    # Melt back to long format with delta
    rows = []
    for _, row in pivot.iterrows():
        for retriever in ("biencoder", "colbert"):
            if retriever in pivot.columns:
                rows.append({
                    "dimension": row["dimension"],
                    "condition": row["condition"],
                    "retriever": retriever,
                    "k": int(row["k"]),
                    "recall": row.get(retriever, 0.0),
                    "delta": row["delta"],
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Line plot for numeric dimensions (corpus_scale, top_k)
# ---------------------------------------------------------------------------

def plot_line(df: pd.DataFrame, dimension: str, output_dir: str, k_values: list[int] | None = None) -> str:
    """Line plot: x = condition (numeric), y = mean recall, lines per retriever x k."""
    subset = df[df["dimension"] == dimension].copy()
    if subset.empty:
        return ""

    subset["condition_num"] = pd.to_numeric(subset["condition"], errors="coerce")
    subset = subset.dropna(subset=["condition_num"])

    # Aggregate to mean recall per (condition, retriever, k)
    agg = subset.groupby(["condition_num", "retriever", "k"])["recall"].mean().reset_index()

    if k_values is None:
        k_values = sorted(agg["k"].unique())
    k_show = [k for k in [1, 5, 10] if k in k_values] or k_values[:3]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"biencoder": "#4c72b0", "colbert": "#dd8452"}
    styles = {k_show[i]: ls for i, ls in enumerate(["-", "--", ":"][:len(k_show)])}

    for retriever in ("biencoder", "colbert"):
        for k in k_show:
            data = agg[(agg["retriever"] == retriever) & (agg["k"] == k)].sort_values("condition_num")
            if data.empty:
                continue
            label_name = "Bi-Encoder" if retriever == "biencoder" else "ColBERTv2"
            ax.plot(
                data["condition_num"], data["recall"],
                marker="o", color=colors[retriever], linestyle=styles.get(k, "-"),
                label=f"{label_name} Recall@{k}", linewidth=2, markersize=6,
            )

    x_label = {"corpus_scale": "Corpus Size (pages)", "top_k": "k"}.get(dimension, dimension)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean Recall")
    title = {"corpus_scale": "Corpus Scale Ablation", "top_k": "Top-k Depth Ablation"}.get(dimension, dimension)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = os.path.join(output_dir, f"{dimension}_recall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_top_k_line(df: pd.DataFrame, output_dir: str) -> str:
    """Line plot for top-k ablation: x = k value, y = mean recall, one line per retriever."""
    subset = df[df["dimension"] == "top_k"].copy()
    if subset.empty:
        return ""

    # Aggregate to mean recall per (retriever, k)
    agg = subset.groupby(["retriever", "k"])["recall"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"biencoder": "#4c72b0", "colbert": "#dd8452"}

    for retriever in ("biencoder", "colbert"):
        data = agg[agg["retriever"] == retriever].sort_values("k")
        if data.empty:
            continue
        label_name = "Bi-Encoder" if retriever == "biencoder" else "ColBERTv2"
        ax.plot(data["k"], data["recall"], marker="o", color=colors[retriever],
                label=label_name, linewidth=2, markersize=6)

    ax.set_xlabel("k")
    ax.set_ylabel("Mean Recall@k")
    ax.set_title("Top-k Depth Ablation")
    ax.legend(fontsize=10, loc="best")
    ax.set_ylim(bottom=0)
    ax.set_xticks(sorted(agg["k"].unique()))
    fig.tight_layout()

    path = os.path.join(output_dir, "top_k_recall.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Grouped bar chart for categorical dimensions (chunk_granularity)
# ---------------------------------------------------------------------------

def plot_grouped_bar(df: pd.DataFrame, dimension: str, output_dir: str) -> str:
    """Grouped bar chart: x = condition (categorical), bars per retriever x k."""
    subset = df[df["dimension"] == dimension].copy()
    if subset.empty:
        return ""

    # Aggregate to mean recall
    agg = subset.groupby(["condition", "retriever", "k"])["recall"].mean().reset_index()

    k_show = [k for k in [1, 5, 10] if k in agg["k"].unique()] or sorted(agg["k"].unique())[:3]
    agg = agg[agg["k"].isin(k_show)]

    agg["label"] = agg["retriever"].map({"biencoder": "Bi-Encoder", "colbert": "ColBERTv2"}) + " @" + agg["k"].astype(str)

    # Order conditions sensibly
    condition_order = ["paragraph", "fixed-256", "fixed-512"]
    agg["condition"] = pd.Categorical(agg["condition"], categories=[c for c in condition_order if c in agg["condition"].unique()], ordered=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    import seaborn as sns
    sns.barplot(data=agg, x="condition", y="recall", hue="label", ax=ax)

    ax.set_xlabel("Chunking Strategy")
    ax.set_ylabel("Mean Recall")
    ax.set_title("Chunk Granularity Ablation")
    ax.legend(fontsize=8, title="Retriever @k", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, min(max(agg["recall"].max() * 1.2, 0.1), 1.0))
    fig.tight_layout()

    path = os.path.join(output_dir, f"{dimension}_recall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Summary CSV and console table
# ---------------------------------------------------------------------------

def export_ablation_csv(summary_df: pd.DataFrame, output_path: str) -> str:
    """Export ablation summary to CSV, sorted by dimension/condition/k."""
    summary_df = summary_df.sort_values(["dimension", "condition", "k", "retriever"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    return output_path


def print_summary_table(summary_df: pd.DataFrame) -> None:
    """Print a formatted summary table to stdout."""
    # Pivot for readability: one row per (dimension, condition, k)
    pivot = summary_df.pivot_table(
        index=["dimension", "condition"],
        columns=["retriever", "k"],
        values="recall",
    )

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)

    for dimension in summary_df["dimension"].unique():
        dim_data = summary_df[summary_df["dimension"] == dimension]
        k_values = sorted(dim_data["k"].unique())
        k_show = [k for k in [1, 5, 10] if k in k_values] or k_values[:3]

        print(f"\n--- {dimension} ---")
        header = f"{'Condition':<15} {'Retriever':<12}"
        for k in k_show:
            header += f" {'R@' + str(k):>8}"
        print(header)
        print("-" * len(header))

        for condition in sorted(dim_data["condition"].unique()):
            for retriever in ("biencoder", "colbert"):
                row_data = dim_data[(dim_data["condition"] == condition) & (dim_data["retriever"] == retriever)]
                label = "Bi-Encoder" if retriever == "biencoder" else "ColBERTv2"
                line = f"{condition:<15} {label:<12}"
                for k in k_show:
                    val = row_data[row_data["k"] == k]["recall"]
                    line += f" {val.values[0]:>8.4f}" if len(val) > 0 else f" {'N/A':>8}"
                print(line)

    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ablation_visualization(all_results: list[dict], config: dict) -> dict:
    """Generate all ablation charts, CSV, and console output."""
    charts_dir = config["paths"]["charts_dir"]
    csv_path = config["paths"]["csv_output"]
    os.makedirs(charts_dir, exist_ok=True)

    # Extract and summarize
    raw_df = _extract_recall_rows(all_results)
    summary_df = _compute_summary(raw_df)

    output_paths = {}

    # Line plot for corpus_scale
    path = plot_line(raw_df, "corpus_scale", charts_dir)
    if path:
        output_paths["corpus_scale_chart"] = path
        print(f"  Chart saved: {path}")

    # Line plot for top_k (uses k as x-axis, not condition)
    path = plot_top_k_line(raw_df, charts_dir)
    if path:
        output_paths["top_k_chart"] = path
        print(f"  Chart saved: {path}")

    # Bar chart for categorical dimensions
    path = plot_grouped_bar(raw_df, "chunk_granularity", charts_dir)
    if path:
        output_paths["chunk_granularity_chart"] = path
        print(f"  Chart saved: {path}")

    # CSV
    csv_out = export_ablation_csv(summary_df, csv_path)
    output_paths["summary_csv"] = csv_out
    print(f"  CSV saved: {csv_out}")

    # Console table
    print_summary_table(summary_df)

    return output_paths
