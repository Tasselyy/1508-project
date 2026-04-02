"""Evaluation: compute Recall@k from the JSON log, per retriever per query group, with summary statistics."""

import json

import pandas as pd


def load_log(path: str) -> dict:
    """Load the benchmark JSON log file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 6.2  Recall@k per retriever per query group
# ---------------------------------------------------------------------------

def compute_grouped_recall(log_data: dict) -> pd.DataFrame:
    """Compute mean Recall@k per retriever per query group.

    Returns a DataFrame with columns: retriever, entity_group, k, mean_recall.
    """
    rows = []
    for q in log_data["queries"]:
        group = q["entity_group"]
        for retriever in ("biencoder", "colbert"):
            recall_key = f"{retriever}_recall_at_k"
            if recall_key not in q:
                continue
            for k_str, recall_val in q[recall_key].items():
                rows.append(
                    {
                        "retriever": retriever,
                        "entity_group": group,
                        "k": int(k_str),
                        "recall": recall_val,
                    }
                )

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["retriever", "entity_group", "k"])["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "mean_recall"})
    )
    return grouped


# ---------------------------------------------------------------------------
# 6.3  Overall (ungrouped) Recall@k per retriever
# ---------------------------------------------------------------------------

def compute_overall_recall(log_data: dict) -> pd.DataFrame:
    """Compute mean Recall@k per retriever across all queries (ungrouped)."""
    rows = []
    for q in log_data["queries"]:
        for retriever in ("biencoder", "colbert"):
            recall_key = f"{retriever}_recall_at_k"
            if recall_key not in q:
                continue
            for k_str, recall_val in q[recall_key].items():
                rows.append(
                    {
                        "retriever": retriever,
                        "k": int(k_str),
                        "recall": recall_val,
                    }
                )

    df = pd.DataFrame(rows)
    overall = (
        df.groupby(["retriever", "k"])["recall"]
        .mean()
        .reset_index()
        .rename(columns={"recall": "mean_recall"})
    )
    overall["entity_group"] = "overall"
    return overall


# ---------------------------------------------------------------------------
# 6.4  Summary statistics table with delta columns
# ---------------------------------------------------------------------------

def generate_summary_table(log_data: dict) -> pd.DataFrame:
    """Generate a summary table with mean Recall@k per retriever/group and delta columns.

    Returns DataFrame with columns:
        entity_group, k, biencoder_recall, colbert_recall, delta
    """
    grouped = compute_grouped_recall(log_data)
    overall = compute_overall_recall(log_data)
    combined = pd.concat([grouped, overall], ignore_index=True)

    # Pivot to get biencoder and colbert side by side
    pivot = combined.pivot_table(
        index=["entity_group", "k"],
        columns="retriever",
        values="mean_recall",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(
        columns={"biencoder": "biencoder_recall", "colbert": "colbert_recall"}
    )
    pivot["delta"] = pivot["colbert_recall"] - pivot["biencoder_recall"]

    # Sort for presentation
    group_order = {"single-entity": 0, "multi-entity": 1, "overall": 2}
    pivot["_sort"] = pivot["entity_group"].map(group_order)
    pivot = pivot.sort_values(["_sort", "k"]).drop(columns="_sort").reset_index(drop=True)

    return pivot


def run_evaluation(log_path: str) -> pd.DataFrame:
    """Load log and produce the full summary table."""
    log_data = load_log(log_path)
    return generate_summary_table(log_data)
