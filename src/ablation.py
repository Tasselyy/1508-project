"""Ablation test set construction: reuse existing cached corpus for fast, low-memory ablation experiments."""

import gc
import random

import pandas as pd

from src.data_pipeline import (
    chunk_corpus,
    chunk_corpus_by_strategy,
    classify_queries_ner,
    extract_ground_truth,
    load_kilt_nq_dev,
    sample_balanced_queries,
)


def build_ablation_test_set(config: dict) -> dict:
    """Construct a small deterministic test set for ablation experiments.

    Strategy: reuse the main benchmark's cached corpus (data/corpus_cache.parquet)
    and sample a small number of queries whose gold pages exist in that corpus.
    This avoids any Wikipedia streaming, keeping RAM < 2 GB.

    Args:
        config: Ablation config dict with keys 'test_set', 'spacy', etc.

    Returns:
        dict with keys:
            - sampled_queries: list[dict] with ground_truth_chunk_ids
            - corpus_df: pd.DataFrame (from cache)
    """
    ts = config["test_set"]
    seed = ts.get("seed", 42)
    queries_per_group = ts.get("queries_per_group", 15)
    main_corpus_cache = ts.get("main_corpus_cache", "data/corpus_cache.parquet")

    # Load existing cached corpus (from the main benchmark run)
    corpus_df = pd.read_parquet(main_corpus_cache)
    corpus_page_ids = set(corpus_df["wikipedia_id"].astype(int).tolist())
    print(f"  Loaded cached corpus: {len(corpus_df)} pages")

    # Chunk the full corpus once to know valid chunk IDs
    all_chunks = chunk_corpus(corpus_df)
    valid_chunk_ids = {c["chunk_id"] for c in all_chunks}
    del all_chunks

    # Load and classify queries
    nq_records = load_kilt_nq_dev()

    # Filter to queries whose gold pages exist in our cached corpus
    eligible = []
    for rec in nq_records:
        gold_ids = set()
        for out in rec.get("output", []):
            for prov in out.get("provenance", []):
                wid = prov.get("wikipedia_id")
                if wid is not None:
                    gold_ids.add(int(wid))
        # Keep only queries where ALL gold pages are in our corpus
        if gold_ids and gold_ids.issubset(corpus_page_ids):
            eligible.append(rec)
    del nq_records
    print(f"  Eligible queries (gold pages in corpus): {len(eligible)}")

    # NER classification on eligible queries only (small set, fast)
    classified = classify_queries_ner(
        eligible,
        spacy_model=config["spacy"]["model"],
        entity_threshold=config["spacy"]["entity_threshold"],
    )
    del eligible

    sampled = sample_balanced_queries(
        classified,
        sample_size_per_group=queries_per_group,
        seed=seed,
    )
    del classified
    gc.collect()

    # Attach ground-truth chunk IDs (only keep those that exist in our corpus chunks)
    for q in sampled:
        raw_gt = extract_ground_truth(q)
        q["ground_truth_chunk_ids"] = [cid for cid in raw_gt if cid in valid_chunk_ids]
        q.pop("record", None)

    return {
        "sampled_queries": sampled,
        "corpus_df": corpus_df,
    }
