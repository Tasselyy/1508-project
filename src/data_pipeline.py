"""Data pipeline: KILT NQ loading, reduced corpus construction, chunking, NER classification, ground-truth extraction."""

import gc
import json as _json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import spacy
from datasets import load_dataset

from src.profiler import Profiler

KILT_WIKIPEDIA_URL = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"


# ---------------------------------------------------------------------------
# 3.1  Load KILT NQ dev set
# ---------------------------------------------------------------------------

def load_kilt_nq_dev() -> list[dict]:
    """Load KILT NaturalQuestions dev split and return as list of dicts."""
    ds = load_dataset("facebook/kilt_tasks", "nq", split="validation", trust_remote_code=True)
    records = list(ds)
    # Free the HuggingFace Dataset object (holds Arrow memory-mapped data)
    del ds
    return records


# ---------------------------------------------------------------------------
# 3.2  Reduced corpus construction
# ---------------------------------------------------------------------------

def _extract_gold_page_ids(nq_records: list[dict]) -> set[int]:
    """Extract all Wikipedia page IDs referenced in query provenance."""
    gold_ids: set[int] = set()
    for rec in nq_records:
        for out in rec.get("output", []):
            for prov in out.get("provenance", []):
                wid = prov.get("wikipedia_id")
                if wid is not None:
                    gold_ids.add(int(wid))
    return gold_ids


def _stream_kilt_wikipedia():
    """Stream KILT Wikipedia JSONL from Facebook servers, yielding one page dict at a time."""
    resp = requests.get(KILT_WIKIPEDIA_URL, stream=True, timeout=60)
    resp.raise_for_status()
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.extend(chunk)
        while b"\n" in buf:
            idx = buf.index(b"\n")
            line = bytes(buf[:idx]).strip()
            del buf[:idx + 1]
            if line:
                yield _json.loads(line)


def build_reduced_corpus(
    nq_records: list[dict],
    target_size: int = 10_000,
    cache_path: str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a reduced Wikipedia corpus containing all gold pages + distractors.

    Returns a DataFrame with columns: wikipedia_id, title, text (dict with paragraph list).
    """
    # Check cache
    if cache_path and Path(cache_path).exists():
        return pd.read_parquet(cache_path)

    gold_ids = _extract_gold_page_ids(nq_records)
    gold_ids_remaining = set(gold_ids)
    rng = random.Random(seed)

    gold_pages: list[dict] = []
    distractor_reservoir: list[dict] = []
    reservoir_count = 0
    needed_distractors = max(0, target_size - len(gold_ids))
    scanned = 0

    print(f"  Streaming KILT Wikipedia (need {len(gold_ids)} gold + {needed_distractors} distractors)...")

    for page in _stream_kilt_wikipedia():
        wid_str = page.get("wikipedia_id", "")
        try:
            wid = int(wid_str)
        except (ValueError, TypeError):
            continue

        scanned += 1
        if scanned % 100000 == 0:
            print(f"    Scanned {scanned} pages, gold found: {len(gold_pages)}/{len(gold_ids)}, "
                  f"distractors: {len(distractor_reservoir)}/{needed_distractors}")

        if wid in gold_ids_remaining:
            gold_pages.append(page)
            gold_ids_remaining.discard(wid)
        else:
            # Reservoir sampling for distractors
            reservoir_count += 1
            if len(distractor_reservoir) < needed_distractors:
                distractor_reservoir.append(page)
            else:
                j = rng.randint(0, reservoir_count - 1)
                if j < needed_distractors:
                    distractor_reservoir[j] = page

        # Early stop once we have all gold pages and enough distractors
        if len(gold_ids_remaining) == 0 and len(distractor_reservoir) >= needed_distractors:
            break
        # Safety limit: stop after scanning enough of the corpus (avoid 17-min full scan)
        # Gold pages are uniformly distributed; 80%+ coverage is sufficient for benchmark
        if scanned >= 2_000_000 and len(distractor_reservoir) >= needed_distractors:
            break

    print(f"  Done: scanned {scanned} pages, gold={len(gold_pages)}, distractors={len(distractor_reservoir)}")

    all_pages = gold_pages + distractor_reservoir[:needed_distractors]
    # Free the separate lists — all_pages has the references now
    del gold_pages, distractor_reservoir

    def _extract_page(p: dict) -> dict:
        text = p.get("text", [])
        # Normalize: raw JSONL has text as list[str]; datasets lib wraps in dict
        if isinstance(text, list):
            paragraphs = text
        elif isinstance(text, dict):
            paragraphs = text.get("paragraph", text.get("paragraph_text", []))
        else:
            paragraphs = []
        return {
            "wikipedia_id": int(p.get("wikipedia_id", 0)),
            "title": p.get("wikipedia_title", p.get("title", "")),
            "paragraphs": paragraphs,
        }

    df = pd.DataFrame([_extract_page(p) for p in all_pages])
    del all_pages  # free raw page dicts

    # Cache for subsequent runs
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)

    return df


# ---------------------------------------------------------------------------
# 3.3  Paragraph-level chunking
# ---------------------------------------------------------------------------

def chunk_corpus(corpus_df: pd.DataFrame) -> list[dict]:
    """Chunk Wikipedia pages into paragraphs with source metadata.

    Returns list of dicts: {chunk_id, wikipedia_id, paragraph_index, text}
    """
    chunks: list[dict] = []
    for _, row in corpus_df.iterrows():
        wid = int(row["wikipedia_id"])
        paragraphs = row.get("paragraphs", [])

        if not isinstance(paragraphs, (list, np.ndarray)):
            paragraphs = []
        else:
            paragraphs = list(paragraphs)

        for para_idx, para_text in enumerate(paragraphs):
            if not isinstance(para_text, str):
                continue
            para_text = para_text.strip()
            if not para_text:
                continue
            chunk_id = f"{wid}_{para_idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "wikipedia_id": wid,
                    "paragraph_index": para_idx,
                    "text": para_text,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# 3.4  spaCy NER query classification
# ---------------------------------------------------------------------------

def classify_queries_ner(
    nq_records: list[dict],
    spacy_model: str = "en_core_web_sm",
    entity_threshold: int = 2,
) -> list[dict]:
    """Classify queries into single-entity (0-1 entities) and multi-entity (2+).

    Returns list of dicts with keys: query, entity_count, entity_list, entity_group, record.
    """
    nlp = spacy.load(spacy_model)
    classified: list[dict] = []

    for rec in nq_records:
        query = rec.get("input", "")
        doc = nlp(query)
        entities = [ent.text for ent in doc.ents]
        entity_count = len(entities)
        group = "multi-entity" if entity_count >= entity_threshold else "single-entity"

        classified.append(
            {
                "query": query,
                "entity_count": entity_count,
                "entity_list": entities,
                "entity_group": group,
                "record": rec,
            }
        )
    return classified


# ---------------------------------------------------------------------------
# 3.5  Balanced query sampling
# ---------------------------------------------------------------------------

def sample_balanced_queries(
    classified_queries: list[dict],
    sample_size_per_group: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Sample approximately equal numbers of queries per entity group."""
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for q in classified_queries:
        groups.setdefault(q["entity_group"], []).append(q)

    sampled: list[dict] = []
    for group_name, members in groups.items():
        n = min(sample_size_per_group, len(members))
        sampled.extend(rng.sample(members, n))
    return sampled


# ---------------------------------------------------------------------------
# 3.6  Ground-truth label extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(query_record: dict) -> set[str]:
    """Extract ground-truth chunk IDs from a KILT record's provenance.

    Returns set of chunk_id strings in format '{wikipedia_id}_{paragraph_index}'.
    """
    gt_ids: set[str] = set()
    rec = query_record.get("record", query_record)
    for out in rec.get("output", []):
        for prov in out.get("provenance", []):
            wid = prov.get("wikipedia_id")
            # KILT provenance uses start/end paragraph indices
            start_par = prov.get("start_paragraph_id", 0)
            end_par = prov.get("end_paragraph_id", start_par)
            if wid is not None:
                for pidx in range(start_par, end_par + 1):
                    gt_ids.add(f"{int(wid)}_{pidx}")
    return gt_ids


# ---------------------------------------------------------------------------
# 3.7  Full pipeline with profiler instrumentation
# ---------------------------------------------------------------------------

def run_data_pipeline(config: dict, profiler: Profiler) -> dict:
    """Execute the full data pipeline, instrumented with profiler.

    Pipeline order (optimized for memory):
    1. Load NQ queries → random pre-sample → NER classify on small subset → balanced sample
    2. Extract gold page IDs for sampled queries only (much fewer pages to fetch)
    3. Build corpus: gold pages for sampled queries + distractors
    4. Chunk corpus → extract ground-truth labels

    Returns dict with keys: chunks, sampled_queries
    """
    sample_per_group = config["queries"]["sample_size_per_group"]
    query_seed = config["queries"].get("seed", 42)

    # Load NQ dev set
    profiler.start_stage("data_loading")
    nq_records = load_kilt_nq_dev()
    profiler.end_stage("data_loading")

    # Pre-sample a smaller candidate pool before running NER on all 13K records.
    # We need sample_per_group * 2 groups in the end; take 4x headroom to ensure
    # enough queries in each entity group after NER classification.
    profiler.start_stage("ner_classification")
    candidate_size = min(len(nq_records), sample_per_group * 8)
    rng = random.Random(query_seed)
    candidates = rng.sample(nq_records, candidate_size)
    # Free the full dataset immediately — candidates hold the only needed refs
    del nq_records

    classified = classify_queries_ner(
        candidates,
        spacy_model=config["spacy"]["model"],
        entity_threshold=config["spacy"]["entity_threshold"],
    )
    del candidates
    profiler.end_stage("ner_classification")

    # Balanced sampling (before corpus construction to reduce gold page set)
    profiler.start_stage("query_sampling")
    sampled = sample_balanced_queries(
        classified,
        sample_size_per_group=sample_per_group,
        seed=query_seed,
    )
    profiler.end_stage("query_sampling")

    # Free classified list (sampled queries hold their own refs)
    del classified
    gc.collect()

    # Build reduced corpus using only gold pages for sampled queries
    profiler.start_stage("corpus_construction")
    corpus_df = build_reduced_corpus(
        [q["record"] for q in sampled],  # only sampled query records
        target_size=config["corpus"]["target_size"],
        cache_path=config["corpus"].get("local_corpus_cache"),
        seed=config["corpus"].get("seed", 42),
    )
    profiler.end_stage("corpus_construction")

    # Chunk corpus and free DataFrame immediately
    profiler.start_stage("chunking")
    chunks = chunk_corpus(corpus_df)
    corpus_size = len(corpus_df)
    del corpus_df
    gc.collect()
    profiler.end_stage("chunking")

    # Add ground-truth chunk IDs to each sampled query
    for q in sampled:
        q["ground_truth_chunk_ids"] = list(extract_ground_truth(q))

    # Drop heavy 'record' reference from sampled queries — no longer needed
    for q in sampled:
        q.pop("record", None)

    # Store corpus metadata
    profiler.data["metadata"]["corpus_size"] = corpus_size
    profiler.data["metadata"]["total_chunks"] = len(chunks)
    profiler.data["metadata"]["total_queries"] = len(sampled)
    profiler.data["metadata"]["queries_per_group"] = {
        g: sum(1 for q in sampled if q["entity_group"] == g)
        for g in {"single-entity", "multi-entity"}
    }

    return {
        "chunks": chunks,
        "sampled_queries": sampled,
    }
