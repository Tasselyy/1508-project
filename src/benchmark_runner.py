"""Reusable benchmark runner utilities for single and multi-config experiments."""

import gc
import json
from copy import deepcopy
from pathlib import Path

import yaml

from src.biencoder_retrieval import run_biencoder_retrieval
from src.colbert_retrieval import run_colbert_retrieval
from src.data_pipeline import run_data_pipeline
from src.evaluation import run_evaluation
from src.profiler import Profiler
from src.visualize import run_visualization


def load_config(path: str = "configs/experiment_config.yaml") -> dict:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge nested dictionaries without mutating the input."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _log_query_results(sampled_queries: list[dict], profiler: Profiler, config: dict) -> None:
    for q in sampled_queries:
        record = {
            "query": q["query"],
            "entity_count": q["entity_count"],
            "entity_list": q["entity_list"],
            "entity_group": q["entity_group"],
            "ground_truth_chunk_ids": q["ground_truth_chunk_ids"],
        }
        for key in (
            "biencoder_retrieved_ids",
            "biencoder_recall_at_k",
            "biencoder_latency_ms",
            "colbert_retrieved_ids",
            "colbert_recall_at_k",
            "colbert_latency_ms",
        ):
            if key in q:
                record[key] = q[key]
        profiler.log_query(record)

    profiler.data["metadata"]["models"] = config["models"]
    profiler.data["metadata"]["k_values"] = config["retrieval"]["k_values"]


def run_full_benchmark(
    config: dict,
    save_outputs: bool = True,
    generate_visualizations: bool = True,
) -> dict:
    """Run the full benchmark and optionally persist outputs."""
    profiler = Profiler(config=config)

    pipeline_out = run_data_pipeline(config, profiler)
    chunks = pipeline_out["chunks"]
    sampled_queries = pipeline_out["sampled_queries"]
    del pipeline_out
    gc.collect()

    run_biencoder_retrieval(chunks, sampled_queries, config, profiler)
    run_colbert_retrieval(chunks, sampled_queries, config, profiler)

    _log_query_results(sampled_queries, profiler, config)

    log_path = config["paths"]["json_log"]
    summary = None
    output_paths = {}

    if save_outputs:
        profiler.save(log_path)
        summary = run_evaluation(log_path)
        if generate_visualizations:
            output_paths = run_visualization(
                log_path=log_path,
                charts_dir=config["paths"]["charts_dir"],
                csv_path=config["paths"]["csv_output"],
            )
    else:
        summary = run_evaluation_from_data(profiler.data)

    return {
        "config": config,
        "profiler_data": profiler.data,
        "summary": summary,
        "log_path": log_path,
        "output_paths": output_paths,
        "chunks": chunks,
        "sampled_queries": sampled_queries,
    }


def run_evaluation_from_data(log_data: dict):
    """Evaluate from in-memory profiler data without touching disk."""
    temp_path = Path("results") / "_temp_eval_log.json"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
    return run_evaluation(str(temp_path))
