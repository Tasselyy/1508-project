"""Lightweight profiler for benchmark stage timing, GPU/RAM/disk measurements, and per-query logging."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch


class Profiler:
    """Accumulates stage timing, GPU VRAM peak, RAM snapshots, disk sizes,
    and per-query results into a dict. Writes everything to a single JSON file."""

    def __init__(self, config: dict | None = None):
        self._process = psutil.Process()
        self._has_cuda = torch.cuda.is_available()
        self._current_stage: str | None = None
        self._stage_start: float | None = None

        self.data: dict = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config or {},
                "gpu_device": torch.cuda.get_device_name(0) if self._has_cuda else None,
                "gpu_total_vram_bytes": (
                    torch.cuda.get_device_properties(0).total_memory if self._has_cuda else None
                ),
            },
            "stages": {},
            "queries": [],
            "disk_sizes": {},
        }

    # ------------------------------------------------------------------
    # Stage timing with GPU/RAM snapshots
    # ------------------------------------------------------------------

    def start_stage(self, name: str) -> None:
        """Begin profiling a named stage. Resets CUDA peak memory counter."""
        self._current_stage = name
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats()
        self._stage_start = time.perf_counter()

    def end_stage(self, name: str | None = None) -> None:
        """End the current (or named) stage and record metrics."""
        stage_name = name or self._current_stage
        if stage_name is None or self._stage_start is None:
            return

        duration = time.perf_counter() - self._stage_start

        self.data["stages"][stage_name] = {
            "duration_seconds": round(duration, 4),
            "peak_vram_bytes": (
                torch.cuda.max_memory_allocated() if self._has_cuda else None
            ),
            "rss_bytes": self._process.memory_info().rss,
        }

        self._current_stage = None
        self._stage_start = None

    # ------------------------------------------------------------------
    # Per-query result logging
    # ------------------------------------------------------------------

    def log_query(self, record: dict) -> None:
        """Append a per-query result record.

        Expected keys (added by caller):
            query, entity_count, entity_list, entity_group,
            ground_truth_chunk_ids,
            <retriever>_retrieved_ids, <retriever>_recall_at_k, <retriever>_latency_ms
        """
        self.data["queries"].append(record)

    # ------------------------------------------------------------------
    # Disk size measurement
    # ------------------------------------------------------------------

    def record_disk_size(self, label: str, path: str) -> None:
        """Measure total on-disk size of a file or directory (recursively)."""
        p = Path(path)
        if p.is_file():
            total = p.stat().st_size
        elif p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        else:
            total = 0
        self.data["disk_sizes"][label] = total

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the full profiling data dict to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)
