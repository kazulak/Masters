"""Public command handlers for the additive M5.5 whole-circuit study lane."""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from quantum_bench.bench.m5_circuit_report import generate_report, load_records
from quantum_bench.bench.m5_circuit_study import (
    apply_rank_path_override,
    load_study_config,
    plan_study,
    run_study,
)


def parse_rank_paths(value: str | None = None) -> list[str]:
    """Return explicit rank paths from CLI or the documented environment."""

    raw = value
    if raw is None:
        raw = os.environ.get("UPMEM_HW_RANK_PATHS")
    if raw is None:
        raw = os.environ.get("UPMEM_HW_RANK_PATH")
    if raw is None:
        raise ValueError(
            "set --rank-paths, UPMEM_HW_RANK_PATHS, or UPMEM_HW_RANK_PATH for physical execution"
        )
    paths = [item.strip() for item in raw.split(",") if item.strip()]
    if not paths:
        raise ValueError("explicit UPMEM rank paths cannot be empty")
    import re

    if any(re.fullmatch(r"/dev/dpu_rank[0-9]+", item) is None for item in paths):
        raise ValueError("rank paths must be explicit /dev/dpu_rankN paths")
    if len(set(paths)) != len(paths):
        raise ValueError("UPMEM rank paths must be unique")
    return paths


def require_physical_environment(rank_paths: str | None = None) -> list[str]:
    """Fail closed before a physical study can construct an engine."""

    if os.environ.get("UPMEM_ALLOW_PHYSICAL_HARDWARE") != "1":
        raise ValueError("UPMEM_ALLOW_PHYSICAL_HARDWARE=1 is required")
    for name in ("DPU_BACKEND", "UPMEM_EXECUTION_MODE"):
        if name in os.environ:
            raise ValueError(f"{name} must be unset for physical M5.5 execution")
    return parse_rank_paths(rank_paths)


def _tasklet_counts(config: Mapping[str, Any]) -> list[int]:
    values = {
        int(variant["topology"]["tasklets_per_device"])
        for variant in config["engine_variants"]
        if variant["topology"]["backend"] != "cpu"
    }
    if not values:
        return []
    if any(value < 1 or value > 24 for value in values):
        raise ValueError("v4 tasklet count must be in [1, 24]")
    return sorted(values)


def build_native_v4(root: Path, config: Mapping[str, Any]) -> list[int]:
    """Build each declared tasklet-keyed v4 host/DPU binary."""

    tasklets = _tasklet_counts(config)
    native = Path(root) / "native" / "upmem" / "simplepim" / "upmem_sdk_execution_plan"
    for count in tasklets:
        completed = subprocess.run(
            ["make", "v4", f"NR_TASKLETS={count}"],
            cwd=native,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"v4 native build failed for NR_TASKLETS={count}")
    return tasklets


def prepare(root: Path, suite: Path, *, build: bool) -> dict[str, Any]:
    """Plan a study, optionally build native v4, and never allocate hardware."""

    config = load_study_config(suite)
    plan_path = plan_study(root, suite)
    tasklets = build_native_v4(root, config) if build else []
    return {
        "plan": str(plan_path),
        "status": "prepared",
        "dpu_allocation_attempted": False,
        "dpu_launch_attempted": False,
        "built_tasklets": tasklets,
    }


def _physical_factory(root: Path):
    from quantum_bench.targets.upmem.m5_whole_circuit_engine import (
        M5WholeCircuitEngine,
    )

    native = root / "native" / "upmem" / "simplepim" / "upmem_sdk_execution_plan"

    def factory(*, topology: Any, engine_variant: Mapping[str, Any], timeout_s: float):
        variant_topology = engine_variant["topology"]
        tasklets = int(topology.tasklets_per_device)
        host_binary = native / "bin" / f"host_upmem_execution_plan_v4_t{tasklets}"
        dpu_binary = native / "bin" / f"dpu_gemm_tile_v4_t{tasklets}"
        initialization_binary = native / "bin" / "dpu_simplepim_management_init"
        for path in (host_binary, dpu_binary, initialization_binary):
            if not path.is_file():
                raise FileNotFoundError(
                    f"required M5.5 native binary is missing: {path}"
                )
        if not os.access(host_binary, os.X_OK):
            raise PermissionError(f"M5.5 host binary is not executable: {host_binary}")
        rank_paths = tuple(str(path) for path in variant_topology["rank_paths"])
        session_root = (
            root
            / "build"
            / "m5_circuit_sessions"
            / f"{engine_variant['id']}-{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        )
        return M5WholeCircuitEngine(
            session_root=session_root,
            host_binary=host_binary,
            dpu_binary=dpu_binary,
            initialization_binary=initialization_binary,
            rank_paths=rank_paths,
            dpu_count=len(topology.device_ids),
            tasklets_per_dpu=tasklets,
            timeout_s=float(timeout_s),
        )

    return factory


def execute(
    root: Path,
    suite: Path,
    *,
    rank_paths: str | None = None,
) -> dict[str, Any]:
    paths = require_physical_environment(rank_paths)
    config = apply_rank_path_override(load_study_config(suite), paths)
    # run_study accepts a path to keep its artifact contract stable.  Write the
    # resolved configuration only through the normal study hook by passing the
    # explicit paths; the original suite remains the source of record.
    factories = {
        variant["id"]: _physical_factory(root)
        for variant in config["engine_variants"]
        if variant["topology"]["backend"] != "cpu"
    }
    run_dir = run_study(
        root,
        suite,
        engine_factories=factories,
        rank_paths=paths,
    )
    summary_path = run_dir / "m5_circuit_study_summary.json"
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "run_dir": str(run_dir),
        "artifact": str(summary_path),
        "status": summary.get("status"),
        "record_count": summary.get("record_count", 0),
    }


def report(
    source: Path, output: Path, *, baselines: tuple[Path, ...] = ()
) -> dict[str, Any]:
    input_path = source / "normalized_records.jsonl" if source.is_dir() else source
    rows = load_records(input_path)
    for baseline in baselines:
        baseline_path = (
            baseline / "normalized_records.jsonl" if baseline.is_dir() else baseline
        )
        rows.extend(load_records(baseline_path))
    result = generate_report(rows, output)
    return {
        "report_dir": str(result.output_dir),
        "status": "completed",
        "plot_count": len(result.manifest.plots),
    }


def baseline_paths(values: list[str] | None = None) -> tuple[Path, ...]:
    """Resolve repeatable CLI baselines or the comma-separated environment."""

    supplied = list(values or [])
    if not supplied:
        supplied = [
            item.strip()
            for item in os.environ.get("M5_CIRCUIT_BASELINES", "").split(",")
            if item.strip()
        ]
    return tuple(Path(item) for item in supplied)


__all__ = [
    "build_native_v4",
    "baseline_paths",
    "execute",
    "parse_rank_paths",
    "prepare",
    "report",
    "require_physical_environment",
]
