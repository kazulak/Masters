from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path


EVIDENCE_ARTIFACT_KIND = "evidence_run"
COMPARISON_ARTIFACT_KIND = "comparison_report"
LEGACY_ARTIFACT_KIND = "legacy_run"

ROUTE_LABELS = {
    "cpu_tn_einsum_exact": "cpu_tn_exact",
    "quest_cpu_full_state_exact": "quest_cpu_exact",
    "quest_gpu_full_state_exact": "quest_gpu_exact",
    "quimb_tn_exact": "quimb_tn_exact",
    "upmem_tn_runtime": "upmem_tn_runtime",
    "upmem_tn_sdk_simulator_quantized": "upmem_generic_int8",
}

# Evidence runs should not contain empty report/debug scaffolding. Writers create
# raw, validation, metrics, and plot directories only when they emit real files.
STANDARD_RUN_SUBDIRS = ("config", "cases")


def create_run_dir(
    root_dir: Path,
    suite_id: str,
    *,
    artifact_kind: str = LEGACY_ARTIFACT_KIND,
    route_label: str | None = None,
    comparison_type: str | None = None,
) -> Path:
    runs_dir = root_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if artifact_kind == EVIDENCE_ARTIFACT_KIND:
        if not route_label:
            raise ValueError("evidence runs require a route_label")
        parent = runs_dir / "evidence" / sanitize(suite_id) / sanitize(route_label)
        base = stamp
    elif artifact_kind == COMPARISON_ARTIFACT_KIND:
        if not comparison_type:
            raise ValueError("comparison reports require a comparison_type")
        parent = runs_dir / "comparisons" / sanitize(suite_id) / sanitize(comparison_type)
        base = stamp
    elif artifact_kind == LEGACY_ARTIFACT_KIND:
        parent = runs_dir
        base = f"{stamp}_{sanitize(suite_id)}"
    else:
        raise ValueError(f"unsupported artifact kind: {artifact_kind}")
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / base
    suffix = 1
    while run_dir.exists():
        run_dir = parent / f"{base}_{suffix:02d}"
        suffix += 1
    for rel in STANDARD_RUN_SUBDIRS:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
    update_latest_symlink(parent, run_dir)
    if artifact_kind != LEGACY_ARTIFACT_KIND:
        update_latest_symlink(parent.parent, run_dir)
    if artifact_kind != COMPARISON_ARTIFACT_KIND:
        update_latest_symlink(runs_dir, run_dir)
    return run_dir


def update_latest_symlink(runs_dir: Path, run_dir: Path) -> None:
    latest = runs_dir / "latest"
    try:
        if latest.is_symlink():
            latest.unlink()
        elif latest.exists():
            return
        latest.symlink_to(os.path.relpath(run_dir, runs_dir))
    except OSError:
        return


def resolve_run_dir(path: Path) -> Path:
    return path.resolve()


def route_label(route_id: str) -> str:
    return sanitize(ROUTE_LABELS.get(route_id, route_id))


def route_label_from_routes(routes: list[str] | tuple[str, ...]) -> str:
    if len(routes) == 1:
        return route_label(str(routes[0]))
    return "multi_route"


def is_within_evidence_root(path: Path, root_dir: Path) -> bool:
    try:
        path.resolve().relative_to((root_dir / "runs" / "evidence").resolve())
        return True
    except ValueError:
        return False


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "suite"
