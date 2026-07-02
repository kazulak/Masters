from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from quantum_bench.bench.result_artifacts import RESULT_FIELDS, load_result_records
from quantum_bench.core.jsonio import read_jsonl, write_json, write_jsonl
from quantum_bench.core.records import JsonDict, to_jsonable


RUN_MANIFEST_SCHEMA_VERSION = "run_manifest_v1"
ARTIFACT_REFERENCE_SCHEMA_VERSION = "artifact_reference_v1"
ARTIFACT_RETENTION_SCHEMA_VERSION = "artifact_retention_v1"
REPORT_RUN_SCHEMA_VERSION = "report_run_v1"
COMPARE_RUNS_SCHEMA_VERSION = "compare_runs_v1"
NORMALIZED_RECORDS_SCHEMA_VERSION = "normalized_records_v1"

RETENTION_MODES = ("full", "compact")
COMPACT_PRUNE_PATTERNS = (
    "runner_work",
    ".bin",
    "operands",
    "references",
    "outputs",
)

REPORT_RESULT_FIELDS = [
    "case_id",
    "workload_id",
    "route_id",
    "backend_family",
    "benchmark_role",
    "route_role_description",
    "route_limitation_scope",
    "execution_model",
    "output_kind",
    "policy",
    "quantization_mode",
    "kernel_family",
    "status",
    "validation_status",
    "contraction_execution_target",
    "accelerator_kind",
    "gpu_backend_verified",
    "gpu_program_executed",
    "gpu_device_name",
    "gpu_runtime_stack",
    "upmem_execution_mode",
    "execution_backend",
    "hardware_execution",
    "hardware_timing_available",
    "hardware_speedup_applicable",
    "cpu_fallback_used",
    "execution_scope",
    "task_count",
    "validated_task_count",
    "unsupported_task_count",
    "planning_time_s",
    "lowering_time_s",
    "total_wall_time_s",
    "kernel_time_s",
    "setup_time_s",
    "circuit_lowering_time_s",
    "data_transfer_time_s",
    "simulation_compute_time_s",
    "validation_time_s",
    "output_materialization_time_s",
    "timing_scope",
    "gpu_synchronized",
    "validation_method",
    "expected_runtime_class",
    "expected_memory_class",
    "intended_use",
    "max_qubits",
    "manual_invocation_required",
    "expected_risk",
    "known_heavy_backends",
    "resource_guard_status",
    "resource_skip_reason",
    "repeat_id",
    "measured_repeat_count",
    "total_wall_time_s_median",
    "simulation_compute_time_s_median",
    "build_time_s",
    "hardware_speedup",
]

TIMING_FIELDS = [
    "cpu_reference_time_s",
    "upmem_runtime_wall_time_s",
    "host_orchestration_time_s",
    "quantization_time_s",
    "bridge_prepare_time_s",
    "native_build_time_s",
    "dpu_program_wall_time_s",
    "simulation_compute_time_s",
    "setup_time_s",
    "data_transfer_time_s",
    "dequantization_time_s",
    "validation_time_s",
    "output_materialization_time_s",
]

BACKEND_LABELS = {
    "quest_cpu_full_state_exact": "QuEST full-state",
    "quest_gpu_full_state_exact": "QuEST GPU full-state",
    "cpu_tn_einsum_exact": "Internal CPU TN",
    "quimb_tn_exact": "Quimb TN",
    "upmem_tn_runtime": "UPMEM TN runtime",
    "upmem_tn_sdk_simulator_quantized": "UPMEM SDK simulator TN",
}

PLOT_DATA_FIELDS = [
    "case_id",
    "case_family",
    "n_qubits",
    "gate_count",
    "route_id",
    "backend_label",
    "backend_family",
    "benchmark_role",
    "route_role_description",
    "route_limitation_scope",
    "execution_model",
    "contraction_execution_target",
    "accelerator_kind",
    "gpu_backend_verified",
    "gpu_program_executed",
    "gpu_device_name",
    "gpu_runtime_stack",
    "status",
    "validation_status",
    "total_wall_time_s",
    "kernel_time_s",
    "planning_time_s",
    "lowering_time_s",
    "setup_time_s",
    "circuit_lowering_time_s",
    "data_transfer_time_s",
    "simulation_compute_time_s",
    "validation_time_s",
    "output_materialization_time_s",
    "timing_scope",
    "gpu_synchronized",
    "validation_method",
    "expected_runtime_class",
    "expected_memory_class",
    "intended_use",
    "max_qubits",
    "manual_invocation_required",
    "expected_risk",
    "known_heavy_backends",
    "resource_guard_status",
    "resource_skip_reason",
    "repeat_id",
    "measured_repeat_count",
    "total_wall_time_s_median",
    "simulation_compute_time_s_median",
    "max_abs_error",
    "l2_error",
    "probability_l1_error",
    "probability_max_abs_error",
    "memory_proxy_bytes",
    "statevector_bytes",
    "tn_task_count",
    "tn_max_intermediate_bytes",
    "tn_estimated_flops",
    "tn_estimated_bytes",
]

PLOT_SPECS = {
    "runtime_by_backend_case.png": "Runtime by backend and case",
    "runtime_scaling_by_qubits.png": "Runtime scaling by qubit count grouped by circuit family",
    "output_error_by_backend.png": "Output agreement error by backend",
    "probability_error_by_backend.png": "Probability error by backend",
    "memory_proxy_by_backend.png": "Memory proxy by backend",
    "tn_task_and_intermediate_by_backend.png": "TN task count and max intermediate size",
    "planning_vs_contraction_time.png": "Planning/lowering versus contraction time",
    "backend_support_summary.png": "Backend support summary",
    "relative_runtime_vs_quest_anchor.png": "Relative runtime versus QuEST anchor",
    "compute_time_by_backend_case.png": "Compute-focused time by backend and case",
    "total_vs_compute_time.png": "Total wall time versus compute-focused time",
}


@dataclass(frozen=True)
class ReportRunResult:
    run_dir: Path
    report_path: Path
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class PruneRunResult:
    run_dir: Path
    manifest_path: Path
    status: str
    pruned_file_count: int


@dataclass(frozen=True)
class CompareRunsResult:
    run_dir: Path
    artifact_path: Path
    summary_path: Path
    status: str


def validate_retention_mode(mode: str) -> None:
    if mode == "summary-only":
        raise ValueError("summary-only artifact retention is deferred for a later wave")
    if mode not in RETENTION_MODES:
        raise ValueError(f"unsupported artifact retention mode: {mode}")


def write_run_manifest(
    run_dir: Path,
    *,
    run_kind: str,
    suite_id: str | None,
    suite_path: str | None,
    policies: Iterable[str] = (),
    quantization_modes: Iterable[str] = (),
    upmem_execution_mode: str | None = None,
    artifact_retention: str = "full",
    command: str | None = None,
    root_dir: Path | None = None,
) -> JsonDict:
    validate_retention_mode(artifact_retention)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "run_kind": run_kind,
        "timestamp": None,
        "git_commit": _git_commit(root_dir or run_dir),
        "dirty_tree": _git_dirty(root_dir or run_dir),
        "suite_id": suite_id,
        "suite_path": suite_path,
        "policies": tuple(policies),
        "quantization_modes": tuple(quantization_modes),
        "upmem_execution_mode": upmem_execution_mode,
        "artifact_retention": artifact_retention,
        "python_version": platform.python_version(),
        "upmem_sdk_available": "unknown",
        "hardware_available": "not_checked",
        "environment_hash": _environment_hash(),
        "command": command,
        "schema_versions": {
            "run_manifest": RUN_MANIFEST_SCHEMA_VERSION,
            "artifact_reference": ARTIFACT_REFERENCE_SCHEMA_VERSION,
            "artifact_retention": ARTIFACT_RETENTION_SCHEMA_VERSION,
            "report_run": REPORT_RUN_SCHEMA_VERSION,
            "compare_runs": COMPARE_RUNS_SCHEMA_VERSION,
            "normalized_records": NORMALIZED_RECORDS_SCHEMA_VERSION,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return to_jsonable(manifest)


def artifact_ref(run_dir: Path, rel_path: str | Path | None, *, role: str) -> JsonDict | None:
    if rel_path is None:
        return None
    rel = Path(rel_path)
    payload = {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "role": role,
        "relative_path": rel.as_posix(),
        "retained": (run_dir / rel).exists(),
        "status": "retained" if (run_dir / rel).exists() else "missing_unexpectedly",
        "prune_reason": None,
        "metadata": _file_metadata(run_dir / rel),
    }
    return to_jsonable(payload)


def write_normalized_records(run_dir: Path, records: Iterable[JsonDict]) -> Path:
    path = run_dir / "normalized_records.jsonl"
    payloads = []
    for record in records:
        normalized = dict(record)
        normalized.setdefault("normalized_record_schema_version", NORMALIZED_RECORDS_SCHEMA_VERSION)
        payloads.append(to_jsonable(normalized))
    write_jsonl(path, payloads)
    return path


def load_normalized_records(run_dir: Path) -> list[JsonDict]:
    return read_jsonl(run_dir / "normalized_records.jsonl")


def report_run(run_dir: Path, *, output_plots: bool = True) -> ReportRunResult:
    run_dir = run_dir.resolve()
    records = _load_run_records(run_dir)
    if not records:
        raise ValueError("report-run found no normalized benchmark records")
    _write_report_artifacts(run_dir, records, output_plots=output_plots)
    payload = {
        "schema_version": REPORT_RUN_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "status": "completed",
        "record_count": len(records),
        "non_destructive": True,
        "overwrites_derived_reports": True,
        "deletes_execution_artifacts": False,
    }
    report_path = run_dir / "report_run.json"
    write_json(report_path, payload)
    _cleanup_empty_report_dirs(run_dir)
    return ReportRunResult(run_dir=run_dir, report_path=report_path, status="completed")


def prune_run(run_dir: Path, *, artifact_retention: str = "compact") -> PruneRunResult:
    validate_retention_mode(artifact_retention)
    run_dir = run_dir.resolve()
    _require_new_run_layout(run_dir)
    if artifact_retention == "full":
        manifest = _retention_manifest(run_dir, mode="full", pruned=[], retained=_all_files(run_dir))
        write_json(run_dir / "artifact_retention_manifest.json", manifest)
        return PruneRunResult(run_dir, run_dir / "artifact_retention_manifest.json", "completed", 0)
    candidates = _compact_prune_candidates(run_dir)
    pruned_refs: list[JsonDict] = []
    for path in candidates:
        if not path.exists():
            continue
        rel = path.relative_to(run_dir)
        ref = _pruned_reference(run_dir, rel, "compact_retention")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        pruned_refs.append(ref)
    _mark_pruned_references(run_dir, pruned_refs)
    manifest = _retention_manifest(run_dir, mode="compact", pruned=pruned_refs, retained=_all_files(run_dir))
    write_json(run_dir / "artifact_retention_manifest.json", manifest)
    _cleanup_empty_report_dirs(run_dir)
    return PruneRunResult(run_dir, run_dir / "artifact_retention_manifest.json", "completed", len(pruned_refs))


def compare_runs(baseline: Path, candidate: Path, out_dir: Path) -> CompareRunsResult:
    baseline_records = _load_run_records(baseline.resolve())
    candidate_records = _load_run_records(candidate.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    final = _compare_grouped(
        baseline_records,
        candidate_records,
        key_fields=(
            "case_id",
            "route_id",
            "execution_model",
            "policy",
            "quantization_mode",
            "contraction_execution_target",
            "accelerator_kind",
            "upmem_execution_mode",
        ),
    )
    cpu = _compare_grouped(
        baseline_records,
        candidate_records,
        key_fields=("case_id", "contraction_execution_target", "execution_scope"),
        predicate=lambda record: record.get("execution_target") == "cpu",
    )
    kernel = _compare_grouped(
        baseline_records,
        candidate_records,
        key_fields=("case_id", "policy", "quantization_mode", "contraction_execution_target", "accelerator_kind", "upmem_execution_mode", "kernel_family"),
    )
    payload = {
        "schema_version": COMPARE_RUNS_SCHEMA_VERSION,
        "status": "completed",
        "baseline_run": baseline.resolve().name,
        "candidate_run": candidate.resolve().name,
        "final_validation_accuracy_timing": final,
        "cpu_reference": cpu,
        "kernel_family_mix": kernel,
        "metadata": {
            "hardware_speedup_not_inferred_from_sdk_simulator": True,
            "comparison_keys": {
                "final": (
                    "case_id",
                    "route_id",
                    "execution_model",
                    "policy",
                    "quantization_mode",
                    "contraction_execution_target",
                    "accelerator_kind",
                    "upmem_execution_mode",
                ),
                "cpu": ("case_id", "contraction_execution_target", "execution_scope"),
                "kernel_family": (
                    "case_id",
                    "policy",
                    "quantization_mode",
                    "contraction_execution_target",
                    "accelerator_kind",
                    "upmem_execution_mode",
                    "kernel_family",
                ),
            },
        },
    }
    artifact_path = out_dir / "compare_runs.json"
    summary_path = out_dir / "compare_runs_summary.md"
    write_json(artifact_path, payload)
    summary_path.write_text(_compare_runs_markdown(payload), encoding="utf-8")
    return CompareRunsResult(out_dir, artifact_path, summary_path, "completed")


def _load_run_records(run_dir: Path) -> list[JsonDict]:
    normalized = run_dir / "normalized_records.jsonl"
    if normalized.exists():
        return read_jsonl(normalized)
    return load_result_records([run_dir])


def _write_report_artifacts(run_dir: Path, records: list[JsonDict], *, output_plots: bool) -> None:
    plot_tables = _write_plot_source_tables(run_dir, records)
    _write_csv(run_dir / "upmem_mvp_benchmark_results.csv", records, REPORT_RESULT_FIELDS)
    _write_csv(run_dir / "kernel_family_summary.csv", _kernel_family_summary(records), ["kernel_family", "record_count", "task_count", "validated_task_count", "unsupported_task_count"])
    _write_csv(run_dir / "quantization_accuracy_summary.csv", _quantization_rows(records), ["case_id", "policy", "quantization_mode", "validation_status", "max_abs_error", "l2_error"])
    _write_csv(run_dir / "unsupported_reasons.csv", _unsupported_rows(records), ["case_id", "policy", "quantization_mode", "reason", "count"])
    _write_csv(run_dir / "metrics" / "per_task_metrics.csv", _per_task_rows(run_dir), sorted(_per_task_fieldnames(run_dir)))
    _write_csv(run_dir / "metrics" / "per_case_metrics.csv", _per_case_rows(records), ["case_id", "policy", "quantization_mode", "task_count", "validated_task_count", "unsupported_task_count", "status"])
    _write_csv(run_dir / "metrics" / "timing_breakdown.csv", _timing_rows(records), ["case_id", "policy", "quantization_mode", *TIMING_FIELDS, "timing_status"])
    write_json(run_dir / "validation" / "validation_summary.json", _validation_summary(records))
    write_jsonl(run_dir / "validation" / "validation_failures.jsonl", _validation_failures(records))
    if output_plots:
        _write_plots(run_dir, plot_tables)
    else:
        write_json(run_dir / "plots" / "plot_manifest.json", {"schema_version": REPORT_RUN_SCHEMA_VERSION, "status": "skipped", "reason": "plot_generation_disabled"})
    _write_text_atomic(run_dir / "comparison_summary.md", _report_markdown(records, run_dir))


def _write_plot_source_tables(run_dir: Path, records: list[JsonDict]) -> JsonDict:
    data_dir = run_dir / "plots" / "data"
    rows = [_plot_data_row(record) for record in records]
    runtime_rows = [row for row in rows if _positive_float(row.get("total_wall_time_s")) is not None]
    compute_rows = [row for row in rows if _positive_float(row.get("simulation_compute_time_s")) is not None]
    total_vs_compute_rows = [
        row
        for row in rows
        if _positive_float(row.get("total_wall_time_s")) is not None and _positive_float(row.get("simulation_compute_time_s")) is not None
    ]
    error_rows = [row for row in rows if _float_or_none(row.get("max_abs_error")) is not None]
    probability_rows = [row for row in rows if _float_or_none(row.get("probability_l1_error")) is not None]
    memory_rows = [row for row in rows if _positive_float(row.get("memory_proxy_bytes")) is not None]
    tn_rows = [row for row in rows if row.get("execution_model") == "tensor_network"]
    support_rows = _backend_support_rows(rows)
    relative_rows, relative_skipped = _relative_runtime_rows(rows)
    table_specs = {
        "backend_results": (data_dir / "backend_results.csv", rows, PLOT_DATA_FIELDS),
        "runtime_by_backend_case": (data_dir / "runtime_by_backend_case.csv", runtime_rows, PLOT_DATA_FIELDS),
        "runtime_scaling_by_qubits": (data_dir / "runtime_scaling_by_qubits.csv", runtime_rows, PLOT_DATA_FIELDS),
        "compute_time_by_backend_case": (data_dir / "compute_time_by_backend_case.csv", compute_rows, PLOT_DATA_FIELDS),
        "total_vs_compute_time": (data_dir / "total_vs_compute_time.csv", total_vs_compute_rows, PLOT_DATA_FIELDS),
        "output_error_by_backend": (data_dir / "output_error_by_backend.csv", error_rows, PLOT_DATA_FIELDS),
        "probability_error_by_backend": (data_dir / "probability_error_by_backend.csv", probability_rows, PLOT_DATA_FIELDS),
        "memory_proxy_by_backend": (data_dir / "memory_proxy_by_backend.csv", memory_rows, PLOT_DATA_FIELDS),
        "tn_task_and_intermediate_by_backend": (data_dir / "tn_task_and_intermediate_by_backend.csv", tn_rows, PLOT_DATA_FIELDS),
        "planning_vs_contraction_time": (data_dir / "planning_vs_contraction_time.csv", tn_rows, PLOT_DATA_FIELDS),
        "backend_support_summary": (
            data_dir / "backend_support_summary.csv",
            support_rows,
            [
                "route_id",
                "backend_label",
                "benchmark_role",
                "route_limitation_scope",
                "execution_model",
                "backend_family",
                "record_count",
                "passed_count",
                "failed_count",
                "unavailable_count",
            ],
        ),
        "relative_runtime_vs_quest_anchor": (
            data_dir / "relative_runtime_vs_quest_anchor.csv",
            relative_rows,
            [*PLOT_DATA_FIELDS, "anchor_route_id", "anchor_total_wall_time_s", "relative_runtime"],
        ),
        "relative_runtime_vs_quest_anchor_skipped": (
            data_dir / "relative_runtime_vs_quest_anchor_skipped.csv",
            relative_skipped,
            ["case_id", "route_id", "backend_label", "reason"],
        ),
    }
    for _, (path, table_rows, fields) in table_specs.items():
        _write_csv(path, table_rows, list(fields))
    return {
        key: {
            "path": path,
            "rows": table_rows,
            "fieldnames": list(fields),
            "relative_path": path.relative_to(run_dir).as_posix(),
        }
        for key, (path, table_rows, fields) in table_specs.items()
    }


def _write_plots(run_dir: Path, plot_tables: JsonDict) -> None:
    plots_dir = run_dir / "plots"
    os.environ.setdefault("MPLCONFIGDIR", str(plots_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_json(plots_dir / "plot_manifest.json", {"schema_version": REPORT_RUN_SCHEMA_VERSION, "status": "skipped", "reason": "matplotlib_unavailable", "error": str(exc)})
        return
    plotters = {
        "runtime_by_backend_case.png": ("runtime_by_backend_case", _plot_runtime_by_backend_case),
        "runtime_scaling_by_qubits.png": ("runtime_scaling_by_qubits", _plot_runtime_scaling_by_qubits),
        "output_error_by_backend.png": ("output_error_by_backend", _plot_output_error_by_backend),
        "probability_error_by_backend.png": ("probability_error_by_backend", _plot_probability_error_by_backend),
        "memory_proxy_by_backend.png": ("memory_proxy_by_backend", _plot_memory_proxy_by_backend),
        "tn_task_and_intermediate_by_backend.png": ("tn_task_and_intermediate_by_backend", _plot_tn_task_and_intermediate),
        "planning_vs_contraction_time.png": ("planning_vs_contraction_time", _plot_planning_vs_contraction_time),
        "backend_support_summary.png": ("backend_support_summary", _plot_backend_support_summary),
        "relative_runtime_vs_quest_anchor.png": ("relative_runtime_vs_quest_anchor", _plot_relative_runtime_vs_quest_anchor),
        "compute_time_by_backend_case.png": ("compute_time_by_backend_case", _plot_compute_time_by_backend_case),
        "total_vs_compute_time.png": ("total_vs_compute_time", _plot_total_vs_compute_time),
    }
    entries: list[JsonDict] = []
    for name, (table_key, fn) in plotters.items():
        source = plot_tables[table_key]
        path = plots_dir / name
        reason = fn(plt, path, source["rows"])
        if reason:
            entries.append(
                {
                    "plot": name,
                    "title": PLOT_SPECS[name],
                    "status": "skipped",
                    "reason": reason,
                    "source_csv": source["relative_path"],
                    "source_row_count": len(source["rows"]),
                }
            )
        else:
            entries.append(
                {
                    "plot": name,
                    "title": PLOT_SPECS[name],
                    "status": "generated",
                    "reason": None,
                    "source_csv": source["relative_path"],
                    "source_row_count": len(source["rows"]),
                    "image": _image_metadata(plt, path),
                }
            )
    write_json(
        plots_dir / "plot_manifest.json",
        {
            "schema_version": REPORT_RUN_SCHEMA_VERSION,
            "status": "completed",
            "readability_contract": {
                "min_width_px": 900,
                "min_height_px": 480,
                "min_file_size_bytes": 1000,
                "source_csv_required": True,
                "manual_quality_review_still_required": True,
            },
            "written": [entry["plot"] for entry in entries if entry["status"] == "generated"],
            "skipped": [entry for entry in entries if entry["status"] == "skipped"],
            "plots": entries,
        },
    )


def _plot_runtime_by_backend_case(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    return _grouped_case_backend_bar(
        plt,
        path,
        rows,
        value_field="total_wall_time_s",
        title="Runtime by backend and case",
        ylabel="Wall time (s, log scale)",
        log_y=True,
    )


def _plot_runtime_scaling_by_qubits(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    rows = [row for row in rows if row.get("case_family") and _int_or_none(row.get("n_qubits")) is not None and _positive_float(row.get("total_wall_time_s"))]
    if not rows:
        return "required_data_unavailable"
    families = sorted({str(row["case_family"]) for row in rows})
    if not families:
        return "required_data_unavailable"
    cols = min(2, len(families))
    rows_count = (len(families) + cols - 1) // cols
    fig, axes = plt.subplots(rows_count, cols, figsize=(max(8.0, cols * 5.5), max(4.8, rows_count * 3.6)), squeeze=False, constrained_layout=True)
    for axis in axes.ravel():
        axis.set_visible(False)
    for index, family in enumerate(families):
        axis = axes[index // cols][index % cols]
        axis.set_visible(True)
        family_rows = [row for row in rows if row["case_family"] == family]
        for backend in _backend_order(family_rows):
            backend_rows = sorted((row for row in family_rows if row["backend_label"] == backend), key=lambda item: int(item["n_qubits"]))
            axis.plot(
                [int(row["n_qubits"]) for row in backend_rows],
                [max(float(row["total_wall_time_s"]), 1.0e-12) for row in backend_rows],
                marker="o",
                label=backend,
            )
        axis.set_title(str(family).upper())
        axis.set_xlabel("Qubits")
        axis.set_ylabel("Wall time (s)")
        axis.set_yscale("log")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(fontsize="small")
    _save_figure(fig, path)
    return None


def _plot_output_error_by_backend(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    return _grouped_case_backend_bar(
        plt,
        path,
        rows,
        value_field="max_abs_error",
        title="Output agreement by backend",
        ylabel="Max abs error (log scale, floor 1e-18)",
        log_y=True,
        floor=1.0e-18,
    )


def _plot_probability_error_by_backend(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    return _grouped_case_backend_bar(
        plt,
        path,
        rows,
        value_field="probability_l1_error",
        title="Probability error by backend",
        ylabel="Probability L1 error (log scale, floor 1e-18)",
        log_y=True,
        floor=1.0e-18,
    )


def _plot_memory_proxy_by_backend(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    return _grouped_case_backend_bar(
        plt,
        path,
        rows,
        value_field="memory_proxy_bytes",
        title="Memory proxy by backend",
        ylabel="Bytes (log scale)",
        log_y=True,
    )


def _plot_tn_task_and_intermediate(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    rows = [row for row in rows if row.get("execution_model") == "tensor_network"]
    if not rows:
        return "required_data_unavailable"
    labels = _case_backend_labels(rows)
    fig, axes = plt.subplots(2, 1, figsize=(max(9.0, len(labels) * 0.42), 7.2), constrained_layout=True)
    axes[0].bar(labels, [float(row.get("tn_task_count") or 0.0) for row in rows], color="#2563eb")
    axes[0].set_ylabel("TN tasks")
    axes[0].set_title("TN task count")
    axes[0].tick_params(axis="x", rotation=55, labelsize=8)
    axes[1].bar(labels, [max(float(row.get("tn_max_intermediate_bytes") or 0.0), 1.0) for row in rows], color="#16a34a")
    axes[1].set_ylabel("Max intermediate bytes")
    axes[1].set_title("TN max intermediate size")
    axes[1].set_yscale("log")
    axes[1].tick_params(axis="x", rotation=55, labelsize=8)
    _save_figure(fig, path)
    return None


def _plot_planning_vs_contraction_time(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    rows = [row for row in rows if row.get("execution_model") == "tensor_network"]
    if not rows:
        return "required_data_unavailable"
    labels = _case_backend_labels(rows)
    planning = [float(row.get("planning_time_s") or 0.0) + float(row.get("lowering_time_s") or 0.0) for row in rows]
    kernel = [float(row.get("kernel_time_s") or 0.0) for row in rows]
    fig, axis = plt.subplots(figsize=(max(9.0, len(labels) * 0.42), 5.2), constrained_layout=True)
    axis.bar(labels, planning, label="planning/lowering", color="#7c3aed")
    axis.bar(labels, kernel, bottom=planning, label="contraction", color="#2563eb")
    axis.set_ylabel("Seconds")
    axis.set_title("Planning/lowering time versus contraction time")
    axis.tick_params(axis="x", rotation=55, labelsize=8)
    axis.legend()
    _save_figure(fig, path)
    return None


def _plot_backend_support_summary(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "required_data_unavailable"
    labels = [str(row["backend_label"]) for row in rows]
    passed = [int(row.get("passed_count") or 0) for row in rows]
    failed = [int(row.get("failed_count") or 0) for row in rows]
    fig, axis = plt.subplots(figsize=(max(7.0, len(labels) * 1.3), 4.8), constrained_layout=True)
    axis.bar(labels, passed, label="passed", color="#16a34a")
    axis.bar(labels, failed, bottom=passed, label="failed", color="#dc2626")
    axis.set_ylabel("Records")
    axis.set_title("Backend support / validation summary")
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    _save_figure(fig, path)
    return None


def _plot_relative_runtime_vs_quest_anchor(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    return _grouped_case_backend_bar(
        plt,
        path,
        rows,
        value_field="relative_runtime",
        title="Relative backend timing versus QuEST anchor",
        ylabel="Relative wall time (QuEST = 1.0; not hardware speedup)",
        log_y=False,
    )


def _plot_compute_time_by_backend_case(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    return _grouped_case_backend_bar(
        plt,
        path,
        rows,
        value_field="simulation_compute_time_s",
        title="Compute-focused time by backend and case",
        ylabel="Compute time (s, log scale)",
        log_y=True,
    )


def _plot_total_vs_compute_time(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    rows = [
        row
        for row in rows
        if _positive_float(row.get("total_wall_time_s")) is not None and _positive_float(row.get("simulation_compute_time_s")) is not None
    ]
    if not rows:
        return "required_data_unavailable"
    labels = _case_backend_labels(rows)
    total = [float(row["total_wall_time_s"]) for row in rows]
    compute = [float(row["simulation_compute_time_s"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(9.0, len(labels) * 0.48), 5.6), constrained_layout=True)
    axis.bar([item - width / 2 for item in x], total, width=width, label="total wall", color="#2563eb")
    axis.bar([item + width / 2 for item in x], compute, width=width, label="compute", color="#16a34a")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    axis.set_ylabel("Seconds (log scale)")
    axis.set_title("Total wall time versus compute-focused time")
    axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    _save_figure(fig, path)
    return None


def _grouped_case_backend_bar(
    plt: Any,
    path: Path,
    rows: list[JsonDict],
    *,
    value_field: str,
    title: str,
    ylabel: str,
    log_y: bool,
    floor: float | None = None,
) -> str | None:
    rows = [row for row in rows if _float_or_none(row.get(value_field)) is not None]
    if not rows:
        return "required_data_unavailable"
    cases = sorted({str(row["case_id"]) for row in rows})
    backends = _backend_order(rows)
    if not cases or not backends:
        return "required_data_unavailable"
    x = list(range(len(cases)))
    width = min(0.8 / max(len(backends), 1), 0.24)
    fig, axis = plt.subplots(figsize=(max(9.0, len(cases) * 0.62), 5.6), constrained_layout=True)
    for offset, backend in enumerate(backends):
        values = []
        for case in cases:
            row = next((item for item in rows if item["case_id"] == case and item["backend_label"] == backend), None)
            value = _float_or_none((row or {}).get(value_field))
            if value is None:
                values.append(float("nan"))
            elif floor is not None:
                values.append(max(value, floor))
            else:
                values.append(max(value, 1.0e-18) if log_y else value)
        positions = [item + (offset - (len(backends) - 1) / 2.0) * width for item in x]
        axis.bar(positions, values, width=width, label=backend)
    axis.set_xticks(x)
    axis.set_xticklabels(cases, rotation=35, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if log_y:
        axis.set_yscale("log")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize="small")
    _save_figure(fig, path)
    return None


def _save_figure(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as _plt

        _plt.close(fig)
    except Exception:  # pragma: no cover - defensive fallback for unusual matplotlib states
        fig.clf()


def _image_metadata(plt: Any, path: Path) -> JsonDict:
    payload: JsonDict = {"size_bytes": path.stat().st_size if path.exists() else 0}
    try:
        image = plt.imread(path)
        height, width = image.shape[:2]
        payload.update(
            {
                "width_px": int(width),
                "height_px": int(height),
                "reasonable_dimensions": int(width) >= 900 and int(height) >= 480,
                "non_empty": payload["size_bytes"] > 1000,
            }
        )
    except Exception as exc:
        payload.update({"read_error": str(exc), "reasonable_dimensions": False, "non_empty": False})
    return payload


def _plot_data_row(record: JsonDict) -> JsonDict:
    notes = _json_value(record.get("notes"))
    metrics = _json_value(record.get("validation_error_metrics"))
    route_id = str(record.get("route_id") or record.get("backend_id") or "unknown")
    case_id = str(record.get("case_id") or "unknown")
    return {
        "case_id": case_id,
        "case_family": _case_family(case_id, record, notes),
        "n_qubits": _int_or_none(record.get("n_qubits") or notes.get("n_qubits")),
        "gate_count": _int_or_none(record.get("gate_count") or notes.get("gate_count")),
        "route_id": route_id,
        "backend_label": _backend_label(route_id),
        "backend_family": record.get("backend_family"),
        "benchmark_role": record.get("benchmark_role"),
        "route_role_description": record.get("route_role_description"),
        "route_limitation_scope": record.get("route_limitation_scope"),
        "execution_model": record.get("execution_model"),
        "contraction_execution_target": record.get("contraction_execution_target") or record.get("execution_target"),
        "accelerator_kind": record.get("accelerator_kind") or "none",
        "gpu_backend_verified": bool(record.get("gpu_backend_verified", False)),
        "gpu_program_executed": bool(record.get("gpu_program_executed", False)),
        "gpu_device_name": record.get("gpu_device_name"),
        "gpu_runtime_stack": record.get("gpu_runtime_stack"),
        "status": record.get("status"),
        "validation_status": record.get("validation_status"),
        "total_wall_time_s": _float_or_none(record.get("total_wall_time_s_median") if record.get("total_wall_time_s_median") is not None else record.get("total_wall_time_s")),
        "kernel_time_s": _float_or_none(record.get("kernel_time_s")),
        "planning_time_s": _float_or_none(record.get("planning_time_s")),
        "lowering_time_s": _float_or_none(record.get("lowering_time_s")),
        "setup_time_s": _float_or_none(record.get("setup_time_s")),
        "circuit_lowering_time_s": _float_or_none(record.get("circuit_lowering_time_s")),
        "data_transfer_time_s": _float_or_none(record.get("data_transfer_time_s")),
        "simulation_compute_time_s": _float_or_none(record.get("simulation_compute_time_s_median") if record.get("simulation_compute_time_s_median") is not None else (record.get("simulation_compute_time_s") if record.get("simulation_compute_time_s") is not None else record.get("kernel_time_s"))),
        "validation_time_s": _float_or_none(record.get("validation_time_s")),
        "output_materialization_time_s": _float_or_none(record.get("output_materialization_time_s")),
        "timing_scope": record.get("timing_scope"),
        "gpu_synchronized": bool(record.get("gpu_synchronized", False)),
        "validation_method": record.get("validation_method"),
        "expected_runtime_class": record.get("expected_runtime_class"),
        "expected_memory_class": record.get("expected_memory_class"),
        "intended_use": record.get("intended_use"),
        "max_qubits": _int_or_none(record.get("max_qubits")),
        "manual_invocation_required": bool(record.get("manual_invocation_required", False)),
        "expected_risk": record.get("expected_risk"),
        "known_heavy_backends": record.get("known_heavy_backends"),
        "resource_guard_status": record.get("resource_guard_status"),
        "resource_skip_reason": record.get("resource_skip_reason"),
        "repeat_id": _int_or_none(record.get("repeat_id")),
        "measured_repeat_count": _int_or_none(record.get("measured_repeat_count")),
        "total_wall_time_s_median": _float_or_none(record.get("total_wall_time_s_median")),
        "simulation_compute_time_s_median": _float_or_none(record.get("simulation_compute_time_s_median")),
        "max_abs_error": _float_or_none(record.get("max_abs_error") if record.get("max_abs_error") is not None else metrics.get("max_abs_error")),
        "l2_error": _float_or_none(record.get("l2_error") if record.get("l2_error") is not None else metrics.get("l2_error")),
        "probability_l1_error": _float_or_none(metrics.get("probability_l1_error") if metrics.get("probability_l1_error") is not None else record.get("probability_l1_error")),
        "probability_max_abs_error": _float_or_none(metrics.get("probability_max_abs_error") if metrics.get("probability_max_abs_error") is not None else record.get("probability_max_abs_error")),
        "memory_proxy_bytes": _memory_proxy_bytes(record),
        "statevector_bytes": _int_or_none(record.get("statevector_bytes")),
        "tn_task_count": _int_or_none(record.get("tn_task_count") if record.get("tn_task_count") is not None else record.get("task_count")),
        "tn_max_intermediate_bytes": _int_or_none(record.get("tn_max_intermediate_bytes")),
        "tn_estimated_flops": _int_or_none(record.get("tn_estimated_flops")),
        "tn_estimated_bytes": _int_or_none(record.get("tn_estimated_bytes")),
    }


def _case_family(case_id: str, record: JsonDict, notes: JsonDict) -> str:
    explicit = record.get("circuit_family") or notes.get("circuit_family")
    if explicit:
        return str(explicit)
    name = case_id
    if name.startswith("quest_"):
        name = name[len("quest_") :]
    name = re.sub(r"_[0-9]+q$", "", name)
    parts = name.split("_")
    return parts[0] if parts else name


def _backend_label(route_id: str) -> str:
    if route_id in BACKEND_LABELS:
        return BACKEND_LABELS[route_id]
    return route_id.replace("_", " ")


def _memory_proxy_bytes(record: JsonDict) -> int | None:
    if record.get("execution_model") == "full_state":
        return _int_or_none(record.get("statevector_bytes"))
    for key in ("tn_max_intermediate_bytes", "estimated_transfer_bytes", "statevector_bytes"):
        value = _int_or_none(record.get(key))
        if value is not None:
            return value
    return None


def _backend_order(rows: list[JsonDict]) -> list[str]:
    labels = sorted({str(row.get("backend_label") or _backend_label(str(row.get("route_id") or "unknown"))) for row in rows})
    preferred = list(BACKEND_LABELS.values())
    return [label for label in preferred if label in labels] + [label for label in labels if label not in preferred]


def _case_backend_labels(rows: list[JsonDict]) -> list[str]:
    return [f"{row.get('case_id')}\n{row.get('backend_label')}" for row in rows]


def _backend_support_rows(rows: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("route_id") or "unknown")].append(row)
    out: list[JsonDict] = []
    for route_id, group in sorted(grouped.items()):
        validation_statuses = {str(row.get("validation_status") or "") for row in group}
        statuses = {str(row.get("status") or "") for row in group}
        out.append(
            {
                "route_id": route_id,
                "backend_label": _backend_label(route_id),
                "benchmark_role": next((row.get("benchmark_role") for row in group if row.get("benchmark_role")), None),
                "route_limitation_scope": next((row.get("route_limitation_scope") for row in group if row.get("route_limitation_scope")), None),
                "execution_model": next((row.get("execution_model") for row in group if row.get("execution_model")), None),
                "backend_family": next((row.get("backend_family") for row in group if row.get("backend_family")), None),
                "record_count": len(group),
                "passed_count": sum(1 for row in group if row.get("validation_status") in {"passed", "reference"}),
                "failed_count": sum(1 for row in group if row.get("validation_status") == "failed" or row.get("status") in {"failed", "validation_failed"}),
                "unavailable_count": sum(1 for row in group if row.get("status") in {"unavailable", "skipped", "not_executed"}),
                "validation_statuses": sorted(validation_statuses),
                "statuses": sorted(statuses),
            }
        )
    return out


def _relative_runtime_rows(rows: list[JsonDict]) -> tuple[list[JsonDict], list[JsonDict]]:
    anchors: dict[str, JsonDict] = {}
    skipped: list[JsonDict] = []
    for row in rows:
        if row.get("route_id") == "quest_cpu_full_state_exact":
            runtime = _positive_float(row.get("total_wall_time_s"))
            if runtime is not None:
                anchors[str(row.get("case_id"))] = row
    relative: list[JsonDict] = []
    for row in rows:
        case_id = str(row.get("case_id") or "unknown")
        anchor = anchors.get(case_id)
        if anchor is None:
            skipped.append({"case_id": case_id, "route_id": row.get("route_id"), "backend_label": row.get("backend_label"), "reason": "missing_valid_quest_anchor"})
            continue
        runtime = _positive_float(row.get("total_wall_time_s"))
        if runtime is None:
            skipped.append({"case_id": case_id, "route_id": row.get("route_id"), "backend_label": row.get("backend_label"), "reason": "missing_valid_backend_runtime"})
            continue
        anchor_runtime = float(anchor["total_wall_time_s"])
        payload = dict(row)
        payload["anchor_route_id"] = "quest_cpu_full_state_exact"
        payload["anchor_total_wall_time_s"] = anchor_runtime
        payload["relative_runtime"] = runtime / anchor_runtime
        relative.append(payload)
    return relative, skipped


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _positive_float(value: Any) -> float | None:
    out = _float_or_none(value)
    if out is None or out <= 0.0:
        return None
    return out


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _retention_manifest(run_dir: Path, *, mode: str, pruned: list[JsonDict], retained: list[Path]) -> JsonDict:
    return to_jsonable(
        {
            "schema_version": ARTIFACT_RETENTION_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "artifact_retention": mode,
            "status": "completed",
            "idempotent": True,
            "pruned_file_count": len(pruned),
            "pruned_byte_count": sum(int((ref.get("metadata") or {}).get("size_bytes", 0) or 0) for ref in pruned),
            "retained_file_count": len(retained),
            "pruned_artifacts": pruned,
            "compare_results_supported": (run_dir / "normalized_records.jsonl").exists(),
            "report_run_supported": (run_dir / "normalized_records.jsonl").exists(),
        }
    )


def _require_new_run_layout(run_dir: Path) -> None:
    if not (run_dir / "run_manifest.json").exists() or not (run_dir / "normalized_records.jsonl").exists():
        raise ValueError("unsupported_legacy_run_layout")


def _compact_prune_candidates(run_dir: Path) -> list[Path]:
    candidates: set[Path] = set()
    for path in run_dir.rglob("*"):
        rel = path.relative_to(run_dir).as_posix()
        if "runner_work/" in rel or rel.endswith("/runner_work") or path.name == "runner_work":
            candidates.add(path)
            continue
        if path.is_file() and path.suffix == ".bin":
            candidates.add(path)
            continue
        if any(part in {"operands", "references", "outputs"} for part in path.relative_to(run_dir).parts):
            candidates.add(path)
    return sorted(candidates, key=lambda item: (len(item.parts), item.as_posix()), reverse=True)


def _pruned_reference(run_dir: Path, rel_path: Path, reason: str) -> JsonDict:
    path = run_dir / rel_path
    return to_jsonable(
        {
            "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
            "role": "pruned_artifact",
            "relative_path": rel_path.as_posix(),
            "retained": False,
            "status": "intentionally_pruned",
            "prune_reason": reason,
            "metadata": _file_metadata(path),
        }
    )


def _mark_pruned_references(run_dir: Path, refs: list[JsonDict]) -> None:
    if not refs:
        return
    by_path = {str(ref["relative_path"]): ref for ref in refs}
    for json_path in list(run_dir.rglob("*.json")):
        if any(part in {"runner_work", "operands", "references", "outputs"} for part in json_path.relative_to(run_dir).parts):
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        updated = _replace_pruned_refs(payload, by_path, run_dir=run_dir, base_dir=json_path.parent)
        if updated != payload:
            write_json(json_path, updated)


def _replace_pruned_refs(value: Any, refs: dict[str, JsonDict], *, run_dir: Path, base_dir: Path) -> Any:
    if isinstance(value, dict):
        current = dict(value)
        rel = current.get("relative_path")
        resolved = _matching_reference_key(rel, refs, run_dir=run_dir, base_dir=base_dir) if isinstance(rel, str) else None
        if resolved in refs:
            return refs[resolved]
        for key, child in list(current.items()):
            resolved_child = _matching_reference_key(child, refs, run_dir=run_dir, base_dir=base_dir) if isinstance(child, str) else None
            if resolved_child in refs and (key.endswith("artifact") or key.endswith("path") or key == "relative_path"):
                current[key] = refs[resolved_child]
            else:
                current[key] = _replace_pruned_refs(child, refs, run_dir=run_dir, base_dir=base_dir)
        return current
    if isinstance(value, list):
        return [_replace_pruned_refs(item, refs, run_dir=run_dir, base_dir=base_dir) for item in value]
    return value


def _matching_reference_key(value: str | None, refs: dict[str, JsonDict], *, run_dir: Path, base_dir: Path) -> str | None:
    for key in _reference_keys(value, run_dir=run_dir, base_dir=base_dir):
        if key in refs:
            return key
    return None


def _reference_keys(value: str | None, *, run_dir: Path, base_dir: Path) -> list[str]:
    if not value:
        return []
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(run_dir / raw)
        candidates.append(base_dir / raw)
    keys: list[str] = []
    for candidate in candidates:
        try:
            keys.append(candidate.resolve().relative_to(run_dir.resolve()).as_posix())
        except ValueError:
            try:
                keys.append(candidate.relative_to(run_dir).as_posix())
            except ValueError:
                continue
    keys.append(value)
    return keys


def _file_metadata(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    if path.is_dir():
        files = [item for item in path.rglob("*") if item.is_file()]
        return {"kind": "directory", "file_count": len(files), "size_bytes": sum(item.stat().st_size for item in files)}
    payload: JsonDict = {"kind": "file", "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    if path.suffix == ".npy":
        try:
            import numpy as np

            array = np.load(path, allow_pickle=False, mmap_mode="r")
            payload["dtype"] = str(array.dtype)
            payload["shape"] = tuple(int(dim) for dim in array.shape)
        except Exception:
            pass
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_files(run_dir: Path) -> list[Path]:
    return sorted(path.relative_to(run_dir) for path in run_dir.rglob("*") if path.is_file())


def _cleanup_empty_report_dirs(run_dir: Path) -> None:
    for directory in sorted((p for p in run_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if directory == run_dir:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass


def _write_csv(path: Path, rows: list[JsonDict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})
    tmp.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return value


def _kernel_family_summary(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("kernel_family") or "unsupported")].append(record)
    rows = []
    for family, group in sorted(grouped.items()):
        rows.append(
            {
                "kernel_family": family,
                "record_count": len(group),
                "task_count": sum(int(row.get("task_count", 0) or 0) for row in group),
                "validated_task_count": sum(int(row.get("validated_task_count", 0) or 0) for row in group),
                "unsupported_task_count": sum(int(row.get("unsupported_task_count", 0) or 0) for row in group),
            }
        )
    return rows


def _resource_profile_counts(records: list[JsonDict]) -> Counter[tuple[str, str, str, str, str, str]]:
    counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    for record in records:
        key = (
            str(record.get("expected_runtime_class") or "unspecified"),
            str(record.get("expected_memory_class") or "unspecified"),
            str(record.get("intended_use") or "unspecified"),
            str(record.get("max_qubits") or ""),
            str(bool(record.get("manual_invocation_required", False))),
            _csv_value(record.get("expected_risk") or ""),
        )
        counts[key] += 1
    return counts


def _quantization_rows(records: list[JsonDict]) -> list[JsonDict]:
    rows = []
    for record in records:
        if record.get("execution_target") != "upmem":
            continue
        metrics = _json_value(record.get("validation_error_metrics"))
        notes = _json_value(record.get("notes"))
        rows.append(
            {
                "case_id": record.get("case_id"),
                "policy": notes.get("policy"),
                "quantization_mode": notes.get("quantization_mode"),
                "validation_status": record.get("validation_status"),
                "max_abs_error": metrics.get("max_abs_error"),
                "l2_error": metrics.get("l2_error"),
            }
        )
    return rows


def _unsupported_rows(records: list[JsonDict]) -> list[JsonDict]:
    grouped: Counter[tuple[str, str, str, str]] = Counter()
    for record in records:
        count = int(record.get("unsupported_task_count", 0) or 0)
        if count <= 0:
            continue
        notes = _json_value(record.get("notes"))
        key = (
            str(record.get("case_id")),
            str(notes.get("policy")),
            str(notes.get("quantization_mode")),
            str(notes.get("reason") or record.get("resource_skip_reason") or record.get("warnings") or record.get("status") or "unsupported"),
        )
        grouped[key] += count
    return [{"case_id": k[0], "policy": k[1], "quantization_mode": k[2], "reason": k[3], "count": v} for k, v in sorted(grouped.items())]


def _per_task_rows(run_dir: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for path in sorted(run_dir.rglob("upmem_taskgraph_task_metrics.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def _per_task_fieldnames(run_dir: Path) -> set[str]:
    fields: set[str] = set()
    for row in _per_task_rows(run_dir):
        fields.update(row)
    return fields or {"status"}


def _per_case_rows(records: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "case_id": record.get("case_id"),
            "policy": record.get("policy") or _json_value(record.get("notes")).get("policy"),
            "quantization_mode": record.get("quantization_mode") or _json_value(record.get("notes")).get("quantization_mode"),
            "task_count": record.get("task_count"),
            "validated_task_count": record.get("validated_task_count"),
            "unsupported_task_count": record.get("unsupported_task_count"),
            "status": record.get("status"),
        }
        for record in records
    ]


def _timing_rows(records: list[JsonDict]) -> list[JsonDict]:
    rows = []
    for record in records:
        notes = _json_value(record.get("notes"))
        rows.append(
            {
                "case_id": record.get("case_id"),
                "policy": notes.get("policy"),
                "quantization_mode": notes.get("quantization_mode"),
                "cpu_reference_time_s": record.get("total_wall_time_s") if record.get("execution_target") == "cpu" else None,
                "upmem_runtime_wall_time_s": record.get("total_wall_time_s") if record.get("execution_target") == "upmem" else None,
                "host_orchestration_time_s": None,
                "quantization_time_s": None,
                "bridge_prepare_time_s": None,
                "native_build_time_s": record.get("build_time_s") if record.get("execution_target") == "upmem" else None,
                "dpu_program_wall_time_s": record.get("kernel_time_s") if record.get("execution_target") == "upmem" else None,
                "simulation_compute_time_s": record.get("simulation_compute_time_s") if record.get("simulation_compute_time_s") is not None else record.get("kernel_time_s"),
                "setup_time_s": record.get("setup_time_s"),
                "data_transfer_time_s": record.get("data_transfer_time_s"),
                "dequantization_time_s": None,
                "validation_time_s": record.get("validation_time_s"),
                "output_materialization_time_s": record.get("output_materialization_time_s"),
                "timing_status": _timing_status(record),
            }
        )
    return rows


def _timing_status(record: JsonDict) -> str:
    if record.get("simulator_or_hardware") == "simulator":
        return "measured_sdk_simulator_wall_clock_not_hardware"
    if record.get("execution_target") == "cpu":
        return "measured_cpu_reference"
    return "not_measured"


def _validation_summary(records: list[JsonDict]) -> JsonDict:
    return {
        "schema_version": REPORT_RUN_SCHEMA_VERSION,
        "record_count": len(records),
        "passed_count": sum(1 for row in records if row.get("validation_status") in {"passed", "reference"}),
        "failed_count": sum(1 for row in records if row.get("validation_status") == "failed"),
        "hardware_speedup_applicable": False,
    }


def _validation_failures(records: list[JsonDict]) -> list[JsonDict]:
    return [row for row in records if row.get("validation_status") == "failed"]


def _report_markdown(records: list[JsonDict], run_dir: Path | None = None) -> str:
    lines = [
        "# Benchmark Run Report",
        "",
        f"Records: {len(records)}",
        "",
        "SDK simulator timings, when present, are wall-clock development measurements, not hardware speedups.",
        "CPU full-state, CPU tensor-network, and future UPMEM tensor-network records are separated by execution model and target.",
        "",
        "## Execution Models",
        "",
        "| Execution model | Records |",
        "| --- | ---: |",
    ]
    for model, count in sorted(Counter(str(record.get("execution_model") or "unspecified") for record in records).items()):
        lines.append(f"| {model} | {count} |")
    lines.extend(
        [
            "",
            "## Resource Profiles",
            "",
            "| Runtime class | Memory class | Intended use | Max qubits | Manual | Risks | Records |",
            "| --- | --- | --- | ---: | --- | --- | ---: |",
        ]
    )
    for profile, count in sorted(_resource_profile_counts(records).items(), key=lambda item: tuple(str(part) for part in item[0])):
        runtime_class, memory_class, intended_use, max_qubits, manual, risks = profile
        lines.append(f"| {runtime_class} | {memory_class} | {intended_use} | {max_qubits} | {manual} | {risks} | {count} |")
    lines.extend(
        [
            "",
            "## Kernel Families",
            "",
            "| Kernel family | Records | Tasks | Validated tasks | Unsupported tasks |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _kernel_family_summary(records):
        lines.append(f"| {row['kernel_family']} | {row['record_count']} | {row['task_count']} | {row['validated_task_count']} | {row['unsupported_task_count']} |")
    lines.extend(
        [
            "",
            "## Backend Metadata",
            "",
            "| Route | Benchmark role | Backend | Model | Target | Accelerator | Output | Limitation scope |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    seen_routes: set[str] = set()
    for record in records:
        route_id = str(record.get("route_id") or "unknown")
        if route_id in seen_routes:
            continue
        seen_routes.add(route_id)
        lines.append(
            f"| {route_id} | {record.get('benchmark_role')} | {record.get('backend_family')} | {record.get('execution_model')} | "
            f"{record.get('contraction_execution_target')} | {record.get('accelerator_kind') or 'none'} | {record.get('output_kind')} | "
            f"{record.get('route_limitation_scope')} |"
        )
    lines.extend(
        [
            "",
            "## Timing Breakdown",
            "",
            "| Case | Route | Total wall time s | Compute time s | Setup s | Transfer s | Validation s | Output write s |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        lines.append(
            f"| {record.get('case_id')} | {record.get('route_id')} | {record.get('total_wall_time_s')} | "
            f"{record.get('simulation_compute_time_s') if record.get('simulation_compute_time_s') is not None else record.get('kernel_time_s')} | "
            f"{record.get('setup_time_s')} | {record.get('data_transfer_time_s')} | {record.get('validation_time_s')} | "
            f"{record.get('output_materialization_time_s')} |"
        )
    lines.extend(
        [
            "",
            "## Validation Methods",
            "",
            "| Validation method | Records |",
            "| --- | ---: |",
        ]
    )
    for method, count in sorted(Counter(str(record.get("validation_method") or "unspecified") for record in records).items()):
        lines.append(f"| {method} | {count} |")
    lines.extend(
        [
            "",
            "## Tensor Network Metrics",
            "",
            "| Case | Route | TN tasks | Max intermediate bytes | Estimated FLOPs | Estimated bytes |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in records:
        if record.get("execution_model") == "tensor_network":
            lines.append(
                f"| {record.get('case_id')} | {record.get('route_id')} | {record.get('tn_task_count')} | "
                f"{record.get('tn_max_intermediate_bytes')} | {record.get('tn_estimated_flops')} | {record.get('tn_estimated_bytes')} |"
            )
    lines.extend(
        [
            "",
            "## Output Agreement",
            "",
            "| Case | Route | Validation | Max abs error | L2 error |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for record in records:
        metrics = _json_value(record.get("validation_error_metrics"))
        lines.append(
            f"| {record.get('case_id')} | {record.get('route_id')} | {record.get('validation_status')} | "
            f"{metrics.get('max_abs_error')} | {metrics.get('l2_error')} |"
        )
    lines.extend(
        [
            "",
            "## Probability Agreement",
            "",
            "| Case | Route | Probability L1 error | Probability max abs error |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for record in records:
        metrics = _json_value(record.get("validation_error_metrics"))
        lines.append(
            f"| {record.get('case_id')} | {record.get('route_id')} | "
            f"{metrics.get('probability_l1_error')} | {metrics.get('probability_max_abs_error')} |"
        )
    lines.extend(
        [
            "",
            "## Memory / Proxy Metrics",
            "",
            "| Case | Route | Statevector bytes | TN max intermediate bytes | Memory proxy bytes |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in (_plot_data_row(record) for record in records):
        lines.append(
            f"| {row.get('case_id')} | {row.get('route_id')} | {row.get('statevector_bytes')} | "
            f"{row.get('tn_max_intermediate_bytes')} | {row.get('memory_proxy_bytes')} |"
        )
    if run_dir is not None:
        lines.extend(_plot_inventory_markdown(run_dir))
    lines.append("")
    return "\n".join(lines)


def _plot_inventory_markdown(run_dir: Path) -> list[str]:
    manifest_path = run_dir / "plots" / "plot_manifest.json"
    lines = [
        "",
        "## Plot Inventory",
        "",
        "| Plot | Status | Source rows | Reason | Source CSV |",
        "| --- | --- | ---: | --- | --- |",
    ]
    if not manifest_path.exists():
        lines.append("| plot_manifest.json | missing | 0 | plot manifest not written |  |")
        return lines
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        lines.append(f"| plot_manifest.json | invalid | 0 | {exc} |  |")
        return lines
    if manifest.get("status") == "skipped":
        lines.append(f"| all plots | skipped | 0 | {manifest.get('reason')} |  |")
        return lines
    for item in manifest.get("plots", []):
        lines.append(
            f"| {item.get('plot')} | {item.get('status')} | {item.get('source_row_count')} | "
            f"{item.get('reason') or ''} | {item.get('source_csv') or ''} |"
        )
    return lines


def _compare_grouped(
    baseline: list[JsonDict],
    candidate: list[JsonDict],
    *,
    key_fields: tuple[str, ...],
    predicate: Any | None = None,
) -> JsonDict:
    def selected(records: list[JsonDict]) -> dict[tuple[Any, ...], JsonDict]:
        out = {}
        for record in records:
            if predicate is not None and not predicate(record):
                continue
            out[tuple(record.get(field) for field in key_fields)] = record
        return out

    old = selected(baseline)
    new = selected(candidate)
    keys = sorted(set(old) | set(new), key=lambda item: tuple(str(part) for part in item))
    rows = []
    for key in keys:
        before = old.get(key)
        after = new.get(key)
        rows.append(
            {
                "key": dict(zip(key_fields, key)),
                "baseline_present": before is not None,
                "candidate_present": after is not None,
                "validation_status_changed": (before or {}).get("validation_status") != (after or {}).get("validation_status"),
                "status_changed": (before or {}).get("status") != (after or {}).get("status"),
                "task_count_delta": int((after or {}).get("task_count", 0) or 0) - int((before or {}).get("task_count", 0) or 0),
                "unsupported_task_count_delta": int((after or {}).get("unsupported_task_count", 0) or 0) - int((before or {}).get("unsupported_task_count", 0) or 0),
                "total_wall_time_delta_s": float((after or {}).get("total_wall_time_s", 0.0) or 0.0) - float((before or {}).get("total_wall_time_s", 0.0) or 0.0),
                "max_abs_error_delta": _metric_delta(before, after, "max_abs_error"),
            }
        )
    return {
        "key_fields": key_fields,
        "row_count": len(rows),
        "newly_supported_count": sum(1 for row in rows if not row["baseline_present"] and row["candidate_present"]),
        "newly_unsupported_count": sum(1 for row in rows if row["baseline_present"] and not row["candidate_present"]),
        "validation_regression_count": sum(1 for row in rows if row["validation_status_changed"]),
        "rows": rows,
    }


def _metric_delta(before: JsonDict | None, after: JsonDict | None, name: str) -> float | None:
    old = _json_value((before or {}).get("validation_error_metrics")).get(name)
    new = _json_value((after or {}).get("validation_error_metrics")).get(name)
    if old is None or new is None:
        return None
    return float(new) - float(old)


def _compare_runs_markdown(payload: JsonDict) -> str:
    final = payload["final_validation_accuracy_timing"]
    kernel = payload["kernel_family_mix"]
    lines = [
        "# Benchmark Run Comparison",
        "",
        f"Baseline: `{payload['baseline_run']}`",
        f"Candidate: `{payload['candidate_run']}`",
        "",
        f"Final comparison rows: {final['row_count']}",
        f"Validation changed rows: {final['validation_regression_count']}",
        f"Kernel-family comparison rows: {kernel['row_count']}",
        "",
        "SDK simulator timing deltas are not hardware speedups.",
        "",
    ]
    return "\n".join(lines)


def _json_value(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _git_commit(root_dir: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root_dir, check=False, text=True, capture_output=True, timeout=5)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty(root_dir: Path) -> bool | None:
    try:
        result = subprocess.run(["git", "status", "--short"], cwd=root_dir, check=False, text=True, capture_output=True, timeout=5)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _environment_hash() -> str:
    keys = {key: value for key, value in os.environ.items() if key.startswith(("UPMEM", "SIMPLEPIM", "PID_COMM", "PYTHON"))}
    payload = json.dumps(keys, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
