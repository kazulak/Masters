from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_bench.bench.result_artifacts import load_result_records  # noqa: E402


SCHEMA_VERSION = "research_benchmark_pack_v1"
RESEARCH_PACK_KIND = "research_benchmark_pack"
DEFAULT_COMPARISON_ROOT = ROOT / "runs" / "comparisons" / "research_pack"

RESEARCH_SUITES = {
    "cpu_gpu": ROOT / "configs" / "suites" / "manual" / "thesis_full_state_cpu_gpu.yml",
    "cpu_gpu_correctness": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_full_state_correctness.yml",
    "cpu_tn": ROOT / "configs" / "suites" / "manual" / "thesis_cpu_tn_quimb.yml",
    "tn_path_quantization": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_tn_paths_quantization.yml",
    # V2 is the active, projected-prefix planner evidence surface.  V1 stays
    # addressable for historical comparison but is never mixed into a default
    # research report because the objective semantics differ.
    "planner_paths": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_planner_semantic_v2.yml",
    "planner_sensitivity": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_planner_sensitivity_v2.yml",
    "planner_paths_v1": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_planner_compare.yml",
    "planner_sensitivity_v1": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_planner_sensitivity.yml",
    # This group intentionally uses the strict generic-only MVP command rather
    # than the route-comparison suite.  The latter permits dense bridge tasks,
    # which is useful for route coverage but is not generic-TN boundary evidence.
    "upmem_boundary": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_upmem_quantization_boundary.yml",
    "upmem_quantization_stress": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "thesis_upmem_quantization_stress.yml",
    "internal_parallelism": ROOT
    / "configs"
    / "suites"
    / "manual"
    / "research_internal_parallelism.yml",
}

SUITE_COMMAND_ORDER = (
    "cpu_gpu_correctness",
    "cpu_gpu",
    "cpu_tn",
    "tn_path_quantization",
    "planner_paths",
    "planner_sensitivity",
    "upmem_boundary",
    "upmem_quantization_stress",
    "internal_parallelism",
)

RELEVANT_ENV_VARS = (
    "BENCH_CPU_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quantum_bench_mplconfig")

FORBIDDEN_EVIDENCE_DERIVED_NAMES = {
    "comparison_summary.md",
    "simulation_backend_compare_results.csv",
    "simulation_backend_compare_pairs.csv",
    "upmem_mvp_benchmark_results.csv",
    "kernel_family_summary.csv",
    "quantization_accuracy_summary.csv",
    "unsupported_reasons.csv",
    "quantization_comparison.csv",
    "upmem_quantization_attribution.csv",
    "upmem_physical_quantization_attribution.csv",
    "upmem_physical_taskgraph_breakdown.csv",
    "upmem_hardware_mvp_summary.csv",
    "upmem_hardware_generic_mvp_summary.csv",
    "cpu_gpu_performance_summary.csv",
    "full_state_tn_comparison.csv",
    "per_case_route_stats.csv",
    "paired_speedups.csv",
    "unsupported_cases.csv",
    "validation_summary.csv",
    "route_capability_matrix.csv",
    "planner_component_diagnostics.csv",
    "benchmark_summary.md",
    "plot_manifest.json",
}


JsonDict = dict[str, Any]


PLOT_STATUSES = (
    "generated_valid",
    "generated_todo_missing_data",
    "generated_todo_no_variance",
    "generated_todo_not_implemented",
    "failed",
)


@dataclass(frozen=True)
class PlotSpec:
    filename: str
    title: str
    source_csv: str
    source_fields: tuple[str, ...]
    claim_basis: str
    caption: str
    x_label: str
    y_label: str
    renderer: Callable[[Any, Path], str | None] | None = None
    not_implemented_reason: str | None = None
    variance_fields: tuple[str, ...] = ()
    data_rows: list[JsonDict] | None = None
    allow_zero_variance: bool = False


@dataclass(frozen=True)
class PlotOutcome:
    status: str
    reason: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.status not in PLOT_STATUSES:
            raise ValueError(f"unknown plot status: {self.status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a thesis research benchmark pack from normalized records."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="Print exact research benchmark commands and preflight context."
    )
    _add_common_args(plan)

    run = subparsers.add_parser(
        "run",
        help="Create a research pack; long benchmark execution requires --full or RUN_RESEARCH=1.",
    )
    _add_common_args(run)
    run.add_argument(
        "--full", action="store_true", help="Run long manual research suites."
    )
    run.add_argument(
        "--suite",
        action="append",
        choices=sorted(RESEARCH_SUITES),
        help="Limit full execution/report discovery to a suite group.",
    )

    report = subparsers.add_parser(
        "report", help="Generate a research pack from existing evidence runs."
    )
    _add_common_args(report)
    report.add_argument(
        "--input",
        action="append",
        dest="inputs",
        help="Evidence run directory or artifact path. May be repeated.",
    )
    report.add_argument(
        "--suite",
        action="append",
        choices=sorted(RESEARCH_SUITES),
        help="Auto-discover latest evidence for a suite group.",
    )

    args = parser.parse_args(argv)
    if args.command == "plan":
        print_plan(args.root)
        return 0
    if args.command == "run":
        return run_pack(
            args.root,
            args.out,
            label=args.label,
            suite_filter=args.suite,
            full=bool(args.full or os.environ.get("RUN_RESEARCH") == "1"),
        )
    if args.command == "report":
        return report_pack(
            args.root,
            args.out,
            label=args.label,
            inputs=[Path(item) for item in args.inputs or ()],
            suite_filter=args.suite,
        )
    raise AssertionError(f"unknown command: {args.command}")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Implementation root directory."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory for generated research pack.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Named comparison output namespace (for example: planner_v2).",
    )


def print_plan(root: Path = ROOT) -> None:
    print("Research benchmark pack plan")
    print("============================")
    print("")
    print("Preflight:")
    print(_command(["scripts/doctor.py"], python_script=True))
    print("")
    print("GPU verification before GPU research rows:")
    print(_bench_command(["simulation-backend-probe", "--verify-gpu", "quest-hip"]))
    print("")
    print("Manual research suites:")
    for key in SUITE_COMMAND_ORDER:
        print(f"- {key}:")
        print(f"  {_bench_command(_research_suite_argv(key, root))}")
    print("")
    print("Pack commands:")
    print("  BENCH_CPU_THREADS=<physical-core-count> make thesis-run")
    print("  make thesis-promote")
    print("  make thesis-report")
    print("")
    print(
        "Derived artifacts are written under runs/comparisons/research_pack/<timestamp>/; evidence runs remain read-only."
    )


def run_pack(
    root: Path,
    out: Path | None,
    *,
    label: str | None = None,
    suite_filter: list[str] | None,
    full: bool,
) -> int:
    out_dir = _pack_dir(root, out, label=label)
    selected = _selected_suites(suite_filter)
    command_results: list[JsonDict] = []
    evidence_inputs: list[Path] = []
    if full:
        command_results.append(_run_capture(root, ["make", "doctor"]))
        gpu_verified = True
        if "cpu_gpu" in selected or "cpu_gpu_correctness" in selected:
            command_results.append(
                _run_capture(
                    root,
                    _bench_argv(
                        ["simulation-backend-probe", "--verify-gpu", "quest-hip"]
                    ),
                )
            )
            gpu_verified = _gpu_verification_passed(root)
        for key in SUITE_COMMAND_ORDER:
            if key not in selected:
                continue
            if key in {"cpu_gpu", "cpu_gpu_correctness"} and not gpu_verified:
                command_results.append(
                    _skipped_group_result(key, _gpu_blocker_reason(root))
                )
                continue
            result = _run_capture(root, _bench_argv(_research_suite_argv(key, root)))
            command_results.append(result)
            latest = root / "runs" / "latest"
            if result["returncode"] == 0 and latest.exists():
                evidence_inputs.append(latest.resolve())
    else:
        command_results.append(
            {
                "command": "lightweight preflight only",
                "returncode": 0,
                "stdout": "Set RUN_RESEARCH=1 or pass --full to run long manual suites.",
                "stderr": "",
            }
        )
    return _write_pack(
        root,
        out_dir,
        evidence_inputs,
        command_results=command_results,
        selected_suite_keys=selected,
        generation_mode="run",
    )


def report_pack(
    root: Path,
    out: Path | None,
    *,
    label: str | None = None,
    inputs: list[Path],
    suite_filter: list[str] | None,
) -> int:
    out_dir = _pack_dir(root, out, label=label)
    selected = _selected_suites(suite_filter)
    evidence_inputs = [path.resolve() for path in inputs]
    if not evidence_inputs:
        for key in selected:
            evidence = _latest_evidence_for_suite(root, RESEARCH_SUITES[key].stem)
            if evidence is not None:
                evidence_inputs.append(evidence)
    command_results = [
        {
            "command": "report existing evidence",
            "returncode": 0,
            "stdout": f"inputs={len(evidence_inputs)}",
            "stderr": "",
        }
    ]
    return _write_pack(
        root,
        out_dir,
        evidence_inputs,
        command_results=command_results,
        selected_suite_keys=selected,
        generation_mode="report",
    )


def _write_pack(
    root: Path,
    out_dir: Path,
    evidence_inputs: list[Path],
    *,
    command_results: list[JsonDict],
    selected_suite_keys: list[str],
    generation_mode: str = "report",
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_result_records(evidence_inputs) if evidence_inputs else []
    planner_semantics = planner_semantic_context(records)
    if planner_semantics["issues"]:
        raise ValueError(
            "planner semantic versions are mixed: "
            + "; ".join(planner_semantics["issues"])
        )
    boundary = validate_artifact_boundaries(root)
    guard_issues = _claim_guard_issues(records)
    manifest = build_manifest(
        root,
        command_results=command_results,
        evidence_inputs=evidence_inputs,
        record_count=len(records),
        selected_suite_keys=selected_suite_keys,
        generation_mode=generation_mode,
        planner_semantics=planner_semantics,
    )
    _write_json(out_dir / "benchmark_manifest.json", manifest)
    stats_rows = per_case_route_stats(records)
    speedup_rows = paired_speedups(records)
    cpu_gpu_rows = cpu_gpu_performance_summary(speedup_rows)
    full_state_tn_rows = full_state_tn_comparison(stats_rows)
    slicing_tradeoff_rows = slicing_tradeoff(stats_rows)
    quantization_rows = upmem_quantization_attribution(records)
    physical_quantization_rows = upmem_physical_quantization_attribution(records)
    physical_taskgraph_rows = upmem_physical_taskgraph_breakdown(records)
    hardware_mvp_rows = upmem_hardware_mvp_summary(records)
    hardware_generic_mvp_rows = upmem_hardware_generic_mvp_summary(records)
    same_plan_rows = same_plan_execution(records)
    planner_rows = planner_comparison(records)
    unsupported_rows = unsupported_cases(records)
    validation_rows = validation_summary(records)
    capability_rows = route_capability_matrix(records)
    planner_component_rows = planner_component_diagnostics(planner_rows)
    _write_csv(
        out_dir / "per_case_route_stats.csv", stats_rows, PER_CASE_ROUTE_STATS_FIELDS
    )
    _write_csv(out_dir / "paired_speedups.csv", speedup_rows, PAIRED_SPEEDUP_FIELDS)
    _write_csv(
        out_dir / "cpu_gpu_performance_summary.csv",
        cpu_gpu_rows,
        CPU_GPU_PERFORMANCE_SUMMARY_FIELDS,
    )
    _write_csv(
        out_dir / "full_state_tn_comparison.csv",
        full_state_tn_rows,
        FULL_STATE_TN_COMPARISON_FIELDS,
    )
    _write_csv(
        out_dir / "cpu_tn_slicing_tradeoff.csv",
        slicing_tradeoff_rows,
        SLICING_TRADEOFF_FIELDS,
    )
    _write_csv(
        out_dir / "upmem_quantization_attribution.csv",
        quantization_rows,
        UPMEM_QUANTIZATION_ATTRIBUTION_FIELDS,
    )
    _write_csv(
        out_dir / "upmem_physical_quantization_attribution.csv",
        physical_quantization_rows,
        UPMEM_PHYSICAL_QUANTIZATION_FIELDS,
    )
    _write_csv(
        out_dir / "upmem_physical_taskgraph_breakdown.csv",
        physical_taskgraph_rows,
        UPMEM_PHYSICAL_TASKGRAPH_FIELDS,
    )
    _write_csv(
        out_dir / "upmem_hardware_mvp_summary.csv",
        hardware_mvp_rows,
        UPMEM_HARDWARE_MVP_FIELDS,
    )
    _write_csv(
        out_dir / "upmem_hardware_generic_mvp_summary.csv",
        hardware_generic_mvp_rows,
        UPMEM_HARDWARE_MVP_FIELDS,
    )
    _write_csv(
        out_dir / "same_plan_execution.csv", same_plan_rows, SAME_PLAN_EXECUTION_FIELDS
    )
    _write_csv(
        out_dir / "planner_comparison.csv", planner_rows, PLANNER_COMPARISON_FIELDS
    )
    _write_csv(
        out_dir / "planner_component_diagnostics.csv",
        planner_component_rows,
        PLANNER_COMPONENT_DIAGNOSTICS_FIELDS,
    )
    _write_csv(out_dir / "unsupported_cases.csv", unsupported_rows, UNSUPPORTED_FIELDS)
    _write_csv(
        out_dir / "validation_summary.csv", validation_rows, VALIDATION_SUMMARY_FIELDS
    )
    _write_csv(
        out_dir / "route_capability_matrix.csv",
        capability_rows,
        ROUTE_CAPABILITY_FIELDS,
    )
    plot_manifest = write_plots(
        out_dir,
        stats_rows,
        cpu_gpu_rows,
        quantization_rows,
        same_plan_rows,
        planner_rows,
        slicing_tradeoff_rows,
        planner_component_rows,
        hardware_mvp_rows,
        hardware_generic_mvp_rows,
        physical_quantization_rows,
        physical_taskgraph_rows,
    )
    _write_json(out_dir / "plot_manifest.json", plot_manifest)
    (out_dir / "benchmark_summary.md").write_text(
        benchmark_summary(
            manifest,
            records,
            stats_rows,
            speedup_rows,
            quantization_rows,
            planner_rows,
            full_state_tn_rows,
            unsupported_rows,
            validation_rows,
            capability_rows,
            plot_manifest,
            boundary,
            guard_issues,
            hardware_mvp_rows,
            hardware_generic_mvp_rows,
            physical_quantization_rows,
            physical_taskgraph_rows,
        ),
        encoding="utf-8",
    )
    if _is_within(out_dir, root / "runs" / "comparisons"):
        _update_latest_link(out_dir.parent, out_dir)
    print(out_dir)
    return 1 if boundary["status"] == "failed" or guard_issues else 0


def build_manifest(
    root: Path,
    *,
    command_results: list[JsonDict],
    evidence_inputs: list[Path],
    record_count: int,
    selected_suite_keys: list[str],
    generation_mode: str = "report",
    planner_semantics: JsonDict | None = None,
) -> JsonDict:
    command_line = " ".join(sys.argv)
    generated_at = datetime.now().isoformat(timespec="seconds")
    source = _evidence_source_provenance(evidence_inputs)
    report_commit = _git(root, ["rev-parse", "HEAD"])
    report_dirty = bool(_git(root, ["status", "--short", "--", "."]))
    report_repository_dirty = bool(_git(root, ["status", "--short"]))
    provenance = {
        "generator": RESEARCH_PACK_KIND,
        "script": "scripts/research_benchmark_pack.py",
        "schema_version": SCHEMA_VERSION,
        "mode": generation_mode,
        "generated_at": generated_at,
        "command_line": command_line,
        "input_count": len(evidence_inputs),
        "input_paths": [path.as_posix() for path in evidence_inputs],
        "source_records_read_only": True,
        "commit": report_commit,
        "worktree_dirty": report_dirty,
        "repository_worktree_dirty": report_repository_dirty,
        "scope": "thesis/implementation",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": RESEARCH_PACK_KIND,
        "created_at": generated_at,
        "root": root.as_posix(),
        "git_commit": source["commit"],
        "dirty_tree": source["worktree_dirty"],
        "dirty_worktree": source["worktree_dirty"],
        "benchmark_source_commit": source["commit"],
        "benchmark_source_commits": source["commits"],
        "benchmark_source_worktree_dirty": source["worktree_dirty"],
        "repository_worktree_dirty": source["repository_worktree_dirty"],
        "provenance_scope": "thesis/implementation",
        "report_generation_commit": report_commit,
        "report_generation_worktree_dirty": report_dirty,
        "report_generation_repository_worktree_dirty": report_repository_dirty,
        "command_line": command_line,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "quimb", "cotengra", "opt_einsum", "matplotlib")
        },
        "cpu": _cpu_metadata(),
        "environment": {name: os.environ.get(name) for name in RELEVANT_ENV_VARS},
        "selected_suites": {
            key: _display_path(RESEARCH_SUITES[key], root)
            for key in selected_suite_keys
        },
        "evidence_inputs": [path.as_posix() for path in evidence_inputs],
        "record_count": record_count,
        "planner_semantics": planner_semantics or planner_semantic_context([]),
        "commands": command_results,
        "report_generation_provenance": provenance,
        "report_generation": provenance,
        "report_generation_script": provenance["script"],
        "report_generation_mode": generation_mode,
        "report_generation_timestamp": generated_at,
        "report_generation_command": command_line,
        "report_generation_input_paths": [path.as_posix() for path in evidence_inputs],
        "gpu_verification": _read_optional_json(
            root / "build" / "gpu_verification" / "quest_gpu_full_state_exact.json"
        ),
        "notes": {
            "long_runs_require_explicit_opt_in": True,
            "derived_outputs_are_under_runs_comparisons": True,
            "evidence_inputs_are_read_only": True,
        },
    }


def planner_semantic_context(records: list[JsonDict]) -> JsonDict:
    """Return the immutable semantic context for planner-derived evidence.

    The report may combine profiles within one objective, but cannot compare
    candidate rows generated with different objective or legacy score models.
    Older non-planner runs remain reportable unchanged.
    """
    rows = [
        record
        for record in records
        if record.get("route_id") == "planner_candidate_model"
    ]
    contexts: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    objective_versions: set[str] = set()
    score_models: set[str] = set()
    semantic_versions: set[str] = set()
    config_hashes: set[str] = set()
    profile_ids: set[str] = set()
    policy_ids: set[str] = set()
    normalization_ids: set[str] = set()
    for row in rows:
        objective = _nonempty_text(row.get("pim_objective_version"))
        score_model = _nonempty_text(row.get("score_model"))
        if objective:
            objective_versions.add(objective)
        if score_model:
            score_models.add(score_model)
        # The score model is the legacy semantic identifier.  Once an
        # objective version is present it is authoritative for that row, but
        # a row carrying only score_model remains comparable only with the
        # same legacy semantic family.
        semantic_version = objective or (
            f"legacy_score_model:{score_model}" if score_model else None
        )
        if semantic_version:
            semantic_versions.add(semantic_version)
        config_hash = _nonempty_text(
            row.get("planner_config_hash")
            or row.get("config_hash")
            or row.get("executor_config_hash")
        )
        if config_hash:
            config_hashes.add(config_hash)
        profile = _nonempty_text(row.get("pim_weight_profile"))
        if profile:
            profile_ids.add(profile)
        policy = _planner_policy_id(row.get("pim_execution_policy"))
        if policy:
            policy_ids.add(policy)
        normalization = _planner_normalization_id(row.get("pim_normalization"))
        if normalization:
            normalization_ids.add(normalization)
        contexts.setdefault(
            (objective, score_model),
            {
                "objective_version": objective,
                "score_model": score_model,
                "record_count": 0,
            },
        )["record_count"] += 1
    issues: list[str] = []
    if len(objective_versions) > 1:
        issues.append("pim_objective_version=" + ",".join(sorted(objective_versions)))
    if len(score_models) > 1:
        issues.append("score_model=" + ",".join(sorted(score_models)))
    if len(semantic_versions) > 1:
        issues.append("semantic_version=" + ",".join(sorted(semantic_versions)))
    return {
        "planner_record_count": len(rows),
        "objective_versions": sorted(objective_versions),
        "score_models": sorted(score_models),
        "legacy_score_models": sorted(score_models),
        "semantic_versions": sorted(semantic_versions),
        "planner_config_hashes": sorted(config_hashes),
        "weight_profiles": sorted(profile_ids),
        "execution_policies": sorted(policy_ids),
        "normalizations": sorted(normalization_ids),
        "contexts": sorted(
            contexts.values(),
            key=lambda item: (str(item["objective_version"]), str(item["score_model"])),
        ),
        "issues": issues,
    }


def _nonempty_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _planner_policy_id(value: object) -> str | None:
    if isinstance(value, dict):
        return _nonempty_text(value.get("policy_id"))
    return None


def _planner_normalization_id(value: object) -> str | None:
    if isinstance(value, dict):
        return _nonempty_text(value.get("normalization_id"))
    return None


PER_CASE_ROUTE_STATS_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "circuit_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "route_id",
    "benchmark_role",
    "backend_family",
    "execution_model",
    "parallelism_mode",
    "parallelism_evidence_type",
    "execution_plan_kind",
    "execution_plan_executed",
    "frontier_scheduler_enabled",
    "frontier_parallel_execution",
    "frontier_worker_count",
    "frontier_wave_count",
    "max_frontier_width",
    "mean_frontier_width",
    "frontier_executed_task_count",
    "source_frontier_completed_task_count",
    "frontier_executed_parallel_task_count",
    "executed_parallel_task_count",
    "source_task_count",
    "source_task_completion_count",
    "circuit_semantics_hash",
    "tensor_network_hash",
    "contraction_plan_hash",
    "contraction_execution_target",
    "upmem_execution_mode",
    "policy",
    "quantization_mode",
    "generic_only_all_tasks_used_generic_backend",
    "valid_primary_upmem_codepath_result",
    "upmem_program_executed",
    "dpu_program_invocations",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "timing_scope",
    "repeat_count",
    "total_wall_time_s_median",
    "total_wall_time_s_p25",
    "total_wall_time_s_p75",
    "total_wall_time_s_iqr",
    "total_wall_time_s_mean",
    "total_wall_time_s_min",
    "total_wall_time_s_max",
    "total_wall_time_s_std",
    "total_host_residual_time_s_median",
    "total_host_residual_time_s_p25",
    "total_host_residual_time_s_p75",
    "total_host_residual_time_s_iqr",
    "total_host_residual_time_s_mean",
    "total_host_residual_time_s_min",
    "total_host_residual_time_s_max",
    "total_host_residual_time_s_std",
    "reported_time_s_median",
    "reported_time_s_mean",
    "reported_time_s_min",
    "reported_time_s_max",
    "reported_time_s_std",
    "timing_basis",
    "energy_joules",
    "energy_source",
    "energy_measurement_status",
    "simulation_compute_time_s_median",
    "simulation_compute_time_s_p25",
    "simulation_compute_time_s_p75",
    "simulation_compute_time_s_iqr",
    "simulation_compute_time_s_mean",
    "simulation_compute_time_s_min",
    "simulation_compute_time_s_max",
    "simulation_compute_time_s_std",
    "planning_time_s_median",
    "planning_time_s_mean",
    "planning_time_s_min",
    "planning_time_s_max",
    "planning_time_s_std",
    "actual_transfer_bytes_median",
    "actual_h2d_bytes_median",
    "actual_d2h_bytes_median",
    "transfer_accounting_scope",
    "actual_transfer_bytes_invariant",
    "tn_estimated_flops",
    "tn_max_intermediate_bytes",
    "validation_passed_count",
    "validation_failed_count",
    "unsupported_count",
    "slice_count",
    "slicing_flop_ratio",
    "slicing_flop_change_kind",
    "max_abs_error",
    "l2_error",
    "execution_max_abs_error",
    "execution_l2_error",
    "full_precision_max_abs_error",
    "full_precision_l2_error",
    "hardware_speedup_applicable",
    "gpu_backend_verified",
    "gpu_device_name",
    "cpu_fallback_used",
    "resource_skip_reason",
]

UPMEM_QUANTIZATION_ATTRIBUTION_FIELDS = [
    "schema_version",
    "run_id",
    "suite_id",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "repeat_id",
    "route_id",
    "policy",
    "same_route_comparison",
    "same_taskgraph",
    "same_kernel_family",
    "unquantized_total_wall_time_s",
    "quantized_total_wall_time_s",
    "unquantized_host_residual_time_s",
    "quantized_host_residual_time_s",
    "route_runtime_ratio_none_over_quantized",
    "unquantized_simulation_compute_time_s",
    "quantized_simulation_compute_time_s",
    "simulator_kernel_ratio_none_over_quantized",
    "unquantized_transfer_bytes",
    "quantized_transfer_bytes",
    "unquantized_h2d_bytes",
    "quantized_h2d_bytes",
    "unquantized_d2h_bytes",
    "quantized_d2h_bytes",
    "transfer_accounting_scope",
    "actual_transfer_bytes_invariant",
    "transfer_ratio_none_over_quantized",
    "unquantized_max_abs_error_vs_full_precision",
    "quantized_max_abs_error_vs_full_precision",
    "unquantized_execution_max_abs_error",
    "quantized_execution_max_abs_error",
    "unquantized_full_precision_max_abs_error",
    "quantized_full_precision_max_abs_error",
    "unquantized_probability_max_abs_error",
    "quantized_probability_max_abs_error",
    "unquantized_probability_l1_error",
    "quantized_probability_l1_error",
    "unquantized_quantization_clipping_count",
    "quantized_quantization_clipping_count",
    "unquantized_quantization_saturation_count",
    "quantized_quantization_saturation_count",
    "accuracy_delta_quantized_minus_unquantized",
    "native_unquantized_upmem_kernel_executed",
]

UPMEM_HARDWARE_MVP_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "route_id",
    "backend_id",
    "hardware_profile_version",
    "execution_class",
    "kernel_strategy",
    "repeat_count",
    "completed_count",
    "validation_passed_count",
    "exact_integer_match_count",
    "hardware_execution_count",
    "hardware_allocation_verified_count",
    "no_simulator_fallback_count",
    "no_cpu_fallback_count",
    "requested_dpu_count",
    "allocated_dpu_count",
    "tasklets_per_dpu",
    "application_visible_h2d_bytes_median",
    "application_visible_d2h_bytes_median",
    "application_visible_transfer_bytes_median",
    "timing_scope",
    "hardware_speedup_applicable",
    "functionality_evidence_status",
]

UPMEM_PHYSICAL_QUANTIZATION_FIELDS = [
    "schema_version",
    "suite_id",
    "run_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "repeat_id",
    "contraction_plan_hash",
    "float32_route_id",
    "int8_route_id",
    "same_route",
    "same_taskgraph",
    "same_plan_verified",
    "float32_warm_runtime_s",
    "int8_warm_runtime_s",
    "warm_runtime_ratio_float32_over_int8",
    "float32_transfer_bytes",
    "int8_transfer_bytes",
    "float32_h2d_bytes",
    "int8_h2d_bytes",
    "float32_d2h_bytes",
    "int8_d2h_bytes",
    "transfer_ratio_float32_over_int8",
    "float32_max_abs_error",
    "int8_max_abs_error",
    "quantization_error_int8_vs_float32",
    "float32_timing_class",
    "int8_timing_class",
    "hardware_speedup_applicable",
]

UPMEM_PHYSICAL_TASKGRAPH_FIELDS = [
    "schema_version",
    "suite_id",
    "run_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "repeat_id",
    "contraction_plan_hash",
    "route_id",
    "quantization_mode",
    "input_dtype_on_dpu",
    "complex_policy",
    "hardware_numeric_coverage",
    "split_complex_component_count",
    "status",
    "validation_status",
    "validation_passed",
    "validation_max_abs_error",
    "full_precision_max_abs_error",
    "quantization_max_abs_error",
    "exact_integer_match",
    "task_count",
    "validated_task_count",
    "unsupported_task_count",
    "hardware_execution",
    "hardware_kernel_executed",
    "simulator_kernel_executed",
    "cpu_fallback_used",
    "timing_class",
    "hardware_timing_available",
    "timing_is_bringup_only",
    "allocation_time_s",
    "binary_load_time_s",
    "h2d_time_s",
    "d2h_time_s",
    "application_visible_h2d_bytes",
    "application_visible_d2h_bytes",
    "application_visible_transfer_bytes",
    "total_route_time_s",
    "warm_runtime_s",
    "total_quantization_time_s",
    "total_dequantization_time_s",
    "total_bridge_time_s",
    "total_build_time_s",
    "validation_time_s",
    "output_materialization_time_s",
    "hardware_speedup_applicable",
]

PAIRED_SPEEDUP_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "repeat_id",
    "cpu_route_id",
    "gpu_route_id",
    "timing_scope",
    "state_output_mode",
    "validation_method",
    "performance_tier",
    "cpu_total_wall_time_s",
    "gpu_total_wall_time_s",
    "wall_time_ratio_cpu_over_gpu",
    "cpu_total_host_residual_time_s",
    "gpu_total_host_residual_time_s",
    "cpu_simulation_compute_time_s",
    "gpu_simulation_compute_time_s",
    "compute_speedup_cpu_over_gpu",
    "gpu_device_name",
]

CPU_GPU_PERFORMANCE_SUMMARY_FIELDS = [
    "schema_version",
    "case_id",
    "case_family",
    "n_qubits",
    "actual_n_qubits",
    "benchmark_n_qubits",
    "actual_n_qubits_source",
    "actual_n_qubits_warning",
    "matched_repeat_count",
    "timing_scope",
    "cpu_simulation_compute_time_s_median",
    "cpu_simulation_compute_time_s_p25",
    "cpu_simulation_compute_time_s_p75",
    "cpu_simulation_compute_time_s_iqr",
    "gpu_simulation_compute_time_s_median",
    "gpu_simulation_compute_time_s_p25",
    "gpu_simulation_compute_time_s_p75",
    "gpu_simulation_compute_time_s_iqr",
    "compute_speedup_cpu_over_gpu_median",
    "compute_speedup_cpu_over_gpu_p25",
    "compute_speedup_cpu_over_gpu_p75",
    "compute_speedup_cpu_over_gpu_iqr",
    "cpu_total_wall_time_s_median",
    "cpu_total_wall_time_s_p25",
    "cpu_total_wall_time_s_p75",
    "cpu_total_wall_time_s_iqr",
    "gpu_total_wall_time_s_median",
    "gpu_total_wall_time_s_p25",
    "gpu_total_wall_time_s_p75",
    "gpu_total_wall_time_s_iqr",
    "wall_time_ratio_cpu_over_gpu_median",
    "wall_time_ratio_cpu_over_gpu_p25",
    "wall_time_ratio_cpu_over_gpu_p75",
    "wall_time_ratio_cpu_over_gpu_iqr",
    "cpu_total_host_residual_time_s_median",
    "cpu_total_host_residual_time_s_p25",
    "cpu_total_host_residual_time_s_p75",
    "cpu_total_host_residual_time_s_iqr",
    "gpu_total_host_residual_time_s_median",
    "gpu_total_host_residual_time_s_p25",
    "gpu_total_host_residual_time_s_p75",
    "gpu_total_host_residual_time_s_iqr",
    "compute_speedup_cpu_over_gpu_crossover_qubit",
    "crossover_qubit",
    "gpu_device_name",
]

FULL_STATE_TN_COMPARISON_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "workload_kind",
    "not_real_quantum_circuit",
    "planner_motif",
    "network_tensor_count",
    "network_index_count",
    "network_max_rank",
    "network_max_tensor_elements",
    "network_size_proxy",
    "quest_cpu_compute_time_s_median",
    "quimb_unsliced_compute_time_s_median",
    "quimb_sliced_compute_time_s_median",
    "quimb_unsliced_time_over_quest_time",
    "quimb_sliced_time_over_quest_time",
    "quimb_sliced_time_over_unsliced_time",
    "quest_validation_passed_count",
    "quimb_unsliced_validation_passed_count",
    "quimb_sliced_validation_passed_count",
]

SLICING_TRADEOFF_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "timing_scope",
    "state_output_mode",
    "performance_tier",
    "unsliced_compute_time_s_median",
    "sliced_compute_time_s_median",
    "runtime_ratio_sliced_over_unsliced",
    "slicing_flop_ratio",
    "unsliced_tn_max_intermediate_bytes",
    "sliced_tn_max_intermediate_bytes",
    "largest_intermediate_ratio_sliced_over_unsliced",
]

UNSUPPORTED_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "route_id",
    "benchmark_role",
    "validation_status",
    "status",
    "unsupported_task_count",
    "resource_skip_reason",
    "warnings",
]

VALIDATION_SUMMARY_FIELDS = [
    "schema_version",
    "route_id",
    "validation_status",
    "record_count",
]

ROUTE_CAPABILITY_FIELDS = [
    "schema_version",
    "route_id",
    "benchmark_role",
    "backend_family",
    "execution_model",
    "contraction_execution_target",
    "accelerator_kind",
    "upmem_execution_mode",
    "record_count",
    "completed_count",
    "unsupported_count",
    "validation_passed_count",
    "gpu_verified_count",
    "cpu_fallback_count",
    "hardware_speedup_applicable_count",
]

SAME_PLAN_EXECUTION_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "contraction_plan_hash",
    "cpu_route_id",
    "upmem_route_id",
    "quantization_mode",
    "cpu_time_s",
    "upmem_simulator_time_s",
    "route_time_ratio_cpu_over_upmem_simulator",
    "actual_transfer_bytes",
    "max_abs_error",
    "same_plan_verified",
    "hardware_speedup_applicable",
]

PLANNER_COMPARISON_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "workload_kind",
    "not_real_quantum_circuit",
    "planner_motif",
    "network_tensor_count",
    "network_index_count",
    "network_max_rank",
    "network_max_tensor_elements",
    "network_size_proxy",
    "planner_id",
    "planner_engine",
    "planner_kind",
    "planner_config_hash",
    "planner_config",
    "planner_selection_scope",
    "optimize_mode",
    "objective",
    "cost_basis",
    "target_estimate_key",
    "target_estimate_model",
    "contraction_plan_hash",
    "contraction_path_structure_hash",
    "candidate_status",
    "candidate_failure_reason",
    "planning_time_s",
    "task_count",
    "tn_estimated_flops",
    "tn_max_intermediate_bytes",
    "total_host_to_dpu_bytes",
    "total_dpu_to_host_bytes",
    "total_mram_to_wram_bytes",
    "unsupported_task_count",
    "tiling_required_task_count",
    "missing_target_estimate_count",
    "estimated_total_tile_count",
    "estimated_max_parallel_tiles",
    "upmem_pressure_score",
    "upmem_rank",
    "flop_rank",
    "score_model",
    "score_components",
    "score_weights",
    "tradeoff_note",
    "pim_objective_version",
    "pim_weight_profile",
    "pim_normalization",
    "pim_execution_policy",
    "pim_feasible",
    "pim_rejection_reasons",
    "pim_estimated_flops",
    "pim_peak_intermediate_bytes",
    "pim_total_intermediate_write_bytes",
    "pim_estimated_host_to_dpu_bytes",
    "pim_estimated_dpu_to_host_bytes",
    "pim_estimated_host_dpu_bytes",
    "pim_estimated_mram_to_wram_bytes",
    "pim_estimated_dpu_local_work",
    "pim_estimated_sync_events",
    "pim_estimated_numerical_penalty",
    "pim_estimated_wram_pressure",
    "pim_estimated_tile_count",
    "pim_largest_tensor_bytes",
    "pim_host_to_dpu_payload_bytes",
    "pim_dpu_to_host_payload_bytes",
    "pim_mram_dma_window_bytes_model",
    "pim_tile_iterations",
    "pim_host_completion_events",
    "pim_numeric_component_invocations",
    "pim_numeric_recombination_flops",
    "pim_task_mram_payload_bytes",
    "pim_native_static_mram_reservation_bytes",
    "pim_mram_capacity_bytes",
    "pim_mram_static_reservation_pressure_ratio",
    "pim_mram_max_region_payload_ratio",
    "pim_mram_payload_pressure_ratio",
    "pim_known_wram_static_bytes",
    "pim_wram_budget_bytes",
    "pim_wram_known_pressure_ratio",
    "pim_objective_components",
    "pim_normalized_components",
    "pim_objective_score",
    "pim_objective_rank",
    "pim_pareto_dominated",
    "pim_selected",
    "parallelism_evidence_type",
    "execution_plan_executed",
]

PLANNER_COMPONENT_DIAGNOSTICS_FIELDS = [
    "schema_version",
    "suite_id",
    "case_id",
    "case_family",
    "benchmark_n_qubits",
    "workload_kind",
    "not_real_quantum_circuit",
    "planner_motif",
    "planner_id",
    "planner_config_hash",
    "planner_selection_scope",
    "score_model",
    "pim_objective_version",
    "pim_weight_profile",
    "pim_feasible",
    "pim_objective_score",
    "pim_objective_rank",
    "pim_selected",
    "pim_estimated_flops",
    "pim_peak_intermediate_bytes",
    "pim_total_intermediate_write_bytes",
    "pim_estimated_host_dpu_bytes",
    "pim_estimated_mram_to_wram_bytes",
    "pim_estimated_tile_count",
    "pim_largest_tensor_bytes",
    "pim_host_to_dpu_payload_bytes",
    "pim_dpu_to_host_payload_bytes",
    "pim_mram_dma_window_bytes_model",
    "pim_tile_iterations",
    "pim_host_completion_events",
    "pim_numeric_component_invocations",
    "pim_numeric_recombination_flops",
    "pim_task_mram_payload_bytes",
    "pim_native_static_mram_reservation_bytes",
    "pim_mram_capacity_bytes",
    "pim_mram_static_reservation_pressure_ratio",
    "pim_mram_max_region_payload_ratio",
    "pim_mram_payload_pressure_ratio",
    "pim_known_wram_static_bytes",
    "pim_wram_budget_bytes",
    "pim_wram_known_pressure_ratio",
]


def per_case_route_stats(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[Any, ...], list[JsonDict]] = defaultdict(list)
    for record in records:
        key = (
            record.get("suite_id"),
            record.get("case_id"),
            record.get("route_id"),
            _record_value(record, "policy"),
            _record_value(record, "quantization_mode"),
            record.get("state_output_mode"),
            record.get("validation_method"),
            bool(record.get("performance_tier", False)),
            record.get("timing_scope"),
        )
        grouped[key].append(record)
    rows: list[JsonDict] = []
    for key, group in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        first = group[0]
        total_values = _numbers(row.get("total_wall_time_s") for row in group)
        residual_values = _numbers(
            row.get("total_host_residual_time_s") for row in group
        )
        reported_values = [_reported_time(row)[0] for row in group]
        reported_values = [value for value in reported_values if value is not None]
        compute_values = _numbers(row.get("simulation_compute_time_s") for row in group)
        planning_values = _numbers(row.get("planning_time_s") for row in group)
        transfer_values = _numbers(row.get("actual_transfer_bytes") for row in group)
        h2d_values = _numbers(row.get("actual_h2d_bytes") for row in group)
        d2h_values = _numbers(row.get("actual_d2h_bytes") for row in group)
        family, qubits = _family_and_qubits(first)
        errors = [_validation_errors(row) for row in group]
        full_precision_errors = [_full_precision_errors(row) for row in group]
        selected_accuracy_errors = [
            _accuracy_errors_for_reporting(row) for row in group
        ]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": first.get("suite_id"),
                "case_id": first.get("case_id"),
                "case_family": family,
                "n_qubits": qubits["actual_n_qubits"],
                **qubits,
                "route_id": first.get("route_id"),
                "benchmark_role": first.get("benchmark_role"),
                "backend_family": first.get("backend_family"),
                "execution_model": first.get("execution_model"),
                "parallelism_mode": first.get("parallelism_mode"),
                "parallelism_evidence_type": _first_present(
                    group, "parallelism_evidence_type"
                ),
                "execution_plan_kind": _first_present(group, "execution_plan_kind"),
                "execution_plan_executed": any(
                    bool(row.get("execution_plan_executed", False)) for row in group
                ),
                "frontier_scheduler_enabled": any(
                    bool(row.get("frontier_scheduler_enabled", False)) for row in group
                ),
                "frontier_parallel_execution": any(
                    bool(row.get("frontier_parallel_execution", False)) for row in group
                ),
                "frontier_worker_count": _first_present(group, "frontier_worker_count"),
                "frontier_wave_count": _max_number(
                    row.get("frontier_wave_count") for row in group
                ),
                "max_frontier_width": _max_number(
                    row.get("max_frontier_width") for row in group
                ),
                "mean_frontier_width": _first_present(group, "mean_frontier_width"),
                "frontier_executed_task_count": _max_number(
                    row.get("frontier_executed_task_count") for row in group
                ),
                "source_frontier_completed_task_count": _max_number(
                    row.get("source_frontier_completed_task_count") for row in group
                ),
                "frontier_executed_parallel_task_count": _max_number(
                    row.get("frontier_executed_parallel_task_count") for row in group
                ),
                "executed_parallel_task_count": _max_number(
                    row.get("executed_parallel_task_count") for row in group
                ),
                "source_task_count": _max_number(
                    row.get("source_task_count") for row in group
                ),
                "source_task_completion_count": _max_number(
                    row.get("source_task_completion_count") for row in group
                ),
                "circuit_semantics_hash": first.get("circuit_semantics_hash"),
                "tensor_network_hash": first.get("tensor_network_hash"),
                "contraction_plan_hash": first.get("contraction_plan_hash"),
                "contraction_execution_target": first.get(
                    "contraction_execution_target"
                ),
                "upmem_execution_mode": first.get("upmem_execution_mode"),
                "policy": _record_value(first, "policy"),
                "quantization_mode": _record_value(first, "quantization_mode"),
                "generic_only_all_tasks_used_generic_backend": _record_value(
                    first, "generic_only_all_tasks_used_generic_backend"
                ),
                "valid_primary_upmem_codepath_result": _record_value(
                    first, "valid_primary_upmem_codepath_result"
                ),
                "upmem_program_executed": _record_value(
                    first, "upmem_program_executed"
                ),
                "dpu_program_invocations": _record_value(
                    first, "dpu_program_invocations"
                ),
                "state_output_mode": first.get("state_output_mode"),
                "validation_method": first.get("validation_method"),
                "performance_tier": bool(first.get("performance_tier", False)),
                "timing_scope": first.get("timing_scope"),
                "repeat_count": len(group),
                **_stats("total_wall_time_s", total_values),
                **_stats("total_host_residual_time_s", residual_values),
                **_stats("reported_time_s", reported_values),
                "timing_basis": _reported_timing_basis(group),
                "energy_joules": _first_present(group, "energy_joules"),
                "energy_source": _first_present(group, "energy_source"),
                "energy_measurement_status": _first_present(
                    group, "energy_measurement_status"
                ),
                **_stats("simulation_compute_time_s", compute_values),
                **_stats("planning_time_s", planning_values),
                **_stats("actual_transfer_bytes", transfer_values),
                **_stats("actual_h2d_bytes", h2d_values),
                **_stats("actual_d2h_bytes", d2h_values),
                "transfer_accounting_scope": _first_present(
                    group, "transfer_accounting_scope"
                ),
                "actual_transfer_bytes_invariant": _transfer_invariant_status(group),
                "tn_estimated_flops": _first_present(group, "tn_estimated_flops"),
                "tn_max_intermediate_bytes": _first_present(
                    group, "tn_max_intermediate_bytes"
                ),
                "validation_passed_count": sum(
                    1
                    for row in group
                    if str(row.get("validation_status"))
                    in {"passed", "passed_native_status", "passed_runtime_only"}
                ),
                "validation_failed_count": sum(
                    1
                    for row in group
                    if str(row.get("validation_status"))
                    not in {
                        "passed",
                        "passed_native_status",
                        "passed_runtime_only",
                        "skipped",
                    }
                ),
                "unsupported_count": sum(1 for row in group if _is_unsupported(row)),
                "slice_count": _first_present(group, "slice_count"),
                "slicing_flop_ratio": _first_present(group, "slicing_flop_ratio"),
                "slicing_flop_change_kind": _first_present(
                    group, "slicing_flop_change_kind"
                ),
                "max_abs_error": _max_number(
                    error.get("max_abs_error") for error in selected_accuracy_errors
                ),
                "l2_error": _max_number(
                    error.get("l2_error") for error in selected_accuracy_errors
                ),
                "execution_max_abs_error": _max_number(
                    error.get("max_abs_error") for error in errors
                ),
                "execution_l2_error": _max_number(
                    error.get("l2_error") for error in errors
                ),
                "full_precision_max_abs_error": _max_number(
                    error.get("max_abs_error") for error in full_precision_errors
                ),
                "full_precision_l2_error": _max_number(
                    error.get("l2_error") for error in full_precision_errors
                ),
                "hardware_speedup_applicable": any(
                    bool(row.get("hardware_speedup_applicable", False)) for row in group
                ),
                "gpu_backend_verified": any(
                    bool(row.get("gpu_backend_verified", False)) for row in group
                ),
                "gpu_device_name": _first_present(group, "gpu_device_name"),
                "cpu_fallback_used": any(
                    bool(row.get("cpu_fallback_used", False)) for row in group
                ),
                "resource_skip_reason": _first_record_value(
                    group, "resource_skip_reason"
                )
                or _first_record_value(group, "reason"),
            }
        )
    return rows


def paired_speedups(records: list[JsonDict]) -> list[JsonDict]:
    cpu_records: dict[tuple[str, int], JsonDict] = {}
    gpu_records: dict[tuple[str, int], JsonDict] = {}
    for record in records:
        route_id = str(record.get("route_id") or "")
        repeat_id = _int_or_none(record.get("repeat_id"))
        case_id = str(record.get("case_id") or "")
        if repeat_id is None or not case_id:
            continue
        if route_id == "quest_cpu_full_state_exact":
            cpu_records[(case_id, repeat_id)] = record
        elif route_id == "quest_gpu_full_state_exact":
            gpu_records[(case_id, repeat_id)] = record
    rows: list[JsonDict] = []
    for key in sorted(set(cpu_records) & set(gpu_records)):
        cpu = cpu_records[key]
        gpu = gpu_records[key]
        if not _valid_for_pair(cpu) or not _valid_for_pair(gpu):
            continue
        if not (
            gpu.get("gpu_backend_verified") is True
            and gpu.get("gpu_program_executed") is True
        ):
            continue
        if str(cpu.get("state_output_mode") or "") != str(
            gpu.get("state_output_mode") or ""
        ):
            continue
        if str(cpu.get("validation_method") or "") != str(
            gpu.get("validation_method") or ""
        ):
            continue
        if bool(cpu.get("performance_tier", False)) != bool(
            gpu.get("performance_tier", False)
        ):
            continue
        cpu_total = _positive(cpu.get("total_wall_time_s"))
        gpu_total = _positive(gpu.get("total_wall_time_s"))
        cpu_compute = _positive(cpu.get("simulation_compute_time_s"))
        gpu_compute = _positive(gpu.get("simulation_compute_time_s"))
        if None in {cpu_total, gpu_total, cpu_compute, gpu_compute}:
            continue
        family, qubits = _family_and_qubits(cpu)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": key[0],
                "case_family": family,
                "n_qubits": qubits["actual_n_qubits"],
                **qubits,
                "repeat_id": key[1],
                "cpu_route_id": "quest_cpu_full_state_exact",
                "gpu_route_id": "quest_gpu_full_state_exact",
                "timing_scope": "performance_compute"
                if bool(cpu.get("performance_tier", False))
                else "correctness_wall_and_compute",
                "state_output_mode": cpu.get("state_output_mode"),
                "validation_method": cpu.get("validation_method"),
                "performance_tier": bool(cpu.get("performance_tier", False)),
                "cpu_total_wall_time_s": cpu_total,
                "gpu_total_wall_time_s": gpu_total,
                "wall_time_ratio_cpu_over_gpu": cpu_total / gpu_total,
                "cpu_total_host_residual_time_s": _float_or_none(
                    cpu.get("total_host_residual_time_s")
                ),
                "gpu_total_host_residual_time_s": _float_or_none(
                    gpu.get("total_host_residual_time_s")
                ),
                "cpu_simulation_compute_time_s": cpu_compute,
                "gpu_simulation_compute_time_s": gpu_compute,
                "compute_speedup_cpu_over_gpu": cpu_compute / gpu_compute,
                "gpu_device_name": gpu.get("gpu_device_name"),
            }
        )
    return rows


def cpu_gpu_performance_summary(speedup_rows: list[JsonDict]) -> list[JsonDict]:
    """Summarize matched performance-tier repeats for thesis CPU/GPU figures."""
    grouped: dict[tuple[str, str, int | None], list[JsonDict]] = defaultdict(list)
    for row in speedup_rows:
        if not _bool(row.get("performance_tier")):
            continue
        qubits = _plot_qubits(row)
        if qubits is None:
            continue
        grouped[
            (str(row.get("case_id") or ""), str(row.get("case_family") or ""), qubits)
        ].append(row)

    summary: list[JsonDict] = []
    for (case_id, family, qubits), group in sorted(grouped.items()):
        first = group[0]
        summary.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "case_family": family,
                "n_qubits": qubits,
                "actual_n_qubits": qubits,
                "benchmark_n_qubits": qubits,
                "actual_n_qubits_source": first.get("actual_n_qubits_source"),
                "actual_n_qubits_warning": first.get("actual_n_qubits_warning"),
                "matched_repeat_count": len(group),
                "timing_scope": "performance_compute",
                "gpu_device_name": _first_present(group, "gpu_device_name"),
                **_stats(
                    "cpu_simulation_compute_time_s",
                    _numbers(row.get("cpu_simulation_compute_time_s") for row in group),
                ),
                **_stats(
                    "gpu_simulation_compute_time_s",
                    _numbers(row.get("gpu_simulation_compute_time_s") for row in group),
                ),
                **_stats(
                    "compute_speedup_cpu_over_gpu",
                    _numbers(row.get("compute_speedup_cpu_over_gpu") for row in group),
                ),
                **_stats(
                    "cpu_total_wall_time_s",
                    _numbers(row.get("cpu_total_wall_time_s") for row in group),
                ),
                **_stats(
                    "gpu_total_wall_time_s",
                    _numbers(row.get("gpu_total_wall_time_s") for row in group),
                ),
                **_stats(
                    "wall_time_ratio_cpu_over_gpu",
                    _numbers(row.get("wall_time_ratio_cpu_over_gpu") for row in group),
                ),
                **_stats(
                    "cpu_total_host_residual_time_s",
                    _numbers(
                        row.get("cpu_total_host_residual_time_s") for row in group
                    ),
                ),
                **_stats(
                    "gpu_total_host_residual_time_s",
                    _numbers(
                        row.get("gpu_total_host_residual_time_s") for row in group
                    ),
                ),
            }
        )
    crossover_by_family: dict[str, int | str] = {}
    family_groups: dict[str, list[JsonDict]] = defaultdict(list)
    for row in summary:
        family_groups[str(row["case_family"])].append(row)
    for family, family_rows in family_groups.items():
        observed = sorted(
            int(row["n_qubits"])
            for row in family_rows
            if _plot_qubits(row) is not None
            and _positive(row.get("compute_speedup_cpu_over_gpu_median")) is not None
            and float(row["compute_speedup_cpu_over_gpu_median"]) > 1.0
        )
        crossover_by_family[family] = observed[0] if observed else "none_observed"
    for row in summary:
        row["compute_speedup_cpu_over_gpu_crossover_qubit"] = crossover_by_family[
            str(row["case_family"])
        ]
        row["crossover_qubit"] = row["compute_speedup_cpu_over_gpu_crossover_qubit"]
    return summary


def full_state_tn_comparison(stats_rows: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, str], dict[str, JsonDict]] = defaultdict(dict)
    for row in stats_rows:
        if row.get("suite_id") not in {"thesis_cpu_tn_quimb", "research_cpu_tn"}:
            continue
        route = str(row.get("route_id") or "")
        if route in {
            "quest_cpu_full_state_exact",
            "quimb_tn_exact",
            "quimb_tn_sliced_exact",
        }:
            grouped[(str(row.get("suite_id")), str(row.get("case_id")))][route] = row
    result: list[JsonDict] = []
    for (suite_id, case_id), routes in sorted(grouped.items()):
        quest = routes.get("quest_cpu_full_state_exact")
        unsliced = routes.get("quimb_tn_exact")
        sliced = routes.get("quimb_tn_sliced_exact")
        if quest is None or unsliced is None:
            continue
        quest_time = _positive(quest.get("simulation_compute_time_s_median"))
        unsliced_time = _positive(unsliced.get("simulation_compute_time_s_median"))
        sliced_time = (
            _positive(sliced.get("simulation_compute_time_s_median"))
            if sliced
            else None
        )
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": suite_id,
                "case_id": case_id,
                "case_family": quest.get("case_family"),
                "benchmark_n_qubits": quest.get("benchmark_n_qubits"),
                "quest_cpu_compute_time_s_median": quest_time,
                "quimb_unsliced_compute_time_s_median": unsliced_time,
                "quimb_sliced_compute_time_s_median": sliced_time,
                "quimb_unsliced_time_over_quest_time": _ratio(
                    unsliced_time, quest_time
                ),
                "quimb_sliced_time_over_quest_time": _ratio(sliced_time, quest_time),
                "quimb_sliced_time_over_unsliced_time": _ratio(
                    sliced_time, unsliced_time
                ),
                "quest_validation_passed_count": quest.get("validation_passed_count"),
                "quimb_unsliced_validation_passed_count": unsliced.get(
                    "validation_passed_count"
                ),
                "quimb_sliced_validation_passed_count": sliced.get(
                    "validation_passed_count"
                )
                if sliced
                else None,
            }
        )
    return result


def slicing_tradeoff(stats_rows: list[JsonDict]) -> list[JsonDict]:
    """Pair compatible external Quimb unsliced and sliced route statistics.

    The sliced planner reports its FLOP ratio against its corresponding
    unsliced plan.  Runtime and intermediate-size ratios are derived only from
    the two matched route rows, and are not interpreted as speedups.
    """
    grouped: dict[tuple[str, str, str, str, bool], dict[str, JsonDict]] = defaultdict(
        dict
    )
    for row in stats_rows:
        if row.get("suite_id") not in {"thesis_cpu_tn_quimb", "research_cpu_tn"}:
            continue
        route = str(row.get("route_id") or "")
        if route not in {"quimb_tn_exact", "quimb_tn_sliced_exact"}:
            continue
        key = (
            str(row.get("suite_id") or ""),
            str(row.get("case_id") or ""),
            str(row.get("timing_scope") or "unspecified"),
            str(row.get("state_output_mode") or "unspecified"),
            bool(row.get("performance_tier", False)),
        )
        grouped[key][route] = row

    result: list[JsonDict] = []
    for (
        suite_id,
        case_id,
        _timing_scope,
        _state_output_mode,
        _performance_tier,
    ), routes in sorted(grouped.items()):
        unsliced = routes.get("quimb_tn_exact")
        sliced = routes.get("quimb_tn_sliced_exact")
        if unsliced is None or sliced is None:
            continue
        unsliced_time = _positive(unsliced.get("simulation_compute_time_s_median"))
        sliced_time = _positive(sliced.get("simulation_compute_time_s_median"))
        flop_ratio = _positive(sliced.get("slicing_flop_ratio"))
        if unsliced_time is None or sliced_time is None or flop_ratio is None:
            continue
        result.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": suite_id,
                "case_id": case_id,
                "case_family": sliced.get("case_family") or unsliced.get("case_family"),
                "benchmark_n_qubits": sliced.get("benchmark_n_qubits")
                or unsliced.get("benchmark_n_qubits"),
                "timing_scope": sliced.get("timing_scope"),
                "state_output_mode": sliced.get("state_output_mode"),
                "performance_tier": bool(sliced.get("performance_tier", False)),
                "unsliced_compute_time_s_median": unsliced_time,
                "sliced_compute_time_s_median": sliced_time,
                "runtime_ratio_sliced_over_unsliced": _ratio(
                    sliced_time, unsliced_time
                ),
                "slicing_flop_ratio": flop_ratio,
                "unsliced_tn_max_intermediate_bytes": _positive(
                    unsliced.get("tn_max_intermediate_bytes")
                ),
                "sliced_tn_max_intermediate_bytes": _positive(
                    sliced.get("tn_max_intermediate_bytes")
                ),
                "largest_intermediate_ratio_sliced_over_unsliced": _ratio(
                    sliced.get("tn_max_intermediate_bytes"),
                    unsliced.get("tn_max_intermediate_bytes"),
                ),
            }
        )
    return result


def upmem_quantization_attribution(records: list[JsonDict]) -> list[JsonDict]:
    """Pair strict generic UPMEM float32 and int8 records from one TaskGraph.

    This is intentionally narrower than a generic route comparison: both records
    must prove the same generic-only SDK-simulator execution family.  Ratios are
    route-level simulator evidence, never hardware speedups.
    """
    grouped: dict[tuple[str, str, str, int, str], dict[str, JsonDict]] = defaultdict(
        dict
    )
    for record in records:
        if not _is_strict_generic_upmem_record(record):
            continue
        if str(record.get("status") or "") != "completed":
            continue
        mode = str(_record_value(record, "quantization_mode") or "")
        if mode not in {"none", "per_task_input_quantize"}:
            continue
        case_id = str(record.get("case_id") or "")
        if not case_id:
            continue
        repeat_id = _int_or_none(record.get("repeat_id"))
        grouped[
            (
                str(record.get("suite_id") or ""),
                str(record.get("run_id") or ""),
                case_id,
                0 if repeat_id is None else repeat_id,
                str(record.get("route_id") or ""),
            )
        ][mode] = record

    rows: list[JsonDict] = []
    for (suite_id, run_id, case_id, repeat_id, route_id), modes in sorted(
        grouped.items()
    ):
        unquantized = modes.get("none")
        quantized = modes.get("per_task_input_quantize")
        if unquantized is None or quantized is None:
            continue
        if str(_record_value(unquantized, "policy") or "") != "generic-only":
            continue
        if str(_record_value(quantized, "policy") or "") != "generic-only":
            continue
        if (
            str(unquantized.get("route_id") or "") != route_id
            or str(quantized.get("route_id") or "") != route_id
        ):
            continue
        if unquantized.get("kernel_family") != quantized.get("kernel_family"):
            continue
        family, qubits = _family_and_qubits(unquantized)
        unquantized_total = _positive(unquantized.get("total_wall_time_s"))
        quantized_total = _positive(quantized.get("total_wall_time_s"))
        unquantized_residual = _reported_time(unquantized)[0]
        quantized_residual = _reported_time(quantized)[0]
        unquantized_compute = _positive(unquantized.get("simulation_compute_time_s"))
        quantized_compute = _positive(quantized.get("simulation_compute_time_s"))
        unquantized_transfer = _positive(unquantized.get("actual_transfer_bytes"))
        quantized_transfer = _positive(quantized.get("actual_transfer_bytes"))
        unquantized_h2d = _positive(unquantized.get("actual_h2d_bytes"))
        quantized_h2d = _positive(quantized.get("actual_h2d_bytes"))
        unquantized_d2h = _positive(unquantized.get("actual_d2h_bytes"))
        quantized_d2h = _positive(quantized.get("actual_d2h_bytes"))
        unquantized_error = _validation_errors(unquantized).get("max_abs_error")
        quantized_error = _validation_errors(quantized).get("max_abs_error")
        unquantized_probability_max_abs_error = _validation_metric(
            unquantized, "probability_max_abs_error"
        )
        quantized_probability_max_abs_error = _validation_metric(
            quantized, "probability_max_abs_error"
        )
        unquantized_probability_l1_error = _validation_metric(
            unquantized, "probability_l1_error"
        )
        quantized_probability_l1_error = _validation_metric(
            quantized, "probability_l1_error"
        )
        unquantized_full_precision_error = _full_precision_errors(unquantized).get(
            "max_abs_error"
        )
        quantized_full_precision_error = _full_precision_errors(quantized).get(
            "max_abs_error"
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": suite_id or unquantized.get("suite_id"),
                "run_id": run_id or unquantized.get("run_id"),
                "case_id": case_id,
                "case_family": family,
                "n_qubits": qubits["actual_n_qubits"],
                **qubits,
                "repeat_id": repeat_id,
                "route_id": route_id,
                "policy": "generic-only",
                "same_route_comparison": True,
                "same_taskgraph": True,
                "same_kernel_family": True,
                "unquantized_total_wall_time_s": unquantized_total,
                "quantized_total_wall_time_s": quantized_total,
                "unquantized_host_residual_time_s": unquantized_residual,
                "quantized_host_residual_time_s": quantized_residual,
                "route_runtime_ratio_none_over_quantized": _ratio(
                    unquantized_residual, quantized_residual
                )
                if unquantized_residual is not None and quantized_residual is not None
                else _ratio(unquantized_total, quantized_total),
                "unquantized_simulation_compute_time_s": unquantized_compute,
                "quantized_simulation_compute_time_s": quantized_compute,
                "simulator_kernel_ratio_none_over_quantized": _ratio(
                    unquantized_compute, quantized_compute
                ),
                "unquantized_transfer_bytes": unquantized_transfer,
                "quantized_transfer_bytes": quantized_transfer,
                "unquantized_h2d_bytes": unquantized_h2d,
                "quantized_h2d_bytes": quantized_h2d,
                "unquantized_d2h_bytes": unquantized_d2h,
                "quantized_d2h_bytes": quantized_d2h,
                "transfer_accounting_scope": _same_or_mixed(
                    unquantized.get("transfer_accounting_scope"),
                    quantized.get("transfer_accounting_scope"),
                ),
                "actual_transfer_bytes_invariant": _paired_transfer_invariant_status(
                    unquantized, quantized
                ),
                "transfer_ratio_none_over_quantized": _ratio(
                    unquantized_transfer, quantized_transfer
                ),
                "unquantized_max_abs_error_vs_full_precision": _float_or_none(
                    unquantized_full_precision_error
                    if unquantized_full_precision_error is not None
                    else unquantized_error
                ),
                "quantized_max_abs_error_vs_full_precision": _float_or_none(
                    quantized_full_precision_error
                    if quantized_full_precision_error is not None
                    else quantized_error
                ),
                "unquantized_execution_max_abs_error": _float_or_none(
                    unquantized_error
                ),
                "quantized_execution_max_abs_error": _float_or_none(quantized_error),
                "unquantized_full_precision_max_abs_error": _float_or_none(
                    unquantized_full_precision_error
                ),
                "quantized_full_precision_max_abs_error": _float_or_none(
                    quantized_full_precision_error
                ),
                "unquantized_probability_max_abs_error": _float_or_none(
                    unquantized_probability_max_abs_error
                ),
                "quantized_probability_max_abs_error": _float_or_none(
                    quantized_probability_max_abs_error
                ),
                "unquantized_probability_l1_error": _float_or_none(
                    unquantized_probability_l1_error
                ),
                "quantized_probability_l1_error": _float_or_none(
                    quantized_probability_l1_error
                ),
                "unquantized_quantization_clipping_count": _int_or_none(
                    _record_value(unquantized, "quantization_clipping_count")
                ),
                "quantized_quantization_clipping_count": _int_or_none(
                    _record_value(quantized, "quantization_clipping_count")
                ),
                "unquantized_quantization_saturation_count": _int_or_none(
                    _record_value(unquantized, "quantization_saturation_count")
                ),
                "quantized_quantization_saturation_count": _int_or_none(
                    _record_value(quantized, "quantization_saturation_count")
                ),
                "accuracy_delta_quantized_minus_unquantized": _difference(
                    quantized_full_precision_error
                    if quantized_full_precision_error is not None
                    else quantized_error,
                    unquantized_full_precision_error
                    if unquantized_full_precision_error is not None
                    else unquantized_error,
                ),
                "native_unquantized_upmem_kernel_executed": _record_value(
                    unquantized, "native_unquantized_upmem_kernel_executed"
                )
                is True,
            }
        )
    return rows


def upmem_physical_quantization_attribution(records: list[JsonDict]) -> list[JsonDict]:
    """Pair future physical TaskGraph float32/int8 rows without claiming speedup.

    Physical rows may keep the existing TaskGraph route and differ only in
    quantization mode.  The plan hash, case, repeat, run, and route must agree;
    missing hashes never become a same-plan match by inference.
    """
    grouped: dict[tuple[str, str, str, str, int, str], dict[str, JsonDict]] = (
        defaultdict(dict)
    )
    for record in records:
        if (
            not _is_physical_upmem_taskgraph_record(record)
            or str(record.get("status") or "") != "completed"
        ):
            continue
        dtype = _physical_taskgraph_dtype(record)
        plan_hash = str(record.get("contraction_plan_hash") or "")
        case_id = str(record.get("case_id") or "")
        route_id = str(record.get("route_id") or "")
        if dtype is None or not plan_hash or not case_id or not route_id:
            continue
        repeat_id = _int_or_none(record.get("repeat_id"))
        grouped[
            (
                str(record.get("suite_id") or ""),
                str(record.get("run_id") or ""),
                case_id,
                plan_hash,
                0 if repeat_id is None else repeat_id,
                route_id,
            )
        ][dtype] = record

    rows: list[JsonDict] = []
    for (suite_id, run_id, case_id, plan_hash, repeat_id, route_id), modes in sorted(
        grouped.items()
    ):
        float32 = modes.get("float32")
        int8 = modes.get("int8")
        if float32 is None or int8 is None:
            continue
        family, qubits = _family_and_qubits(float32)
        float32_transfer = _physical_transfer_bytes(float32)
        int8_transfer = _physical_transfer_bytes(int8)
        float32_error = _physical_error(float32)
        int8_error = _physical_error(int8)
        float32_runtime = _physical_warm_runtime(float32)
        int8_runtime = _physical_warm_runtime(int8)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": suite_id or float32.get("suite_id"),
                "run_id": run_id or float32.get("run_id"),
                "case_id": case_id,
                "case_family": family,
                "benchmark_n_qubits": qubits["benchmark_n_qubits"],
                "repeat_id": repeat_id,
                "contraction_plan_hash": plan_hash,
                "float32_route_id": route_id,
                "int8_route_id": route_id,
                "same_route": True,
                "same_taskgraph": True,
                "same_plan_verified": True,
                "float32_warm_runtime_s": float32_runtime,
                "int8_warm_runtime_s": int8_runtime,
                "warm_runtime_ratio_float32_over_int8": _ratio(
                    float32_runtime, int8_runtime
                ),
                "float32_transfer_bytes": float32_transfer,
                "int8_transfer_bytes": int8_transfer,
                "float32_h2d_bytes": _physical_directional_transfer(float32, "h2d"),
                "int8_h2d_bytes": _physical_directional_transfer(int8, "h2d"),
                "float32_d2h_bytes": _physical_directional_transfer(float32, "d2h"),
                "int8_d2h_bytes": _physical_directional_transfer(int8, "d2h"),
                "transfer_ratio_float32_over_int8": _ratio(
                    float32_transfer, int8_transfer
                ),
                "float32_max_abs_error": float32_error,
                "int8_max_abs_error": int8_error,
                "quantization_error_int8_vs_float32": _difference(
                    int8_error, float32_error
                ),
                "float32_timing_class": _physical_timing_class(float32),
                "int8_timing_class": _physical_timing_class(int8),
                "hardware_speedup_applicable": False,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["case_family"]),
            int(row.get("benchmark_n_qubits") or 0),
            int(row.get("repeat_id") or 0),
        ),
    )


def upmem_physical_taskgraph_breakdown(records: list[JsonDict]) -> list[JsonDict]:
    """Project physical TaskGraph validation and timing metadata verbatim."""
    rows: list[JsonDict] = []
    for record in records:
        if not _is_physical_upmem_taskgraph_record(record):
            continue
        family, qubits = _family_and_qubits(record)
        validation_status = str(record.get("validation_status") or "unknown")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": record.get("suite_id"),
                "run_id": record.get("run_id"),
                "case_id": record.get("case_id"),
                "case_family": family,
                "benchmark_n_qubits": qubits["benchmark_n_qubits"],
                "repeat_id": _int_or_none(record.get("repeat_id")),
                "contraction_plan_hash": record.get("contraction_plan_hash"),
                "route_id": record.get("route_id"),
                "quantization_mode": _record_value(record, "quantization_mode"),
                "input_dtype_on_dpu": _record_value(record, "input_dtype_on_dpu")
                or record.get("input_dtype"),
                "complex_policy": _record_value(record, "complex_policy"),
                "hardware_numeric_coverage": _record_value(
                    record, "hardware_numeric_coverage"
                ),
                "split_complex_component_count": _record_value(
                    record, "split_complex_component_count"
                ),
                "status": record.get("status"),
                "validation_status": validation_status,
                "validation_passed": validation_status
                in {"passed", "passed_native_status", "passed_runtime_only"},
                "validation_max_abs_error": _physical_number(
                    record, "validation_max_abs_error", "max_abs_error"
                ),
                "full_precision_max_abs_error": _physical_number(
                    record, "full_precision_max_abs_error"
                ),
                "quantization_max_abs_error": _physical_number(
                    record, "quantization_max_abs_error"
                ),
                "exact_integer_match": record.get("exact_integer_match"),
                "task_count": _record_value(record, "task_count")
                or _record_value(record, "upmem_task_count"),
                "validated_task_count": _record_value(record, "validated_task_count"),
                "unsupported_task_count": _record_value(
                    record, "unsupported_task_count"
                ),
                "hardware_execution": _physical_bool(record, "hardware_execution"),
                "hardware_kernel_executed": _physical_bool(
                    record, "hardware_kernel_executed"
                ),
                "simulator_kernel_executed": _physical_bool(
                    record, "simulator_kernel_executed"
                ),
                "cpu_fallback_used": bool(record.get("cpu_fallback_used", False)),
                "timing_class": _physical_timing_class(record),
                "hardware_timing_available": _physical_bool(
                    record, "hardware_timing_available"
                ),
                "timing_is_bringup_only": _physical_bool(
                    record, "timing_is_bringup_only"
                ),
                "allocation_time_s": _physical_number(record, "allocation_time_s"),
                "binary_load_time_s": _physical_number(record, "binary_load_time_s"),
                "h2d_time_s": _physical_number(record, "h2d_time_s"),
                "d2h_time_s": _physical_number(record, "d2h_time_s"),
                "application_visible_h2d_bytes": _physical_number(
                    record, "application_visible_h2d_bytes"
                ),
                "application_visible_d2h_bytes": _physical_number(
                    record, "application_visible_d2h_bytes"
                ),
                "application_visible_transfer_bytes": _physical_number(
                    record, "application_visible_transfer_bytes"
                ),
                "total_route_time_s": _physical_number(record, "total_route_time_s"),
                "warm_runtime_s": _physical_warm_runtime(record),
                "total_quantization_time_s": _physical_number(
                    record, "total_quantization_time_s"
                ),
                "total_dequantization_time_s": _physical_number(
                    record, "total_dequantization_time_s"
                ),
                "total_bridge_time_s": _physical_number(record, "total_bridge_time_s"),
                "total_build_time_s": _physical_number(
                    record, "total_build_time_s", "build_time_s"
                ),
                "validation_time_s": _physical_number(record, "validation_time_s"),
                "output_materialization_time_s": _physical_number(
                    record, "output_materialization_time_s"
                ),
                "hardware_speedup_applicable": False,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("case_family") or ""),
            int(row.get("benchmark_n_qubits") or 0),
            str(row.get("quantization_mode") or ""),
            int(row.get("repeat_id") or 0),
        ),
    )


def same_plan_execution(records: list[JsonDict]) -> list[JsonDict]:
    cpu_by_plan: dict[tuple[str, str, str], JsonDict] = {}
    upmem_rows: list[JsonDict] = []
    for record in records:
        plan_hash = str(record.get("contraction_plan_hash") or "")
        if not plan_hash:
            continue
        key = (
            str(record.get("suite_id") or ""),
            str(record.get("case_id") or ""),
            plan_hash,
        )
        if (
            record.get("route_id") == "cpu_tn_einsum_exact"
            and record.get("contraction_execution_target") == "cpu"
            and str(record.get("status") or "") == "completed"
        ):
            cpu_by_plan[key] = record
        elif (
            _is_strict_generic_upmem_record(record)
            and str(record.get("status") or "") == "completed"
        ):
            upmem_rows.append(record)

    rows: list[JsonDict] = []
    for upmem in upmem_rows:
        plan_hash = str(upmem["contraction_plan_hash"])
        key = (
            str(upmem.get("suite_id") or ""),
            str(upmem.get("case_id") or ""),
            plan_hash,
        )
        cpu = cpu_by_plan.get(key)
        if cpu is None:
            continue
        cpu_time = _positive(
            cpu.get("simulation_compute_time_s") or cpu.get("kernel_time_s")
        )
        upmem_time = _positive(
            upmem.get("simulation_compute_time_s") or upmem.get("kernel_time_s")
        )
        if cpu_time is None or upmem_time is None:
            continue
        family, qubits = _family_and_qubits(upmem)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": key[0],
                "case_id": key[1],
                "case_family": family,
                "benchmark_n_qubits": qubits["benchmark_n_qubits"],
                "contraction_plan_hash": plan_hash,
                "cpu_route_id": "cpu_tn_einsum_exact",
                "upmem_route_id": str(upmem.get("route_id") or "upmem_tn_runtime"),
                "quantization_mode": _record_value(upmem, "quantization_mode"),
                "cpu_time_s": cpu_time,
                "upmem_simulator_time_s": upmem_time,
                "route_time_ratio_cpu_over_upmem_simulator": cpu_time / upmem_time,
                "actual_transfer_bytes": upmem.get("actual_transfer_bytes"),
                "max_abs_error": upmem.get("max_abs_error")
                or _validation_errors(upmem).get("max_abs_error"),
                "same_plan_verified": True,
                "hardware_speedup_applicable": False,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["case_family"]),
            int(row["benchmark_n_qubits"] or 0),
            str(row["quantization_mode"]),
        ),
    )


def planner_comparison(records: list[JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for record in records:
        if record.get("route_id") != "planner_candidate_model":
            continue
        family, qubits = _family_and_qubits(record)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": record.get("suite_id"),
                "case_id": record.get("case_id"),
                "case_family": family,
                "benchmark_n_qubits": qubits["benchmark_n_qubits"],
                "workload_kind": record.get("workload_kind"),
                "not_real_quantum_circuit": record.get("not_real_quantum_circuit"),
                "planner_motif": record.get("planner_motif"),
                "network_tensor_count": record.get("network_tensor_count"),
                "network_index_count": record.get("network_index_count"),
                "network_max_rank": record.get("network_max_rank"),
                "network_max_tensor_elements": record.get(
                    "network_max_tensor_elements"
                ),
                "network_size_proxy": record.get("network_size_proxy"),
                "planner_id": record.get("planner_id") or record.get("backend_id"),
                "planner_engine": record.get("planner_engine"),
                "planner_kind": record.get("planner_kind"),
                "planner_config_hash": record.get("planner_config_hash")
                or record.get("config_hash")
                or record.get("executor_config_hash"),
                "planner_config": record.get("planner_config"),
                "planner_selection_scope": record.get("planner_selection_scope"),
                "optimize_mode": record.get("optimize_mode"),
                "objective": record.get("objective"),
                "cost_basis": record.get("cost_basis"),
                "target_estimate_key": record.get("target_estimate_key"),
                "target_estimate_model": record.get("target_estimate_model"),
                "contraction_plan_hash": record.get("contraction_plan_hash"),
                "contraction_path_structure_hash": record.get(
                    "contraction_path_structure_hash"
                ),
                "candidate_status": record.get("candidate_status"),
                "candidate_failure_reason": record.get("candidate_failure_reason"),
                "planning_time_s": record.get("planning_time_s"),
                "task_count": record.get("task_count"),
                "tn_estimated_flops": record.get("tn_estimated_flops"),
                "tn_max_intermediate_bytes": record.get("tn_max_intermediate_bytes"),
                "total_host_to_dpu_bytes": record.get("total_host_to_dpu_bytes"),
                "total_dpu_to_host_bytes": record.get("total_dpu_to_host_bytes"),
                "total_mram_to_wram_bytes": record.get("total_mram_to_wram_bytes"),
                "unsupported_task_count": record.get("unsupported_task_count"),
                "tiling_required_task_count": record.get("tiling_required_task_count"),
                "missing_target_estimate_count": record.get(
                    "missing_target_estimate_count"
                ),
                "estimated_total_tile_count": record.get("estimated_total_tile_count"),
                "estimated_max_parallel_tiles": record.get(
                    "estimated_max_parallel_tiles"
                ),
                "upmem_pressure_score": record.get("upmem_pressure_score"),
                "upmem_rank": record.get("upmem_rank"),
                "flop_rank": record.get("flop_rank"),
                "score_model": record.get("score_model"),
                "score_components": record.get("score_components"),
                "score_weights": record.get("score_weights"),
                "tradeoff_note": record.get("tradeoff_note"),
                "pim_objective_version": record.get("pim_objective_version"),
                "pim_weight_profile": record.get("pim_weight_profile"),
                "pim_normalization": record.get("pim_normalization"),
                "pim_execution_policy": record.get("pim_execution_policy"),
                "pim_feasible": record.get("pim_feasible"),
                "pim_rejection_reasons": record.get("pim_rejection_reasons"),
                "pim_estimated_flops": record.get("pim_estimated_flops"),
                "pim_peak_intermediate_bytes": record.get(
                    "pim_peak_intermediate_bytes"
                ),
                "pim_total_intermediate_write_bytes": record.get(
                    "pim_total_intermediate_write_bytes"
                ),
                "pim_estimated_host_to_dpu_bytes": record.get(
                    "pim_estimated_host_to_dpu_bytes"
                ),
                "pim_estimated_dpu_to_host_bytes": record.get(
                    "pim_estimated_dpu_to_host_bytes"
                ),
                "pim_estimated_host_dpu_bytes": record.get(
                    "pim_estimated_host_dpu_bytes"
                ),
                "pim_estimated_mram_to_wram_bytes": record.get(
                    "pim_estimated_mram_to_wram_bytes"
                ),
                "pim_estimated_dpu_local_work": record.get(
                    "pim_estimated_dpu_local_work"
                ),
                "pim_estimated_sync_events": record.get("pim_estimated_sync_events"),
                "pim_estimated_numerical_penalty": record.get(
                    "pim_estimated_numerical_penalty"
                ),
                "pim_estimated_wram_pressure": record.get(
                    "pim_estimated_wram_pressure"
                ),
                "pim_estimated_tile_count": record.get("pim_estimated_tile_count"),
                "pim_largest_tensor_bytes": record.get("pim_largest_tensor_bytes"),
                "pim_host_to_dpu_payload_bytes": record.get(
                    "pim_host_to_dpu_payload_bytes"
                ),
                "pim_dpu_to_host_payload_bytes": record.get(
                    "pim_dpu_to_host_payload_bytes"
                ),
                "pim_mram_dma_window_bytes_model": record.get(
                    "pim_mram_dma_window_bytes_model"
                ),
                "pim_tile_iterations": record.get("pim_tile_iterations"),
                "pim_host_completion_events": record.get("pim_host_completion_events"),
                "pim_numeric_component_invocations": record.get(
                    "pim_numeric_component_invocations"
                ),
                "pim_numeric_recombination_flops": record.get(
                    "pim_numeric_recombination_flops"
                ),
                "pim_task_mram_payload_bytes": record.get(
                    "pim_task_mram_payload_bytes"
                ),
                "pim_native_static_mram_reservation_bytes": record.get(
                    "pim_native_static_mram_reservation_bytes"
                ),
                "pim_mram_capacity_bytes": record.get("pim_mram_capacity_bytes"),
                "pim_mram_static_reservation_pressure_ratio": record.get(
                    "pim_mram_static_reservation_pressure_ratio"
                ),
                "pim_mram_max_region_payload_ratio": record.get(
                    "pim_mram_max_region_payload_ratio"
                ),
                "pim_mram_payload_pressure_ratio": record.get(
                    "pim_mram_payload_pressure_ratio"
                ),
                "pim_known_wram_static_bytes": record.get(
                    "pim_known_wram_static_bytes"
                ),
                "pim_wram_budget_bytes": record.get("pim_wram_budget_bytes"),
                "pim_wram_known_pressure_ratio": record.get(
                    "pim_wram_known_pressure_ratio"
                ),
                "pim_objective_components": record.get("pim_objective_components"),
                "pim_normalized_components": record.get("pim_normalized_components"),
                "pim_objective_score": record.get("pim_objective_score"),
                "pim_objective_rank": record.get("pim_objective_rank"),
                "pim_pareto_dominated": record.get("pim_pareto_dominated"),
                "pim_selected": record.get("pim_selected"),
                "parallelism_evidence_type": record.get("parallelism_evidence_type"),
                "execution_plan_executed": record.get("execution_plan_executed"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["case_family"]),
            int(row.get("benchmark_n_qubits") or 0),
            str(row.get("planner_id") or ""),
        ),
    )


def planner_component_diagnostics(planner_rows: list[JsonDict]) -> list[JsonDict]:
    """Project planner scoring components into a compact diagnostic table."""
    return [
        {field: row.get(field) for field in PLANNER_COMPONENT_DIAGNOSTICS_FIELDS}
        for row in planner_rows
    ]


def unsupported_cases(records: list[JsonDict]) -> list[JsonDict]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "suite_id": record.get("suite_id"),
            "case_id": record.get("case_id"),
            "route_id": record.get("route_id"),
            "benchmark_role": record.get("benchmark_role"),
            "validation_status": record.get("validation_status"),
            "status": record.get("status"),
            "unsupported_task_count": int(record.get("unsupported_task_count", 0) or 0),
            "resource_skip_reason": _record_value(record, "resource_skip_reason")
            or _record_value(record, "reason"),
            "warnings": record.get("warnings"),
        }
        for record in records
        if _is_unsupported(record)
    ]


def validation_summary(records: list[JsonDict]) -> list[JsonDict]:
    counts = Counter(
        (
            str(record.get("route_id") or ""),
            str(record.get("validation_status") or "unknown"),
        )
        for record in records
    )
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "route_id": route_id,
            "validation_status": status,
            "record_count": count,
        }
        for (route_id, status), count in sorted(counts.items())
    ]


def route_capability_matrix(records: list[JsonDict]) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("route_id") or "unknown")].append(record)
    rows: list[JsonDict] = []
    for route_id, group in sorted(grouped.items()):
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route_id": route_id,
                "benchmark_role": _first_present(group, "benchmark_role"),
                "backend_family": _first_present(group, "backend_family"),
                "execution_model": _first_present(group, "execution_model"),
                "contraction_execution_target": _first_present(
                    group, "contraction_execution_target"
                ),
                "accelerator_kind": _first_present(group, "accelerator_kind"),
                "upmem_execution_mode": _first_present(group, "upmem_execution_mode"),
                "record_count": len(group),
                "completed_count": sum(
                    1
                    for row in group
                    if str(row.get("status")) in {"completed", "executable"}
                ),
                "unsupported_count": sum(1 for row in group if _is_unsupported(row)),
                "validation_passed_count": sum(
                    1
                    for row in group
                    if str(row.get("validation_status"))
                    in {"passed", "passed_native_status", "passed_runtime_only"}
                ),
                "gpu_verified_count": sum(
                    1 for row in group if bool(row.get("gpu_backend_verified", False))
                ),
                "cpu_fallback_count": sum(
                    1 for row in group if bool(row.get("cpu_fallback_used", False))
                ),
                "hardware_speedup_applicable_count": sum(
                    1
                    for row in group
                    if bool(row.get("hardware_speedup_applicable", False))
                ),
            }
        )
    return rows


def upmem_hardware_mvp_summary(records: list[JsonDict]) -> list[JsonDict]:
    """Summarize fixed dense physical-MVP functionality evidence."""

    return _upmem_hardware_functionality_summary(records, _is_hardware_mvp_record)


def upmem_hardware_generic_mvp_summary(records: list[JsonDict]) -> list[JsonDict]:
    """Summarize the separate synthetic generic TaskGraph physical MVP."""

    return _upmem_hardware_functionality_summary(
        records, _is_hardware_generic_mvp_record
    )


def _upmem_hardware_functionality_summary(
    records: list[JsonDict],
    predicate: Callable[[JsonDict], bool],
) -> list[JsonDict]:
    """Summarize one physical functionality evidence class without timing claims."""

    grouped: dict[tuple[str, str, str], list[JsonDict]] = defaultdict(list)
    for record in records:
        if not predicate(record):
            continue
        grouped[
            (
                str(record.get("suite_id") or ""),
                str(record.get("case_id") or ""),
                str(record.get("route_id") or ""),
            )
        ].append(record)

    rows: list[JsonDict] = []
    for (_, _, _), group in sorted(grouped.items()):
        first = group[0]
        h2d_values = _numbers(row.get("application_visible_h2d_bytes") for row in group)
        d2h_values = _numbers(row.get("application_visible_d2h_bytes") for row in group)
        total_values = _numbers(
            row.get("application_visible_transfer_bytes") for row in group
        )
        completed_count = sum(
            1 for row in group if str(row.get("status") or "") == "completed"
        )
        validation_passed_count = sum(
            1 for row in group if str(row.get("validation_status") or "") == "passed"
        )
        exact_integer_match_count = sum(
            1 for row in group if row.get("exact_integer_match") is True
        )
        hardware_execution_count = sum(
            1 for row in group if row.get("hardware_kernel_executed") is True
        )
        allocation_count = sum(
            1 for row in group if row.get("hardware_allocation_verified") is True
        )
        no_simulator_count = sum(
            1 for row in group if row.get("simulator_kernel_executed") is False
        )
        no_cpu_fallback_count = sum(
            1 for row in group if row.get("cpu_fallback_used") is False
        )
        passed = all(
            count == len(group)
            for count in (
                completed_count,
                validation_passed_count,
                exact_integer_match_count,
                hardware_execution_count,
                allocation_count,
                no_simulator_count,
                no_cpu_fallback_count,
            )
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "suite_id": first.get("suite_id"),
                "case_id": first.get("case_id"),
                "route_id": first.get("route_id"),
                "backend_id": first.get("backend_id"),
                "hardware_profile_version": first.get("hardware_profile_version"),
                "execution_class": first.get("execution_class"),
                "kernel_strategy": first.get("kernel_strategy"),
                "repeat_count": len(group),
                "completed_count": completed_count,
                "validation_passed_count": validation_passed_count,
                "exact_integer_match_count": exact_integer_match_count,
                "hardware_execution_count": hardware_execution_count,
                "hardware_allocation_verified_count": allocation_count,
                "no_simulator_fallback_count": no_simulator_count,
                "no_cpu_fallback_count": no_cpu_fallback_count,
                "requested_dpu_count": _first_present(group, "requested_dpu_count"),
                "allocated_dpu_count": _first_present(group, "allocated_dpu_count"),
                "tasklets_per_dpu": _first_present(group, "tasklets_per_dpu"),
                "application_visible_h2d_bytes_median": statistics.median(h2d_values)
                if h2d_values
                else None,
                "application_visible_d2h_bytes_median": statistics.median(d2h_values)
                if d2h_values
                else None,
                "application_visible_transfer_bytes_median": statistics.median(
                    total_values
                )
                if total_values
                else None,
                "timing_scope": _first_present(group, "timing_scope"),
                "hardware_speedup_applicable": any(
                    bool(row.get("hardware_speedup_applicable", False)) for row in group
                ),
                "functionality_evidence_status": "passed" if passed else "failed",
            }
        )
    return rows


def write_plots(
    out_dir: Path,
    stats_rows: list[JsonDict],
    cpu_gpu_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    same_plan_rows: list[JsonDict],
    planner_rows: list[JsonDict],
    slicing_tradeoff_rows: list[JsonDict] | None = None,
    planner_component_rows: list[JsonDict] | None = None,
    hardware_mvp_rows: list[JsonDict] | None = None,
    hardware_generic_mvp_rows: list[JsonDict] | None = None,
    physical_quantization_rows: list[JsonDict] | None = None,
    physical_taskgraph_rows: list[JsonDict] | None = None,
) -> JsonDict:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    if slicing_tradeoff_rows is None:
        slicing_tradeoff_rows = slicing_tradeoff(stats_rows)
        _write_csv(
            out_dir / "cpu_tn_slicing_tradeoff.csv",
            slicing_tradeoff_rows,
            SLICING_TRADEOFF_FIELDS,
        )
    if planner_component_rows is None:
        planner_component_rows = planner_component_diagnostics(planner_rows)
    if hardware_mvp_rows is None:
        hardware_mvp_rows = []
    if hardware_generic_mvp_rows is None:
        hardware_generic_mvp_rows = []
    if physical_quantization_rows is None:
        physical_quantization_rows = []
    if physical_taskgraph_rows is None:
        physical_taskgraph_rows = []
    _write_csv(
        out_dir / "upmem_physical_quantization_attribution.csv",
        physical_quantization_rows,
        UPMEM_PHYSICAL_QUANTIZATION_FIELDS,
    )
    _write_csv(
        out_dir / "upmem_physical_taskgraph_breakdown.csv",
        physical_taskgraph_rows,
        UPMEM_PHYSICAL_TASKGRAPH_FIELDS,
    )
    _write_csv(
        out_dir / "planner_component_diagnostics.csv",
        planner_component_rows,
        PLANNER_COMPONENT_DIAGNOSTICS_FIELDS,
    )
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "plots": [
                {
                    "plot": spec.filename,
                    "title": spec.title,
                    "source_csv": spec.source_csv,
                    "source_fields": list(spec.source_fields),
                    "claim_basis": spec.claim_basis,
                    "caption": spec.caption,
                    "status": "failed",
                    "reason": f"matplotlib_unavailable: {exc}",
                }
                for spec in _plot_specs(
                    stats_rows,
                    cpu_gpu_rows,
                    quantization_rows,
                    same_plan_rows,
                    planner_rows,
                    slicing_tradeoff_rows,
                    planner_component_rows,
                    hardware_mvp_rows,
                    hardware_generic_mvp_rows,
                    physical_quantization_rows,
                    physical_taskgraph_rows,
                )
            ],
            "generated_valid": [],
            "todo_figures": [],
            "failed_figures": [
                spec.filename
                for spec in _plot_specs(
                    stats_rows,
                    cpu_gpu_rows,
                    quantization_rows,
                    same_plan_rows,
                    planner_rows,
                    slicing_tradeoff_rows,
                    planner_component_rows,
                    hardware_mvp_rows,
                    hardware_generic_mvp_rows,
                    physical_quantization_rows,
                    physical_taskgraph_rows,
                )
            ],
        }
    entries: list[JsonDict] = []
    for spec in _plot_specs(
        stats_rows,
        cpu_gpu_rows,
        quantization_rows,
        same_plan_rows,
        planner_rows,
        slicing_tradeoff_rows,
        planner_component_rows,
        hardware_mvp_rows,
        hardware_generic_mvp_rows,
        physical_quantization_rows,
        physical_taskgraph_rows,
    ):
        path = plots_dir / spec.filename
        outcome = _render_plot_spec(plt, path, spec)
        entry = {
            "plot": spec.filename,
            "title": spec.title,
            "source_csv": spec.source_csv,
            "source_fields": list(spec.source_fields),
            "claim_basis": spec.claim_basis,
            "caption": spec.caption,
            "status": outcome.status,
            "reason": outcome.reason,
        }
        if outcome.size_bytes is not None:
            entry["size_bytes"] = outcome.size_bytes
        entries.append(entry)
    generated_valid = [
        entry["plot"] for entry in entries if entry["status"] == "generated_valid"
    ]
    todo_entries = [
        entry for entry in entries if entry["status"].startswith("generated_todo_")
    ]
    failed = [entry["plot"] for entry in entries if entry["status"] == "failed"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "plots": entries,
        "generated_valid": generated_valid,
        "todo_figures": todo_entries,
        "failed_figures": failed,
    }


def _plot_specs(
    stats_rows: list[JsonDict],
    cpu_gpu_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    same_plan_rows: list[JsonDict],
    planner_rows: list[JsonDict],
    slicing_tradeoff_rows: list[JsonDict] | None = None,
    planner_component_rows: list[JsonDict] | None = None,
    hardware_mvp_rows: list[JsonDict] | None = None,
    hardware_generic_mvp_rows: list[JsonDict] | None = None,
    physical_quantization_rows: list[JsonDict] | None = None,
    physical_taskgraph_rows: list[JsonDict] | None = None,
) -> list[PlotSpec]:
    def render(
        function: Callable[..., str | None], *args: Any, **kwargs: Any
    ) -> Callable[[Any, Path], str | None]:
        return lambda plt, path: function(plt, path, *args, **kwargs)

    specs = [
        PlotSpec(
            "cpu_gpu_runtime_by_qubits.png",
            "Measured QuEST CPU/GPU compute time by qubits",
            "cpu_gpu_performance_summary.csv",
            (
                "benchmark_n_qubits",
                "cpu_simulation_compute_time_s_median",
                "gpu_simulation_compute_time_s_median",
            ),
            "Verified QuEST CPU/GPU performance-tier compute measurements.",
            "Measured QuEST CPU and verified QuEST GPU compute time by circuit size.",
            "Qubits",
            "Measured compute time (s, log)",
            render(_plot_cpu_gpu_runtime, cpu_gpu_rows),
            variance_fields=(
                "cpu_simulation_compute_time_s_median",
                "gpu_simulation_compute_time_s_median",
            ),
        ),
        PlotSpec(
            "cpu_gpu_speedup_by_qubits.png",
            "Measured QuEST CPU/GPU compute ratio by qubits",
            "cpu_gpu_performance_summary.csv",
            ("benchmark_n_qubits", "compute_speedup_cpu_over_gpu_median"),
            "Paired verified QuEST compute-time measurements; ratio is CPU divided by GPU.",
            "Measured CPU/GPU compute ratio; values above 1 mean GPU compute was faster.",
            "Qubits",
            "CPU/GPU compute ratio",
            render(_plot_cpu_gpu_speedup, cpu_gpu_rows),
            variance_fields=("compute_speedup_cpu_over_gpu_median",),
        ),
        PlotSpec(
            "cpu_gpu_energy_efficiency_by_qubits.png",
            "Measured CPU/GPU energy efficiency by qubits",
            "per_case_route_stats.csv",
            (
                "benchmark_n_qubits",
                "energy_joules",
                "energy_measurement_status",
                "simulation_compute_time_s_median",
            ),
            "Requires measured energy for both paired CPU and GPU executions; unavailable energy is not inferred.",
            "TODO: energy metadata is unavailable or not a validated paired measurement.",
            "Qubits",
            "Energy efficiency (measured joules per compute work unit)",
            not_implemented_reason="energy measurements are unavailable in the research evidence contract",
        ),
        PlotSpec(
            "cpu_tn_runtime_by_qubits.png",
            "Measured Quimb unsliced/sliced compute time by qubits",
            "per_case_route_stats.csv",
            ("benchmark_n_qubits", "route_id", "simulation_compute_time_s_median"),
            "Measured external Quimb tensor-network contraction compute time, with unsliced and sliced routes kept separate.",
            "Measured Quimb unsliced and sliced tensor-network compute time by circuit size.",
            "Qubits",
            "Measured contraction compute time (s, log)",
            render(_plot_cpu_tn_runtime, stats_rows),
            variance_fields=("simulation_compute_time_s_median",),
        ),
        PlotSpec(
            "full_state_vs_tn_runtime_by_qubits.png",
            "Cross-algorithm/backend measured compute time",
            "per_case_route_stats.csv",
            ("benchmark_n_qubits", "route_id", "simulation_compute_time_s_median"),
            "Cross-algorithm/backend comparison of measured QuEST full-state and Quimb tensor-network compute time; not a same-plan speedup.",
            "Cross-algorithm/backend measured compute time on the same shallow circuits.",
            "Qubits",
            "Measured compute time (s, log)",
            render(_plot_full_state_vs_tn_runtime, stats_rows),
            variance_fields=("simulation_compute_time_s_median",),
        ),
        PlotSpec(
            "tn_planning_vs_contraction.png",
            "Measured Quimb planning versus contraction compute time",
            "per_case_route_stats.csv",
            (
                "benchmark_n_qubits",
                "planning_time_s_median",
                "simulation_compute_time_s_median",
            ),
            "Measured Quimb planning and contraction timings are reported as separate software phases.",
            "Measured Quimb planning and contraction compute time; phases are not combined.",
            "Qubits",
            "Measured time (s, log)",
            render(_plot_tn_planning_vs_contraction, stats_rows),
            variance_fields=(
                "planning_time_s_median",
                "simulation_compute_time_s_median",
            ),
        ),
        PlotSpec(
            "tn_path_flops_by_family_size.png",
            "Planner-estimated contraction FLOPs",
            "per_case_route_stats.csv",
            ("benchmark_n_qubits", "tn_estimated_flops", "route_id"),
            "Planner-reported contraction-path FLOP estimates, not measured processor instructions.",
            "Planner-estimated contraction FLOPs by circuit family and size.",
            "Qubits",
            "Planner-estimated FLOPs (log)",
            render(
                _plot_tn_path_metric,
                stats_rows,
                metric="tn_estimated_flops",
                ylabel="Planner-estimated FLOPs",
                title="Planner-estimated contraction FLOPs",
            ),
            variance_fields=("tn_estimated_flops",),
        ),
        PlotSpec(
            "tn_path_peak_memory_by_family_size.png",
            "Planner-estimated largest intermediate tensor",
            "per_case_route_stats.csv",
            ("benchmark_n_qubits", "tn_max_intermediate_bytes", "route_id"),
            "Planner-reported largest intermediate tensor size, not measured resident memory.",
            "Planner-estimated largest intermediate tensor bytes by circuit family and size.",
            "Qubits",
            "Largest intermediate bytes (log)",
            render(
                _plot_tn_path_metric,
                stats_rows,
                metric="tn_max_intermediate_bytes",
                ylabel="Largest intermediate bytes",
                title="Planner-estimated largest intermediate tensor",
            ),
            variance_fields=("tn_max_intermediate_bytes",),
        ),
        PlotSpec(
            "cpu_tn_slicing_flop_ratio.png",
            "Planner-estimated slicing arithmetic ratio",
            "per_case_route_stats.csv",
            ("benchmark_n_qubits", "slicing_flop_ratio", "slicing_flop_change_kind"),
            "Quimb/cotengra reported sliced-to-unsliced plan FLOPs for the external sliced route; optimizer choices can change this ratio.",
            "Slicing FLOP ratio = sliced cotengra plan reported FLOPs / unsliced cotengra plan reported FLOPs.",
            "Case",
            "Sliced plan FLOPs / unsliced plan FLOPs",
            render(_plot_slicing_ratio, stats_rows),
            variance_fields=("slicing_flop_ratio",),
        ),
        PlotSpec(
            "cpu_tn_slicing_tradeoff.png",
            "Quimb slicing tradeoff",
            "cpu_tn_slicing_tradeoff.csv",
            (
                "benchmark_n_qubits",
                "runtime_ratio_sliced_over_unsliced",
                "slicing_flop_ratio",
                "largest_intermediate_ratio_sliced_over_unsliced",
            ),
            "Matched external Quimb sliced/unsliced route statistics with compatible timing and output scopes; ratios are descriptive slicing trade-offs, not speedup claims.",
            "Matched Quimb slicing trade-off ratios: runtime, planner-estimated FLOPs, and largest intermediate size (sliced / unsliced).",
            "Qubits",
            "Ratio (sliced / unsliced)",
            render(_plot_slicing_tradeoff, slicing_tradeoff_rows or []),
            variance_fields=(
                "runtime_ratio_sliced_over_unsliced",
                "slicing_flop_ratio",
                "largest_intermediate_ratio_sliced_over_unsliced",
            ),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "upmem_supported_boundary.png",
            "UPMEM SDK simulator support boundary",
            "per_case_route_stats.csv",
            (
                "benchmark_n_qubits",
                "status",
                "unsupported_count",
                "upmem_execution_mode",
            ),
            "Strict generic-only UPMEM SDK simulator support/unsupported records; no hardware claim.",
            "Supported versus unsupported strict generic-only UPMEM SDK simulator rows.",
            "Qubits",
            "SDK simulator support (0/1)",
            render(_plot_upmem_boundary, stats_rows),
            variance_fields=("unsupported_count",),
        ),
        PlotSpec(
            "upmem_accuracy_error.png",
            "UPMEM SDK simulator accuracy error",
            "per_case_route_stats.csv",
            (
                "benchmark_n_qubits",
                "max_abs_error",
                "validation_method",
                "upmem_execution_mode",
            ),
            "Validation error from strict generic UPMEM SDK simulator execution against its recorded reference.",
            "Strict generic UPMEM SDK simulator maximum absolute error where validation data exists.",
            "Case",
            "Maximum absolute error (log)",
            render(_plot_upmem_accuracy, stats_rows),
            variance_fields=("max_abs_error",),
        ),
        PlotSpec(
            "upmem_hardware_mvp_validation.png",
            "Physical UPMEM single-DPU MVP validation",
            "upmem_hardware_mvp_summary.csv",
            (
                "case_id",
                "repeat_count",
                "validation_passed_count",
                "exact_integer_match_count",
            ),
            "Physical one-DPU/one-tasklet int8 x int8 -> int32 functionality evidence. This figure does not report timing, speedup, energy, or scaling.",
            "Physical UPMEM functionality MVP: exact CPU-reference validation counts by fixed dense case.",
            "Fixed dense case",
            "Validated physical executions",
            render(_plot_upmem_hardware_mvp_validation, hardware_mvp_rows or []),
            variance_fields=("validation_passed_count",),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "upmem_hardware_generic_mvp_validation.png",
            "Physical UPMEM generic TaskGraph MVP validation",
            "upmem_hardware_generic_mvp_summary.csv",
            (
                "case_id",
                "repeat_count",
                "validation_passed_count",
                "exact_integer_match_count",
            ),
            "Physical one-DPU/one-tasklet synthetic real generic TaskGraph functionality evidence. This figure does not report timing, speedup, energy, scaling, or general quantum-circuit execution.",
            "Physical generic TaskGraph MVP: exact CPU-reference validation counts for the fixed synthetic contraction.",
            "Synthetic generic TaskGraph case",
            "Validated physical executions",
            render(
                _plot_upmem_hardware_mvp_validation,
                hardware_generic_mvp_rows or [],
                title="Physical UPMEM generic TaskGraph MVP validation (functionality only)",
            ),
            variance_fields=("validation_passed_count",),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "upmem_quantization_attribution.png",
            "UPMEM SDK simulator quantization attribution",
            "upmem_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "route_runtime_ratio_none_over_quantized",
                "transfer_ratio_none_over_quantized",
            ),
            "Same-route float32/int8 attribution from SDK simulator measurements; not hardware speedup.",
            "Same-route float32 versus int8 attribution for strict generic UPMEM SDK simulator execution.",
            "Case",
            "Recorded ratio",
            render(_plot_upmem_quantization_attribution, quantization_rows),
            variance_fields=(
                "route_runtime_ratio_none_over_quantized",
                "transfer_ratio_none_over_quantized",
            ),
        ),
        PlotSpec(
            "quantization_runtime_by_executor.png",
            "UPMEM SDK simulator software-recorded host/control residual ratio",
            "upmem_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "route_runtime_ratio_none_over_quantized",
                "unquantized_host_residual_time_s",
                "quantized_host_residual_time_s",
            ),
            "Same-route SDK simulator software-recorded host/control residual time; attribution is not a hardware speedup claim.",
            "Same-plan SDK simulator float32/int8 software-recorded host/control residual-time ratio.",
            "Case",
            "Float32/int8 host/control residual ratio",
            render(
                _plot_quantization_metric,
                quantization_rows,
                metric="route_runtime_ratio_none_over_quantized",
                ylabel="Float32 / int8 software-recorded host/control residual time",
                title="UPMEM SDK simulator software-recorded host/control residual ratio",
            ),
            variance_fields=("route_runtime_ratio_none_over_quantized",),
        ),
        PlotSpec(
            "quantization_transfer_bytes.png",
            "UPMEM SDK simulator software-recorded directional transfer bytes",
            "upmem_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "unquantized_h2d_bytes",
                "quantized_h2d_bytes",
                "unquantized_d2h_bytes",
                "quantized_d2h_bytes",
                "transfer_ratio_none_over_quantized",
            ),
            "Application-visible SDK transfer bytes from matched routes. They are not physical bus/DIMM traffic and exclude unobservable SDK overhead.",
            "Absolute application-visible H2D/D2H bytes plus the same-plan float32/int8 total-byte ratio.",
            "Case",
            "Software-recorded application-visible bytes",
            render(_plot_quantization_transfer_bytes, quantization_rows),
            variance_fields=(
                "unquantized_h2d_bytes",
                "quantized_h2d_bytes",
                "unquantized_d2h_bytes",
                "quantized_d2h_bytes",
                "transfer_ratio_none_over_quantized",
            ),
        ),
        PlotSpec(
            "quantization_error_by_family_size.png",
            "UPMEM SDK simulator int8 error",
            "upmem_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "quantized_max_abs_error_vs_full_precision",
                "quantized_execution_max_abs_error",
            ),
            "Quantized SDK simulator error against the recorded full-precision reference.",
            "UPMEM SDK simulator int8 maximum absolute error against the full-precision reference.",
            "Case",
            "Maximum absolute error (log)",
            render(
                _plot_quantization_metric,
                quantization_rows,
                metric="quantized_max_abs_error_vs_full_precision",
                ylabel="Maximum absolute error",
                title="UPMEM SDK simulator int8 error versus full precision",
                log_scale=True,
            ),
            variance_fields=("quantized_max_abs_error_vs_full_precision",),
        ),
        PlotSpec(
            "quantization_probability_error_by_family_size.png",
            "UPMEM SDK simulator probability error",
            "upmem_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "quantized_probability_max_abs_error",
                "quantized_probability_l1_error",
            ),
            "Probability-error fields from matched quantized UPMEM SDK-simulator validation records; no amplitude error is relabeled as probability error.",
            "Matched UPMEM SDK-simulator quantized probability error against the recorded validation reference.",
            "Case",
            "Probability error",
            render(_plot_quantization_probability_error, quantization_rows),
            variance_fields=(
                "quantized_probability_max_abs_error",
                "quantized_probability_l1_error",
            ),
        ),
        PlotSpec(
            "same_plan_cpu_upmem_runtime.png",
            "Same-plan CPU versus UPMEM SDK simulator compute time",
            "same_plan_execution.csv",
            (
                "benchmark_n_qubits",
                "contraction_plan_hash",
                "cpu_time_s",
                "upmem_simulator_time_s",
            ),
            "CPU replay and UPMEM SDK simulator rows share an identical contraction-plan hash; timing is not hardware speedup.",
            "Same-plan CPU and UPMEM SDK simulator execution timing.",
            "Qubits",
            "Measured/software-recorded compute time (s, log)",
            render(_plot_same_plan_runtime, same_plan_rows),
            variance_fields=("cpu_time_s", "upmem_simulator_time_s"),
        ),
        PlotSpec(
            "upmem_physical_quantization_runtime.png",
            "Physical UPMEM TaskGraph float32/int8 warm runtime",
            "upmem_physical_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "contraction_plan_hash",
                "float32_warm_runtime_s",
                "int8_warm_runtime_s",
                "warm_runtime_ratio_float32_over_int8",
            ),
            "Matched same-plan physical UPMEM TaskGraph rows with explicit measured warm hardware timing. Bring-up-only wall time is excluded.",
            "Physical UPMEM float32/int8 warm runtime attribution; no speedup claim.",
            "Case",
            "Measured warm route time (s)",
            render(
                _plot_physical_quantization_runtime, physical_quantization_rows or []
            ),
            variance_fields=("float32_warm_runtime_s", "int8_warm_runtime_s"),
        ),
        PlotSpec(
            "upmem_physical_quantization_transfer.png",
            "Physical UPMEM TaskGraph float32/int8 transfer bytes",
            "upmem_physical_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "float32_transfer_bytes",
                "int8_transfer_bytes",
                "transfer_ratio_float32_over_int8",
            ),
            "Matched same-plan physical UPMEM application-visible transfer accounting; byte counts are recorded transfers, not a bus-bandwidth or speedup claim.",
            "Physical UPMEM float32/int8 application-visible transfer attribution.",
            "Case",
            "Recorded transfer bytes",
            render(
                _plot_physical_quantization_transfer, physical_quantization_rows or []
            ),
            variance_fields=("float32_transfer_bytes", "int8_transfer_bytes"),
        ),
        PlotSpec(
            "upmem_physical_quantization_error.png",
            "Physical UPMEM TaskGraph float32/int8 error",
            "upmem_physical_quantization_attribution.csv",
            (
                "benchmark_n_qubits",
                "float32_max_abs_error",
                "int8_max_abs_error",
                "quantization_error_int8_vs_float32",
            ),
            "Matched same-plan physical UPMEM validation error fields are reported as recorded; absent error metadata is not inferred.",
            "Physical UPMEM float32/int8 maximum absolute error attribution.",
            "Case",
            "Recorded maximum absolute error",
            render(_plot_physical_quantization_error, physical_quantization_rows or []),
            variance_fields=(
                "float32_max_abs_error",
                "int8_max_abs_error",
                "quantization_error_int8_vs_float32",
            ),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "upmem_physical_taskgraph_validation.png",
            "Physical UPMEM TaskGraph validation",
            "upmem_physical_taskgraph_breakdown.csv",
            (
                "case_id",
                "validation_passed",
                "exact_integer_match",
                "task_count",
                "validated_task_count",
            ),
            "Physical UPMEM TaskGraph validation and execution evidence, including bring-up rows; no performance claim.",
            "Physical UPMEM TaskGraph validation status and task coverage.",
            "Case/repeat",
            "Validation result (1=passed)",
            render(_plot_physical_taskgraph_validation, physical_taskgraph_rows or []),
            variance_fields=("validation_passed", "task_count", "validated_task_count"),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "upmem_physical_taskgraph_timing_breakdown.png",
            "Physical UPMEM TaskGraph timing breakdown",
            "upmem_physical_taskgraph_breakdown.csv",
            (
                "case_id",
                "timing_class",
                "allocation_time_s",
                "binary_load_time_s",
                "h2d_time_s",
                "d2h_time_s",
                "total_quantization_time_s",
                "total_dequantization_time_s",
                "total_bridge_time_s",
                "total_build_time_s",
                "warm_runtime_s",
                "validation_time_s",
                "output_materialization_time_s",
            ),
            "Physical UPMEM TaskGraph timing components are shown only when recorded. Bring-up-only route time is labeled and excluded from measured warm timing.",
            "Physical UPMEM TaskGraph timing breakdown; bring-up and measured warm timing remain distinct.",
            "Case/repeat",
            "Recorded time (s)",
            render(_plot_physical_taskgraph_timing, physical_taskgraph_rows or []),
            variance_fields=(
                "allocation_time_s",
                "binary_load_time_s",
                "h2d_time_s",
                "d2h_time_s",
                "total_quantization_time_s",
                "total_dequantization_time_s",
                "total_bridge_time_s",
                "total_build_time_s",
                "warm_runtime_s",
                "validation_time_s",
                "output_materialization_time_s",
            ),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "planner_flops_vs_upmem_pressure.png",
            "Planner-estimated FLOPs versus normalized modeled PIM objective",
            "planner_comparison.csv",
            (
                "planner_id",
                "pim_estimated_flops",
                "pim_objective_score",
                "pim_feasible",
            ),
            "Modeled generic-single-DPU planning objective; no executor timing or hardware speedup claim.",
            "Planner-estimated FLOPs versus normalized modeled PIM objective for feasible candidates.",
            "Planner-estimated FLOPs",
            "Normalized modeled PIM objective",
            render(_plot_planner_pressure, planner_rows),
            variance_fields=("pim_estimated_flops", "pim_objective_score"),
        ),
        PlotSpec(
            "planner_component_scores.png",
            "Modeled PIM planner objective components",
            "planner_comparison.csv",
            (
                "planner_id",
                "pim_estimated_flops",
                "pim_estimated_host_dpu_bytes",
                "pim_estimated_mram_to_wram_bytes",
                "pim_estimated_tile_count",
            ),
            "Normalized modeled PIM planning components; scenario weights are not measured hardware constants.",
            "Modeled PIM objective components for feasible planner candidates.",
            "Planner candidate",
            "Normalized component value",
            render(_plot_planner_components, planner_rows),
            variance_fields=(
                "pim_estimated_flops",
                "pim_estimated_host_dpu_bytes",
                "pim_estimated_mram_to_wram_bytes",
                "pim_estimated_tile_count",
            ),
        ),
        PlotSpec(
            "planner_selection.png",
            "Modeled PIM planner selection",
            "planner_comparison.csv",
            ("planner_id", "pim_objective_rank", "pim_selected"),
            "Selection is by the recorded modeled PIM objective within one profile, not execution performance.",
            "Selected feasible planner candidate per modeled PIM objective profile.",
            "Case",
            "Modeled objective rank",
            render(_plot_planner_selection, planner_rows),
            variance_fields=("pim_objective_rank",),
        ),
        PlotSpec(
            "planner_pareto_frontier.png",
            "Planner-estimated Pareto frontier",
            "planner_comparison.csv",
            (
                "planner_id",
                "pim_estimated_flops",
                "pim_peak_intermediate_bytes",
                "pim_pareto_dominated",
            ),
            "Pareto status compares modeled planner components only; it is not a runtime ranking.",
            "Feasible planner candidates colored by modeled Pareto status.",
            "Planner-estimated FLOPs",
            "Planner-estimated largest intermediate bytes",
            render(_plot_planner_pareto, planner_rows),
            variance_fields=("pim_estimated_flops", "pim_peak_intermediate_bytes"),
        ),
        PlotSpec(
            "planner_sensitivity.png",
            "Modeled PIM planner sensitivity",
            "planner_comparison.csv",
            ("planner_id", "pim_weight_profile", "pim_objective_score", "pim_selected"),
            "Scenario weight profiles are literature-informed sensitivity cases, not calibrated hardware constants.",
            "Selected planner candidates across modeled PIM weight profiles.",
            "Weight profile",
            "Normalized modeled PIM objective",
            render(_plot_planner_sensitivity, planner_rows),
            variance_fields=("pim_objective_score",),
        ),
        PlotSpec(
            "planner_component_diagnostics.png",
            "Modeled PIM planner component diagnostics",
            "planner_component_diagnostics.csv",
            (
                "planner_id",
                "pim_objective_version",
                "pim_numeric_component_invocations",
                "pim_numeric_recombination_flops",
                "pim_mram_payload_pressure_ratio",
                "pim_wram_known_pressure_ratio",
            ),
            "Versioned modeled planner component diagnostics; these are structural cost-model values, not measured hardware counters.",
            "Versioned planner component diagnostics where v2 records provide numeric execution decomposition fields.",
            "Planner candidate",
            "Modeled component value",
            render(_plot_planner_component_diagnostics, planner_component_rows or []),
            variance_fields=(
                "pim_numeric_component_invocations",
                "pim_numeric_recombination_flops",
                "pim_mram_payload_pressure_ratio",
                "pim_wram_known_pressure_ratio",
            ),
            allow_zero_variance=True,
        ),
        PlotSpec(
            "internal_parallelism_metadata_by_qubits.png",
            "Internal diagnostic frontier metadata",
            "per_case_route_stats.csv",
            (
                "benchmark_n_qubits",
                "parallelism_mode",
                "parallelism_evidence_type",
                "frontier_worker_count",
                "frontier_wave_count",
                "max_frontier_width",
                "frontier_executed_parallel_task_count",
            ),
            "Executed internal TaskGraph frontier metadata; diagnostic only and not a parallel speedup claim.",
            "Diagnostic internal TaskGraph frontier metadata, not serious baseline performance.",
            "Case",
            "Maximum frontier width",
            render(_plot_internal_parallelism, stats_rows),
            variance_fields=("max_frontier_width",),
        ),
    ]
    source_rows = {
        "per_case_route_stats.csv": stats_rows,
        "cpu_gpu_performance_summary.csv": cpu_gpu_rows,
        "upmem_quantization_attribution.csv": quantization_rows,
        "same_plan_execution.csv": same_plan_rows,
        "planner_comparison.csv": planner_rows,
        "cpu_tn_slicing_tradeoff.csv": slicing_tradeoff_rows or [],
        "planner_component_diagnostics.csv": planner_component_rows or [],
        "upmem_hardware_mvp_summary.csv": hardware_mvp_rows or [],
        "upmem_hardware_generic_mvp_summary.csv": hardware_generic_mvp_rows or [],
        "upmem_physical_quantization_attribution.csv": physical_quantization_rows or [],
        "upmem_physical_taskgraph_breakdown.csv": physical_taskgraph_rows or [],
    }
    return [replace(spec, data_rows=source_rows[spec.source_csv]) for spec in specs]


def _render_plot_spec(plt: Any, path: Path, spec: PlotSpec) -> PlotOutcome:
    if spec.not_implemented_reason:
        _save_todo_plot(plt, path, spec, spec.not_implemented_reason)
        return PlotOutcome(
            "generated_todo_not_implemented",
            spec.not_implemented_reason,
            path.stat().st_size,
        )
    field_values = [
        [
            number
            for number in (
                _float_or_none(row.get(field)) for row in (spec.data_rows or [])
            )
            if number is not None
        ]
        for field in spec.variance_fields
    ]
    values = [
        number for field_values_item in field_values for number in field_values_item
    ]
    if not values:
        reason = "source fields contain no numeric data: " + ", ".join(
            spec.source_fields
        )
        _save_todo_plot(plt, path, spec, reason)
        return PlotOutcome("generated_todo_missing_data", reason, path.stat().st_size)
    if not spec.allow_zero_variance and all(
        len(set(field_values_item)) <= 1
        for field_values_item in field_values
        if field_values_item
    ):
        reason = "source data has zero variance across the available records"
        _save_todo_plot(plt, path, spec, reason)
        return PlotOutcome("generated_todo_no_variance", reason, path.stat().st_size)
    try:
        reason = (
            spec.renderer(plt, path)
            if spec.renderer is not None
            else "renderer_not_configured"
        )
    except Exception as exc:  # Plot failures remain distinct from evidence TODOs.
        return PlotOutcome("failed", f"rendering_failed: {type(exc).__name__}: {exc}")
    if reason:
        _save_todo_plot(plt, path, spec, reason)
        return PlotOutcome("generated_todo_missing_data", reason, path.stat().st_size)
    if not path.exists():
        return PlotOutcome(
            "failed", "rendering_failed: renderer did not create the expected PNG"
        )
    return PlotOutcome("generated_valid", None, path.stat().st_size)


def _save_todo_plot(plt: Any, path: Path, spec: PlotSpec, reason: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.set_title(spec.title)
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel(spec.y_label)
    ax.text(
        0.5,
        0.5,
        f"TODO\n{reason}",
        ha="center",
        va="center",
        wrap=True,
        transform=ax.transAxes,
        color="#b45309",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    _save_plot(fig, path)


def benchmark_summary(
    manifest: JsonDict,
    records: list[JsonDict],
    stats_rows: list[JsonDict],
    speedup_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    planner_rows: list[JsonDict],
    full_state_tn_rows: list[JsonDict],
    unsupported_rows: list[JsonDict],
    validation_rows: list[JsonDict],
    capability_rows: list[JsonDict],
    plot_manifest: JsonDict,
    boundary: JsonDict,
    guard_issues: list[str],
    hardware_mvp_rows: list[JsonDict] | None = None,
    hardware_generic_mvp_rows: list[JsonDict] | None = None,
    physical_quantization_rows: list[JsonDict] | None = None,
    physical_taskgraph_rows: list[JsonDict] | None = None,
) -> str:
    hardware_mvp_rows = (
        upmem_hardware_mvp_summary(records)
        if hardware_mvp_rows is None
        else hardware_mvp_rows
    )
    hardware_generic_mvp_rows = (
        upmem_hardware_generic_mvp_summary(records)
        if hardware_generic_mvp_rows is None
        else hardware_generic_mvp_rows
    )
    physical_quantization_rows = (
        [] if physical_quantization_rows is None else physical_quantization_rows
    )
    physical_taskgraph_rows = (
        [] if physical_taskgraph_rows is None else physical_taskgraph_rows
    )
    lines = [
        "# Research Benchmark Summary",
        "",
        "This is a derived research pack generated from normalized benchmark records. Evidence inputs remain read-only.",
        "",
        "## Benchmark Matrix",
        "",
        "| Route | Role | Target | Records | Unsupported |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in capability_rows:
        lines.append(
            f"| {row['route_id']} | {row.get('benchmark_role') or ''} | {row.get('contraction_execution_target') or ''} | "
            f"{row['record_count']} | {row['unsupported_count']} |"
        )
    lines.extend(
        [
            "",
            "## Suite Paths",
            "",
        ]
    )
    for key, path in manifest["selected_suites"].items():
        lines.append(f"- `{key}`: `{path}`")
    lines.extend(
        [
            "",
            "## Exact Commands",
            "",
            "Run `make research-plan` to print the underlying commands.",
            "Run `BENCH_CPU_THREADS=<physical-core-count> make thesis-run` for the complete local benchmark matrix.",
            "",
            "## Hardware And Software Manifest",
            "",
            f"- Git commit: `{manifest.get('git_commit') or 'unknown'}`",
            f"- Dirty worktree: `{manifest.get('dirty_worktree')}`",
            f"- Host: `{manifest.get('hostname')}`",
            f"- Python: `{manifest.get('python_version')}`",
            f"- Packages: `{json.dumps(manifest.get('packages', {}), sort_keys=True)}`",
            "",
            "## Benchmark Group Status",
            "",
        ]
    )
    for command in manifest.get("commands", []):
        lines.append(
            f"- `{command.get('command')}`: returncode `{command.get('returncode')}`."
        )
        if command.get("skipped_group"):
            lines.append(
                f"  - skipped group `{command.get('skipped_group')}`: {command.get('blocker_reason')}"
            )
    lines.extend(
        [
            "",
            "## Repeats, Warmups, And Timing",
            "",
            "Statistics are computed from normalized records. Median and spread fields are reported per case/route.",
            "CPU/GPU performance-tier speedup uses `simulation_compute_time_s`; UPMEM route timing prefers `total_host_residual_time_s` when present and retains execution validation separately.",
            "Qubit-scaling tables and plots use `benchmark_n_qubits` / `actual_n_qubits`; suite caps and output caps are not used as circuit size.",
            "",
            "## Validation Methods",
            "",
        ]
    )
    for row in validation_rows:
        lines.append(
            f"- `{row['route_id']}` `{row['validation_status']}`: {row['record_count']} records"
        )
    if hardware_mvp_rows:
        lines.extend(
            [
                "",
                "## Physical UPMEM Hardware MVP",
                "",
                "This section reports physical hardware functionality evidence only. It deliberately excludes performance, speedup, energy, scaling, multi-DPU, generic TaskGraph, and quantum-circuit claims.",
                "",
                "| Case | Repeats | Exact matches | One-DPU allocations | Hardware kernel executions | H2D/D2H bytes | Status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in hardware_mvp_rows:
            lines.append(
                f"| {row['case_id']} | {row['repeat_count']} | {row['exact_integer_match_count']} | "
                f"{row['hardware_allocation_verified_count']} | {row['hardware_execution_count']} | "
                f"{row.get('application_visible_h2d_bytes_median')}/{row.get('application_visible_d2h_bytes_median')} | "
                f"{row['functionality_evidence_status']} |"
            )
    if hardware_generic_mvp_rows:
        lines.extend(
            [
                "",
                "## Physical UPMEM Generic TaskGraph MVP",
                "",
                "This section reports one synthetic real-valued generic TaskGraph contraction on physical hardware. It is exact functionality evidence only and deliberately excludes general quantum-circuit, performance, speedup, energy, scaling, multi-DPU, and scheduler claims.",
                "",
                "| Case | Repeats | Exact matches | One-DPU allocations | Hardware kernel executions | H2D/D2H bytes | Status |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in hardware_generic_mvp_rows:
            lines.append(
                f"| {row['case_id']} | {row['repeat_count']} | {row['exact_integer_match_count']} | "
                f"{row['hardware_allocation_verified_count']} | {row['hardware_execution_count']} | "
                f"{row.get('application_visible_h2d_bytes_median')}/{row.get('application_visible_d2h_bytes_median')} | "
                f"{row['functionality_evidence_status']} |"
            )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "### Completed Scientific Figures",
            "",
        ]
    )
    entries = plot_manifest.get("plots", [])
    for entry in entries:
        if entry.get("status") == "generated_valid":
            lines.append(f"- `{entry['plot']}`: {entry.get('caption') or ''}")
    lines.extend(["", "### TODO Figures", ""])
    for entry in entries:
        if str(entry.get("status", "")).startswith("generated_todo_"):
            lines.append(
                f"- `{entry['plot']}`: {entry['status']} ({entry.get('reason') or 'TODO'}). {entry.get('caption') or ''}"
            )
    lines.extend(["", "### Failed Figures", ""])
    for entry in entries:
        if entry.get("status") == "failed":
            lines.append(
                f"- `{entry['plot']}`: failed ({entry.get('reason') or 'unknown rendering failure'})."
            )
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Normalized records loaded: {len(records)}.",
            f"- Per-case route statistic rows: {len(stats_rows)}.",
            f"- Valid CPU/GPU paired speedup rows: {len(speedup_rows)}.",
            f"- Matched strict generic UPMEM float32/int8 attribution rows: {len(quantization_rows)}.",
            f"- Matched physical UPMEM TaskGraph float32/int8 attribution rows: {len(physical_quantization_rows)}.",
            f"- Physical UPMEM TaskGraph validation/timing breakdown rows: {len(physical_taskgraph_rows)}.",
            f"- Modeled contraction-path candidate rows: {len(planner_rows)}.",
            f"- Unsupported/skipped rows preserved: {len(unsupported_rows)}.",
            "",
            "## Observed Result Snapshot",
            "",
            *_observed_result_lines(
                speedup_rows,
                full_state_tn_rows,
                quantization_rows,
                planner_rows,
                unsupported_rows,
            ),
            "",
            "## Planner Interpretation",
            "",
            *_planner_interpretation_lines(planner_rows),
            "",
            "## Unsupported Cases",
            "",
        ]
    )
    if unsupported_rows:
        for row in unsupported_rows[:20]:
            lines.append(
                f"- `{row.get('case_id')}` / `{row.get('route_id')}`: {row.get('resource_skip_reason') or row.get('validation_status') or row.get('status')}"
            )
        if len(unsupported_rows) > 20:
            lines.append(
                f"- ... {len(unsupported_rows) - 20} more rows in `unsupported_cases.csv`."
            )
    else:
        lines.append("- None in loaded records.")
    lines.extend(
        [
            "",
            "## Claims Allowed",
            "",
            "- QuEST CPU/GPU full-state speedup only for matched CPU/GPU rows with the same timing scope.",
            "- Quimb unsliced vs sliced comparisons as CPU TN implementation evidence, with slicing metrics labeled as `slicing_flop_ratio`.",
            "- Strict generic-only UPMEM SDK simulator rows as bounded generic code-path and boundary evidence.",
            "- Same-route float32 versus int8 generic UPMEM ratios as SDK-simulator route attribution, not hardware speedup.",
            "- Matched physical UPMEM TaskGraph float32/int8 transfer and validation-error attribution only when the plan hash matches; runtime figures use explicit measured warm timing.",
            "",
            "## Claims Not Allowed",
            "",
            "- No hardware speedup claim from UPMEM SDK simulator timing.",
            "- No speedup claim from physical UPMEM TaskGraph attribution; bring-up-only timing is not measured warm timing.",
            "- No fake GPU rows without verified GPU execution.",
            "- No energy-efficiency claim without real measured energy metadata.",
            "- No speedup across incompatible route families such as Quimb versus internal TaskGraph diagnostics.",
            "",
            "## Artifact Boundary Checks",
            "",
            f"- Evidence boundary status: `{boundary['status']}`.",
            f"- Guard issues: {len(guard_issues)}.",
        ]
    )
    if hardware_mvp_rows:
        allowed_index = lines.index(
            "- Strict generic-only UPMEM SDK simulator rows as bounded generic code-path and boundary evidence."
        )
        lines.insert(
            allowed_index,
            "- Physical one-DPU/one-tasklet dense int8 x int8 -> int32 exact-validation rows as hardware functionality evidence only.",
        )
        not_allowed_index = lines.index(
            "- No hardware speedup claim from UPMEM SDK simulator timing."
        )
        lines.insert(
            not_allowed_index,
            "- No performance, speedup, energy, scaling, or generic-TN claim from the physical hardware MVP bring-up rows.",
        )
    for issue in guard_issues:
        lines.append(f"  - {issue}")
    lines.extend(
        [
            "",
            "## Next UPMEM Implementation Readiness",
            "",
            *_upmem_readiness_lines(records, unsupported_rows),
            "",
            "## Missing Evidence",
            "",
        ]
    )
    missing = _missing_evidence(records)
    for item in missing:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _planner_interpretation_lines(planner_rows: list[JsonDict]) -> list[str]:
    """Describe planner rows without turning modeled estimates into measurements."""
    if not planner_rows:
        return ["- No planner candidate evidence was loaded."]

    cases = {str(row.get("case_id")) for row in planner_rows if row.get("case_id")}
    selected = [row for row in planner_rows if row.get("pim_selected") is True]
    planners = sorted(
        {str(row.get("planner_id")) for row in selected if row.get("planner_id")}
    )
    profiles = sorted(
        {
            str(row.get("pim_weight_profile"))
            for row in selected
            if row.get("pim_weight_profile")
        }
    )
    paths = sorted(
        {
            str(row.get("contraction_path_structure_hash"))
            for row in selected
            if row.get("contraction_path_structure_hash")
        }
    )
    objective_versions = sorted(
        {
            str(row.get("pim_objective_version"))
            for row in planner_rows
            if row.get("pim_objective_version")
        }
    )

    component_fields = (
        "pim_estimated_flops",
        "pim_estimated_host_dpu_bytes",
        "pim_estimated_mram_to_wram_bytes",
        "pim_estimated_tile_count",
        "pim_estimated_dpu_local_work",
        "pim_estimated_sync_events",
        "pim_estimated_numerical_penalty",
    )
    sources = [
        field
        for field in component_fields
        if any(row.get(field) is not None for row in planner_rows)
    ]
    component_keys = sorted(
        {
            str(key)
            for row in planner_rows
            for field in ("pim_objective_components", "score_components")
            if isinstance(row.get(field), dict)
            for key in row[field]
        }
    )
    cost_sources = sources + [f"objective component: {key}" for key in component_keys]
    lines = [
        f"- Planner evidence is model-only hypothesis evidence: `{len(cases)}` modeled cases and `{len(planner_rows)}` candidate records.",
        "- It cannot claim hardware performance, hardware speedup, measured runtime, energy, or executed UPMEM behavior.",
    ]
    if objective_versions:
        lines.append(
            f"- Modeled objective/profile semantics: {', '.join(f'`{item}`' for item in objective_versions)}."
        )
    if selected:
        structural = [f"{len(selected)} selected rows"]
        if planners:
            structural.append("planners=" + ", ".join(f"`{item}`" for item in planners))
        if profiles:
            structural.append("profiles=" + ", ".join(f"`{item}`" for item in profiles))
        if paths:
            structural.append(
                f"{len(paths)} distinct contraction path structure hash(es)"
            )
        lines.append(
            "- Selected planner/profile/path structure: " + "; ".join(structural) + "."
        )
    else:
        lines.append("- No selected planner/profile/path structure was recorded.")
    if cost_sources:
        lines.append(
            "- Modeled cost-component sources: "
            + ", ".join(f"`{item}`" for item in cost_sources)
            + "."
        )
    else:
        lines.append("- No modeled cost-component source fields were populated.")
    return lines


def _observed_result_lines(
    speedup_rows: list[JsonDict],
    full_state_tn_rows: list[JsonDict],
    quantization_rows: list[JsonDict],
    planner_rows: list[JsonDict],
    unsupported_rows: list[JsonDict],
) -> list[str]:
    lines: list[str] = []
    gpu_ratios = _numbers(
        row.get("compute_speedup_cpu_over_gpu")
        for row in speedup_rows
        if _bool(row.get("performance_tier"))
    )
    if gpu_ratios:
        lines.append(
            "- Verified QuEST GPU compute ratio (CPU/GPU): "
            f"median `{statistics.median(gpu_ratios):.3g}x`, range `{min(gpu_ratios):.3g}x` to `{max(gpu_ratios):.3g}x`; "
            f"GPU was faster in `{sum(value > 1.0 for value in gpu_ratios)}/{len(gpu_ratios)}` matched repeats."
        )
    else:
        lines.append(
            "- No matched verified CPU/GPU performance repeats were available."
        )

    tn_ratios = _numbers(
        row.get("quimb_unsliced_time_over_quest_time") for row in full_state_tn_rows
    )
    slicing_ratios = _numbers(
        row.get("quimb_sliced_time_over_unsliced_time") for row in full_state_tn_rows
    )
    if tn_ratios:
        lines.append(
            "- Shallow exact CPU comparison (Quimb TN time / QuEST full-state time): "
            f"median `{statistics.median(tn_ratios):.3g}x` across `{len(tn_ratios)}` cases. "
            "This is an algorithm/backend runtime ratio, not same-plan speedup."
        )
    if slicing_ratios:
        lines.append(
            "- Executed Quimb slicing time / unsliced Quimb time: "
            f"median `{statistics.median(slicing_ratios):.3g}x`, range `{min(slicing_ratios):.3g}x` to `{max(slicing_ratios):.3g}x`; "
            "slice reconstruction used one worker."
        )

    quant_runtime = _numbers(
        row.get("route_runtime_ratio_none_over_quantized") for row in quantization_rows
    )
    quant_transfer = _numbers(
        row.get("transfer_ratio_none_over_quantized") for row in quantization_rows
    )
    quant_error = _numbers(
        row.get("quantized_max_abs_error_vs_full_precision")
        for row in quantization_rows
    )
    if quant_runtime:
        lines.append(
            "- Strict generic UPMEM SDK-simulator float32/int8 attribution: "
            f"median host-residual-time ratio `{statistics.median(quant_runtime):.3g}x`, "
            f"median transfer ratio `{statistics.median(quant_transfer):.3g}x`"
            + (
                f", maximum observed int8 absolute error `{max(quant_error):.3g}`."
                if quant_error
                else "."
            )
            + " These are simulator-route measurements, not hardware speedup."
        )

    planner_cases = {str(row.get("case_id")) for row in planner_rows}
    planner_ids = {str(row.get("planner_id")) for row in planner_rows}
    if planner_rows:
        lines.append(
            f"- Planner evidence covers `{len(planner_cases)}` cases and `{len(planner_ids)}` candidates "
            "with plan hashes, FLOP/peak-memory estimates, and modeled UPMEM pressure."
        )
    if unsupported_rows:
        reason_counts = Counter(
            str(row.get("resource_skip_reason") or "unknown")
            for row in unsupported_rows
        )
        lines.append(
            "- Explicit boundary rows: "
            + ", ".join(
                f"`{reason}` = {count}" for reason, count in reason_counts.most_common()
            )
            + "."
        )
    return lines


def validate_artifact_boundaries(root: Path) -> JsonDict:
    evidence_root = root / "runs" / "evidence"
    violations: list[str] = []
    if evidence_root.exists():
        for path in evidence_root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if path.name in FORBIDDEN_EVIDENCE_DERIVED_NAMES:
                violations.append(rel)
            if path.suffix.lower() in {".png", ".svg", ".pdf"} or "/plots/" in rel:
                violations.append(rel)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed" if violations else "passed",
        "violations": violations,
    }


def _claim_guard_issues(records: list[JsonDict]) -> list[str]:
    issues: list[str] = []
    for record in records:
        route = record.get("route_id")
        case = record.get("case_id")
        if record.get("contraction_execution_target") == "gpu" and not (
            record.get("gpu_backend_verified") is True
            and record.get("gpu_program_executed") is True
        ):
            issues.append(f"unverified gpu row: {case}/{route}")
        if record.get("upmem_execution_mode") == "sdk_simulator":
            if bool(record.get("hardware_speedup_applicable", False)):
                issues.append(
                    f"sdk simulator row marked hardware speedup applicable: {case}/{route}"
                )
            if record.get("hardware_speedup") not in {None, "", "not_applicable"}:
                issues.append(
                    f"sdk simulator row has hardware_speedup value: {case}/{route}"
                )
        if record.get("energy_joules") not in {None, ""} and str(
            record.get("energy_measurement_status") or ""
        ) not in {"measured", "available"}:
            issues.append(f"energy value without measured status: {case}/{route}")
        issues.extend(_transfer_accounting_issues(record, case=case, route=route))
        if record.get("contraction_execution_target") == "upmem":
            if _is_physical_hardware_mvp_record(record):
                issues.extend(_hardware_mvp_issues(record))
            elif _is_physical_upmem_taskgraph_record(record):
                if bool(record.get("hardware_speedup_applicable", False)):
                    issues.append(
                        f"physical TaskGraph row marked hardware speedup applicable: {case}/{route}"
                    )
                if bool(record.get("cpu_fallback_used", False)):
                    issues.append(
                        f"physical TaskGraph row used CPU fallback: {case}/{route}"
                    )
                if bool(record.get("simulator_kernel_executed", False)):
                    issues.append(
                        f"physical TaskGraph row executed simulator kernel: {case}/{route}"
                    )
            elif str(record.get("upmem_execution_mode") or "") == "sdk_simulator":
                issues.extend(_strict_generic_upmem_issues(record))
            else:
                issues.append(
                    f"UPMEM research row has unknown evidence class: {case}/{route} "
                    f"mode={record.get('upmem_execution_mode') or 'missing'}"
                )
    return issues


def _transfer_accounting_issues(
    record: JsonDict, *, case: Any, route: Any
) -> list[str]:
    """Validate directional byte totals when the producing runtime exposes them.

    Older evidence may have only ``actual_transfer_bytes``. That remains
    readable, but it cannot satisfy or fail a directional invariant that was
    not recorded at the time.
    """
    h2d = _float_or_none(record.get("actual_h2d_bytes"))
    d2h = _float_or_none(record.get("actual_d2h_bytes"))
    total = _float_or_none(record.get("actual_transfer_bytes"))
    issues: list[str] = []
    if h2d is not None or d2h is not None:
        if h2d is None or d2h is None or total is None:
            issues.append(f"incomplete directional transfer accounting: {case}/{route}")
        elif not math.isclose(total, h2d + d2h, rel_tol=0.0, abs_tol=0.0):
            issues.append(f"transfer-byte invariant failed: {case}/{route}")
    declared = record.get("actual_transfer_bytes_invariant")
    if declared not in {None, "", "passed"}:
        issues.append(f"transfer-byte invariant not passed: {case}/{route}")
    return issues


def _upmem_readiness_lines(
    records: list[JsonDict], unsupported_rows: list[JsonDict]
) -> list[str]:
    upmem_records = [
        row
        for row in records
        if row.get("contraction_execution_target") == "upmem"
        or row.get("upmem_execution_mode")
        in {"sdk_simulator", "sdk_hardware_single_dpu"}
    ]
    if not upmem_records:
        return [
            "- No UPMEM records were loaded in this pack.",
            "- Next target: run the selected strict generic-only UPMEM suite or physical MVP and regenerate this pack before making stronger UPMEM claims.",
        ]
    dense_hardware_records = [
        row for row in upmem_records if _is_hardware_mvp_record(row)
    ]
    generic_hardware_records = [
        row for row in upmem_records if _is_hardware_generic_mvp_record(row)
    ]
    simulator_records = [
        row
        for row in upmem_records
        if str(row.get("upmem_execution_mode") or "") == "sdk_simulator"
    ]
    unsupported = [
        row
        for row in unsupported_rows
        if str(row.get("route_id") or "").startswith("upmem")
        or row.get("route_id") == "upmem_tn_sdk_simulator_quantized"
    ]
    fallback_count = sum(
        1 for row in upmem_records if bool(row.get("cpu_fallback_used", False))
    )
    generic_records = [
        row for row in simulator_records if _is_strict_generic_upmem_record(row)
    ]
    reasons = Counter(_unsupported_reason(row) for row in unsupported)
    supported = [
        row
        for row in simulator_records
        if str(row.get("status") or "") == "completed" and not _is_unsupported(row)
    ]
    supported_qubits = [
        (qubits, row)
        for row in supported
        if (qubits := _family_and_qubits(row)[1]["benchmark_n_qubits"]) is not None
    ]
    highest_supported = max(
        supported_qubits,
        key=lambda item: (int(item[0]), str(item[1].get("case_id") or "")),
        default=None,
    )
    first_unsupported = min(
        (
            (qubits, row)
            for row in unsupported
            if (qubits := _family_and_qubits(row)[1]["benchmark_n_qubits"]) is not None
        ),
        key=lambda item: (int(item[0]), str(item[1].get("case_id") or "")),
        default=None,
    )
    tiling_records = [
        row for row in simulator_records if _tiling_status(row) is not None
    ]
    tiling_supported = [row for row in tiling_records if _tiling_status(row) is True]
    lines = [f"- UPMEM records loaded: {len(upmem_records)}."]
    if dense_hardware_records:
        completed = sum(
            1
            for row in dense_hardware_records
            if str(row.get("status") or "") == "completed"
        )
        exact = sum(
            1
            for row in dense_hardware_records
            if row.get("exact_integer_match") is True
        )
        physical_cases = sorted(
            {str(row.get("case_id") or "unknown") for row in dense_hardware_records}
        )
        lines.extend(
            [
                f"- Physical UPMEM dense single-DPU MVP rows: {len(dense_hardware_records)}; completed: {completed}; exact int8/int32 matches: {exact}.",
                f"- Physical MVP cases: {', '.join(f'`{case}`' for case in physical_cases)}.",
                "- Physical dense MVP scope: one DPU, one tasklet, fixed tiny dense contractions; this is hardware functionality evidence only, not performance, energy, scaling, or generic-TN evidence.",
            ]
        )
    if generic_hardware_records:
        completed = sum(
            1
            for row in generic_hardware_records
            if str(row.get("status") or "") == "completed"
        )
        exact = sum(
            1
            for row in generic_hardware_records
            if row.get("exact_integer_match") is True
        )
        physical_cases = sorted(
            {str(row.get("case_id") or "unknown") for row in generic_hardware_records}
        )
        lines.extend(
            [
                f"- Physical UPMEM generic TaskGraph MVP rows: {len(generic_hardware_records)}; completed: {completed}; exact int8/int32 matches: {exact}.",
                f"- Physical generic TaskGraph MVP cases: {', '.join(f'`{case}`' for case in physical_cases)}.",
                "- Physical generic TaskGraph scope: one DPU, one tasklet, one synthetic real-valued contraction; this is functionality evidence only, not a general quantum-TN, performance, energy, scaling, or scheduler result.",
            ]
        )
    lines.extend(
        [
            f"- UPMEM SDK simulator rows: {len(simulator_records)}.",
            f"- Strict generic-only UPMEM SDK simulator rows: {len(generic_records)}.",
            f"- CPU fallback flagged in UPMEM rows: {fallback_count}.",
            f"- Unsupported/boundary rows: {len(unsupported)}.",
        ]
    )
    if reasons:
        lines.append(
            f"- Top blocker reasons: {', '.join(f'{reason}={count}' for reason, count in reasons.most_common(5))}."
        )
    if tiling_records:
        lines.append(
            f"- Tiling support derived from records: {len(tiling_supported)}/{len(tiling_records)} rows report executable or observed tiling metadata."
        )
    else:
        lines.append(
            "- Tiling support derived from records: no tiling status was recorded."
        )
    if highest_supported is not None:
        lines.append(
            f"- Highest supported UPMEM case in these records: `{highest_supported[1].get('case_id')}` at `{int(highest_supported[0])}` qubits."
        )
    else:
        lines.append("- Highest supported UPMEM case in these records: none.")
    if first_unsupported is not None:
        first_case = first_unsupported[1].get("case_id")
        first_reason = _unsupported_reason(first_unsupported[1])
        lines.append(
            f"- First unsupported case by recorded qubit count: `{first_case}` at `{int(first_unsupported[0])}` qubits; reason: `{first_reason}`."
        )
        lines.append(
            f"- Next target derived from the boundary: investigate `{first_case}` and address `{first_reason}` without CPU fallback."
        )
    elif highest_supported is not None:
        lines.append(
            f"- Next target derived from the records: extend the strict generic-only sweep beyond `{highest_supported[1].get('case_id')}` and record the resulting boundary."
        )
    else:
        lines.append(
            "- Next target derived from the records: obtain a completed strict generic-only UPMEM record with capability metadata."
        )
    return lines


def _missing_evidence(records: list[JsonDict]) -> list[str]:
    routes = {str(record.get("route_id") or "") for record in records}
    missing: list[str] = []
    if "quest_cpu_full_state_exact" not in routes:
        missing.append("QuEST CPU full-state baseline records are absent.")
    if "quest_gpu_full_state_exact" not in routes:
        missing.append("Verified QuEST GPU records are absent.")
    if "quimb_tn_exact" not in routes:
        missing.append("Quimb TN baseline records are absent.")
    if not any(_is_strict_generic_upmem_record(record) for record in records):
        missing.append(
            "Strict generic-only UPMEM SDK-simulator boundary records are absent."
        )
    if not any(_is_physical_hardware_mvp_record(record) for record in records):
        missing.append(
            "Physical single-DPU UPMEM functionality-MVP records are absent."
        )
    return missing or [
        "No mandatory evidence class is obviously absent from loaded records."
    ]


def _plot_cpu_gpu_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    usable = [row for row in rows if _plot_qubits(row) is not None]
    if not usable:
        return "no_performance_tier_cpu_gpu_rows"
    families = sorted({str(row["case_family"]) for row in usable})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True
    )
    for axis, family in zip(axes.flat, families):
        group = sorted(
            (row for row in usable if str(row["case_family"]) == family),
            key=lambda row: int(_plot_qubits(row) or 0),
        )
        qubits = [int(_plot_qubits(row) or 0) for row in group]
        axis.errorbar(
            qubits,
            [float(row["cpu_simulation_compute_time_s_median"]) for row in group],
            yerr=_plot_iqr_error(group, "cpu_simulation_compute_time_s"),
            marker="o",
            capsize=3,
            label="QuEST CPU",
        )
        axis.errorbar(
            qubits,
            [float(row["gpu_simulation_compute_time_s_median"]) for row in group],
            yerr=_plot_iqr_error(group, "gpu_simulation_compute_time_s"),
            marker="s",
            capsize=3,
            label="QuEST GPU",
        )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Measured compute time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("Measured QuEST CPU/GPU full-state compute time (performance tier)")
    _save_plot(fig, path)
    return None


def _plot_cpu_gpu_speedup(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        if _plot_qubits(row) is not None:
            grouped[str(row["case_family"])].append(row)
    if not grouped:
        return "no_actual_qubit_metadata"
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for family, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: int(_plot_qubits(row) or 0))
        ax.errorbar(
            [int(_plot_qubits(row) or 0) for row in ordered],
            [float(row["compute_speedup_cpu_over_gpu_median"]) for row in ordered],
            yerr=_plot_iqr_error(ordered, "compute_speedup_cpu_over_gpu"),
            marker="o",
            capsize=3,
            label=family,
        )
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Qubits")
    ax.set_ylabel("Measured CPU/GPU compute ratio")
    ax.set_title("Measured CPU/GPU compute ratio by circuit family")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize="small", ncol=2)
    _save_plot(fig, path)
    return None


def _plot_cpu_tn_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id") in {"quimb_tn_exact", "quimb_tn_sliced_exact"}
        and _positive(row.get("simulation_compute_time_s_median")) is not None
    ]
    if not selected:
        return "no_quimb_tn_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True
    )
    for axis, family in zip(axes.flat, families):
        for route, label, marker in (
            ("quimb_tn_exact", "Quimb unsliced", "o"),
            ("quimb_tn_sliced_exact", "Quimb sliced", "s"),
        ):
            group = sorted(
                (
                    row
                    for row in selected
                    if row["route_id"] == route and str(row["case_family"]) == family
                ),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if group:
                axis.errorbar(
                    [int(_plot_qubits(row) or 0) for row in group],
                    [float(row["simulation_compute_time_s_median"]) for row in group],
                    yerr=_plot_iqr_error(group, "simulation_compute_time_s"),
                    marker=marker,
                    capsize=3,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median contraction time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("Measured Quimb unsliced and sliced contraction compute time")
    _save_plot(fig, path)
    return None


def _plot_full_state_vs_tn_runtime(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    routes = {
        "quest_cpu_full_state_exact": ("QuEST CPU full state", "o"),
        "quimb_tn_exact": ("Quimb TN unsliced", "s"),
        "quimb_tn_sliced_exact": ("Quimb TN sliced", "^"),
    }
    selected = [
        row
        for row in rows
        if row.get("suite_id") in {"thesis_cpu_tn_quimb", "research_cpu_tn"}
        and row.get("route_id") in routes
        and _plot_qubits(row) is not None
        and _positive(row.get("simulation_compute_time_s_median")) is not None
    ]
    if not selected:
        return "no_matching_full_state_and_tn_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True
    )
    for axis, family in zip(axes.flat, families):
        for route, (label, marker) in routes.items():
            group = sorted(
                (
                    row
                    for row in selected
                    if row["route_id"] == route and str(row["case_family"]) == family
                ),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if group:
                axis.errorbar(
                    [int(_plot_qubits(row) or 0) for row in group],
                    [float(row["simulation_compute_time_s_median"]) for row in group],
                    yerr=_plot_iqr_error(group, "simulation_compute_time_s"),
                    marker=marker,
                    capsize=3,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median simulation/contraction time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="x-small")
    fig.suptitle("Cross-algorithm/backend measured compute time")
    _save_plot(fig, path)
    return None


def _plot_tn_planning_vs_contraction(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id") in {"quimb_tn_exact", "quimb_tn_sliced_exact"}
        and _plot_qubits(row) is not None
        and _positive(row.get("simulation_compute_time_s_median")) is not None
    ]
    if not selected:
        return "no_quimb_timing_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True
    )
    styles = {
        "quimb_tn_exact": ("Quimb unsliced", "o"),
        "quimb_tn_sliced_exact": ("Quimb sliced", "s"),
    }
    for axis, family in zip(axes.flat, families):
        for route, (label, marker) in styles.items():
            group = sorted(
                (
                    row
                    for row in selected
                    if row["route_id"] == route and str(row["case_family"]) == family
                ),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if not group:
                continue
            x = [int(_plot_qubits(row) or 0) for row in group]
            axis.plot(
                x,
                [
                    max(float(row.get("planning_time_s_median") or 0.0), 1e-12)
                    for row in group
                ],
                marker=marker,
                linestyle="--",
                label=f"{label} planning",
            )
            axis.plot(
                x,
                [float(row["simulation_compute_time_s_median"]) for row in group],
                marker=marker,
                linestyle="-",
                label=f"{label} contraction",
            )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Median time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="x-small")
    fig.suptitle("Measured Quimb planning versus contraction compute time")
    _save_plot(fig, path)
    return None


def _plot_tn_path_metric(
    plt: Any,
    path: Path,
    rows: list[JsonDict],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id") in {"quimb_tn_exact", "quimb_tn_sliced_exact"}
        and _plot_qubits(row) is not None
        and _positive(row.get(metric)) is not None
    ]
    if not selected:
        return f"no_{metric}_rows"
    families = sorted({str(row["case_family"]) for row in selected})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True
    )
    for axis, family in zip(axes.flat, families):
        for route, label, marker in (
            ("quimb_tn_exact", "Quimb unsliced", "o"),
            ("quimb_tn_sliced_exact", "Quimb sliced", "s"),
        ):
            group = sorted(
                (
                    row
                    for row in selected
                    if row["route_id"] == route and str(row["case_family"]) == family
                ),
                key=lambda row: int(_plot_qubits(row) or 0),
            )
            if group:
                axis.plot(
                    [int(_plot_qubits(row) or 0) for row in group],
                    [float(row[metric]) for row in group],
                    marker=marker,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel(f"{ylabel} (log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle(title)
    _save_plot(fig, path)
    return None


def _plot_quantization_metric(
    plt: Any,
    path: Path,
    rows: list[JsonDict],
    *,
    metric: str,
    ylabel: str,
    title: str,
    log_scale: bool = False,
) -> str | None:
    selected = [
        row
        for row in rows
        if _plot_qubits(row) is not None and _positive(row.get(metric)) is not None
    ]
    if not selected:
        return f"no_{metric}_rows"
    ordered = sorted(
        selected,
        key=lambda row: (
            str(row["case_family"]),
            int(_plot_qubits(row) or 0),
            str(row["case_id"]),
        ),
    )
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.5), 5.2), constrained_layout=True
    )
    ax.bar(
        range(len(ordered)), [float(row[metric]) for row in ordered], color="#0f766e"
    )
    if "ratio" in metric:
        ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save_plot(fig, path)
    return None


def _plot_quantization_probability_error(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    """Render recorded probability errors without substituting amplitude errors."""
    selected = [
        row
        for row in rows
        if _plot_qubits(row) is not None
        and any(
            _positive(row.get(field)) is not None
            for field in (
                "quantized_probability_max_abs_error",
                "quantized_probability_l1_error",
            )
        )
    ]
    if not selected:
        return "no_quantized_probability_error_rows"
    ordered = sorted(
        selected,
        key=lambda row: (
            str(row["case_family"]),
            int(_plot_qubits(row) or 0),
            str(row["case_id"]),
        ),
    )
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.2), constrained_layout=True
    )
    x = list(range(len(ordered)))
    width = 0.38
    max_abs = [
        float(row["quantized_probability_max_abs_error"])
        if _positive(row.get("quantized_probability_max_abs_error")) is not None
        else 0.0
        for row in ordered
    ]
    l1 = [
        float(row["quantized_probability_l1_error"])
        if _positive(row.get("quantized_probability_l1_error")) is not None
        else 0.0
        for row in ordered
    ]
    ax.bar(
        [value - width / 2 for value in x],
        max_abs,
        width=width,
        label="maximum probability error",
        color="#ea580c",
    )
    ax.bar(
        [value + width / 2 for value in x],
        l1,
        width=width,
        label="probability L1 error",
        color="#0f766e",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Recorded probability error (log)")
    ax.set_title("UPMEM SDK simulator quantized probability error")
    ax.legend(fontsize="small")
    _save_plot(fig, path)
    return None


def _plot_same_plan_runtime(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    if not rows:
        return "no_same_plan_cpu_upmem_rows"
    families = sorted({str(row["case_family"]) for row in rows})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), constrained_layout=True, sharey=True
    )
    for axis, family in zip(axes.flat, families):
        group = [row for row in rows if str(row["case_family"]) == family]
        by_qubits: dict[int, list[JsonDict]] = defaultdict(list)
        for row in group:
            by_qubits[int(row.get("benchmark_n_qubits") or 0)].append(row)
        qubits = sorted(by_qubits)
        axis.plot(
            qubits,
            [float(by_qubits[value][0]["cpu_time_s"]) for value in qubits],
            marker="o",
            label="CPU TaskGraph",
        )
        for mode, label, marker in (
            ("none", "UPMEM float32", "s"),
            ("per_task_input_quantize", "UPMEM int8", "^"),
        ):
            mode_rows = {
                int(row.get("benchmark_n_qubits") or 0): row
                for row in group
                if row.get("quantization_mode") == mode
            }
            mode_qubits = sorted(mode_rows)
            if mode_qubits:
                axis.plot(
                    mode_qubits,
                    [
                        float(mode_rows[value]["upmem_simulator_time_s"])
                        for value in mode_qubits
                    ],
                    marker=marker,
                    label=label,
                )
        axis.set_yscale("log")
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.grid(True, alpha=0.3)
    for axis in axes[:, 0]:
        axis.set_ylabel("Route compute time (s, log)")
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    axes.flat[0].legend(fontsize="small")
    fig.suptitle("Same-plan CPU and UPMEM SDK simulator compute time")
    _save_plot(fig, path)
    return None


def _plot_planner_pressure(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("pim_feasible") is True
        and _positive(row.get("pim_estimated_flops")) is not None
        and _float_or_none(row.get("pim_objective_score")) is not None
    ]
    if len({str(row.get("planner_id")) for row in selected}) < 2:
        return "multiple_feasible_planner_candidates_not_available"
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("planner_id") or "unknown")].append(row)
    for planner_id, group in sorted(grouped.items()):
        ax.scatter(
            [float(row["pim_estimated_flops"]) for row in group],
            [float(row["pim_objective_score"]) for row in group],
            label=planner_id,
            alpha=0.75,
            marker="*" if any(bool(row.get("pim_selected")) for row in group) else "o",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Planner-estimated FLOPs")
    ax.set_ylabel("Normalized modeled PIM objective")
    ax.set_title("Planner-estimated FLOPs versus normalized modeled PIM objective")
    ax.legend(fontsize="small")
    _save_plot(fig, path)
    return None


def _plot_planner_components(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("pim_feasible") is True and _component_mapping(row)
    ]
    if len({str(row.get("planner_id")) for row in selected}) < 2:
        return "multiple_feasible_planner_candidates_not_available"
    component_names = (
        "flops",
        "host_dpu_bytes",
        "mram_wram_bytes",
        "local_work",
        "sync_events",
        "wram_pressure",
        "tiles",
        "numeric_penalty",
    )
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in selected:
        grouped[str(row.get("planner_id") or "unknown")].append(row)
    labels = sorted(grouped)
    values = {
        name: [
            statistics.mean(
                float(_component_mapping(row).get(name, 0.0) or 0.0)
                for row in grouped[label]
            )
            for label in labels
        ]
        for name in component_names
    }
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 1.4), 5.5), constrained_layout=True
    )
    bottom = [0.0] * len(labels)
    for name in component_names:
        current = values[name]
        ax.bar(labels, current, bottom=bottom, label=name)
        bottom = [left + right for left, right in zip(bottom, current)]
    ax.set_ylabel("Mean normalized modeled component")
    ax.set_title("Modeled PIM planner objective components")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(fontsize="x-small", ncol=2)
    _save_plot(fig, path)
    return None


def _plot_planner_component_diagnostics(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [
        row
        for row in rows
        if _float_or_none(row.get("pim_numeric_component_invocations")) is not None
        or _float_or_none(row.get("pim_mram_payload_pressure_ratio")) is not None
    ]
    if not selected:
        return "no v2 planner component diagnostics"
    selected = sorted(
        selected,
        key=lambda row: (
            str(row.get("case_id") or ""),
            str(row.get("planner_id") or ""),
        ),
    )
    labels = [str(row.get("planner_id") or "candidate") for row in selected]
    invocations = [
        float(row.get("pim_numeric_component_invocations") or 0.0) for row in selected
    ]
    recombination = [
        float(row.get("pim_numeric_recombination_flops") or 0.0) for row in selected
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(max(8.0, len(labels) * 0.5), 7.0), constrained_layout=True
    )
    axes[0].bar(range(len(labels)), invocations, color="#0f766e")
    axes[0].set_ylabel("Numeric component invocations")
    axes[0].set_title("V2 modeled numeric decomposition")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axes[1].bar(range(len(labels)), recombination, color="#be123c")
    axes[1].set_ylabel("Recombination FLOPs")
    axes[1].set_xlabel("Planner candidate")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    _save_plot(fig, path)
    return None


def _plot_planner_selection(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("pim_feasible") is True and row.get("pim_objective_rank") is not None
    ]
    if not selected:
        return "no_modeled_pim_selection_rows"
    labels = [f"{row['case_id']}\n{row['planner_id']}" for row in selected]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.2), constrained_layout=True
    )
    colors = ["#0f766e" if row.get("pim_selected") else "#94a3b8" for row in selected]
    ax.bar(
        range(len(selected)),
        [float(row["pim_objective_rank"]) for row in selected],
        color=colors,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Modeled objective rank (1 = selected)")
    ax.set_title("Modeled PIM planner selection")
    _save_plot(fig, path)
    return None


def _plot_planner_pareto(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("pim_feasible") is True
        and _positive(row.get("pim_estimated_flops")) is not None
        and _positive(row.get("pim_peak_intermediate_bytes")) is not None
        and row.get("pim_pareto_dominated") is not None
    ]
    if len(selected) < 2:
        return "insufficient_feasible_pareto_candidates"
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    for dominated, color, label in (
        (False, "#0f766e", "Pareto non-dominated"),
        (True, "#94a3b8", "Pareto dominated"),
    ):
        group = [
            row
            for row in selected
            if bool(row.get("pim_pareto_dominated")) is dominated
        ]
        if group:
            ax.scatter(
                [float(row["pim_estimated_flops"]) for row in group],
                [float(row["pim_peak_intermediate_bytes"]) for row in group],
                color=color,
                label=label,
                alpha=0.8,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Planner-estimated FLOPs")
    ax.set_ylabel("Planner-estimated largest intermediate bytes")
    ax.set_title("Planner-estimated Pareto frontier")
    ax.legend(fontsize="small")
    _save_plot(fig, path)
    return None


def _plot_planner_sensitivity(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("pim_feasible") is True
        and row.get("pim_selected") is True
        and row.get("pim_weight_profile")
        and _float_or_none(row.get("pim_objective_score")) is not None
    ]
    profiles = sorted({str(row["pim_weight_profile"]) for row in selected})
    if len(profiles) < 2:
        return "multiple_weight_profiles_not_available"
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in selected:
        grouped[str(row["pim_weight_profile"])].append(row)
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(profiles) * 1.8), 5.0), constrained_layout=True
    )
    ax.bar(
        profiles,
        [
            statistics.mean(float(row["pim_objective_score"]) for row in grouped[name])
            for name in profiles
        ],
        color="#2563eb",
    )
    ax.set_ylabel("Mean selected normalized modeled PIM objective")
    ax.set_title("Modeled PIM planner sensitivity by weight profile")
    ax.tick_params(axis="x", labelrotation=25)
    _save_plot(fig, path)
    return None


def _component_mapping(row: JsonDict) -> JsonDict:
    value = row.get("pim_normalized_components")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _plot_slicing_ratio(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id") == "quimb_tn_sliced_exact"
        and _positive(row.get("slicing_flop_ratio")) is not None
    ]
    if not selected:
        return "no_slicing_flop_ratio_rows"
    ordered = sorted(
        (row for row in selected if _plot_qubits(row) is not None),
        key=lambda row: (
            str(row["case_family"]),
            int(_plot_qubits(row) or 0),
            str(row["case_id"]),
        ),
    )
    if not ordered:
        return "no_actual_qubit_metadata"
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.45), 5.2), constrained_layout=True
    )
    ax.bar(
        range(len(ordered)),
        [float(row["slicing_flop_ratio"]) for row in ordered],
        color="#7c3aed",
    )
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Case")
    ax.set_ylabel("Sliced FLOPs / unsliced FLOPs")
    ax.set_title("Planner-estimated Quimb/cotengra slicing arithmetic ratio")
    _save_plot(fig, path)
    return None


def _plot_slicing_tradeoff(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if _plot_qubits(row) is not None
        and _positive(row.get("runtime_ratio_sliced_over_unsliced")) is not None
        and _positive(row.get("slicing_flop_ratio")) is not None
        and _positive(row.get("largest_intermediate_ratio_sliced_over_unsliced"))
        is not None
    ]
    if not selected:
        return "no_complete_compatible_slicing_tradeoff_pairs"
    ordered = sorted(
        selected,
        key=lambda row: (
            str(row.get("case_family") or ""),
            int(_plot_qubits(row) or 0),
            str(row.get("case_id") or ""),
        ),
    )
    labels = [
        f"{row.get('case_family') or 'case'}_{_plot_qubits(row)}q" for row in ordered
    ]
    x = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.6), 5.6), constrained_layout=True
    )
    metrics = (
        ("runtime_ratio_sliced_over_unsliced", "Runtime"),
        ("slicing_flop_ratio", "Planner FLOPs"),
        ("largest_intermediate_ratio_sliced_over_unsliced", "Largest intermediate"),
    )
    for index, (field, label) in enumerate(metrics):
        values = [float(row[field]) for row in ordered]
        offsets = [value + (index - 1) * width for value in x]
        ax.bar(offsets, values, width=width, label=label)
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Case")
    ax.set_ylabel("Ratio (sliced / unsliced, log)")
    ax.set_yscale("log")
    ax.set_title("Quimb slicing trade-off ratios")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize="small")
    _save_plot(fig, path)
    return None


def _plot_upmem_hardware_mvp_validation(
    plt: Any,
    path: Path,
    rows: list[JsonDict],
    *,
    title: str = "Physical UPMEM single-DPU MVP validation (functionality only)",
) -> str | None:
    if not rows:
        return "no_physical_upmem_mvp_rows"
    ordered = sorted(rows, key=lambda row: str(row.get("case_id") or ""))
    labels = [str(row.get("case_id") or "unknown") for row in ordered]
    passed = [int(row.get("validation_passed_count") or 0) for row in ordered]
    repeats = [int(row.get("repeat_count") or 0) for row in ordered]
    if not any(repeats):
        return "physical_upmem_mvp_rows_have_no_repeat_counts"
    fig, ax = plt.subplots(
        figsize=(max(7.0, len(labels) * 2.6), 5.2), constrained_layout=True
    )
    x = list(range(len(ordered)))
    bars = ax.bar(x, passed, color="#0f766e", label="exact CPU-reference validations")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(repeats) + 0.8)
    ax.set_ylabel("Validated physical executions")
    ax.set_title(title)
    for bar, pass_count, repeat_count in zip(bars, passed, repeats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            pass_count + 0.08,
            f"{pass_count}/{repeat_count}",
            ha="center",
            va="bottom",
        )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize="small")
    _save_plot(fig, path)
    return None


def _plot_upmem_boundary(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [row for row in rows if _is_strict_generic_upmem_record(row)]
    if not selected:
        return "no_upmem_rows"
    usable = [row for row in selected if _plot_qubits(row) is not None]
    if not usable:
        return "no_actual_qubit_metadata"
    families = sorted({str(row["case_family"]) for row in usable})
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 6.8), constrained_layout=True, sharey=True
    )
    for axis, family in zip(axes.flat, families):
        grouped: dict[tuple[str, int], list[JsonDict]] = defaultdict(list)
        for row in usable:
            if str(row["case_family"]) == family:
                grouped[(str(row["case_id"]), int(_plot_qubits(row) or 0))].append(row)
        points = sorted(grouped.items(), key=lambda item: item[0][1])
        qubits = [key[1] for key, _ in points]
        supported = [
            int(all(int(row.get("unsupported_count", 0) or 0) == 0 for row in group))
            for _, group in points
        ]
        axis.scatter(
            qubits,
            supported,
            c=["#16a34a" if value else "#dc2626" for value in supported],
            s=65,
            zorder=3,
        )
        axis.plot(qubits, supported, color="#94a3b8", linewidth=1.0, zorder=2)
        axis.set_title(family)
        axis.set_xlabel("Qubits")
        axis.set_yticks([0, 1])
        axis.set_yticklabels(["unsupported", "supported"])
        axis.set_ylim(-0.25, 1.25)
        axis.grid(True, alpha=0.3)
    for axis in axes.flat[len(families) :]:
        axis.set_visible(False)
    fig.suptitle("Strict generic UPMEM SDK simulator support boundary")
    _save_plot(fig, path)
    return None


def _plot_upmem_accuracy(plt: Any, path: Path, rows: list[JsonDict]) -> str | None:
    selected = [
        row
        for row in rows
        if _is_strict_generic_upmem_record(row)
        and _positive(row.get("max_abs_error")) is not None
    ]
    if not selected:
        return "no_upmem_error_rows"
    ordered = sorted(
        (row for row in selected if _plot_qubits(row) is not None),
        key=lambda row: (
            str(row["case_family"]),
            int(_plot_qubits(row) or 0),
            str(row["case_id"]),
        ),
    )
    if not ordered:
        return "no_actual_qubit_metadata"
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.45), 4.8), constrained_layout=True
    )
    ax.bar(
        range(len(ordered)),
        [float(row["max_abs_error"]) for row in ordered],
        color="#ea580c",
    )
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Max abs error (log scale)")
    ax.set_title("Strict generic UPMEM SDK simulator accuracy versus reference")
    _save_plot(fig, path)
    return None


def _plot_upmem_quantization_attribution(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    usable = [row for row in rows if _plot_qubits(row) is not None]
    if not usable:
        return "no_matched_generic_quantization_rows"
    ordered = sorted(
        usable,
        key=lambda row: (
            str(row["case_family"]),
            int(_plot_qubits(row) or 0),
            str(row["case_id"]),
        ),
    )
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    runtime_values = [
        row.get("route_runtime_ratio_none_over_quantized") for row in ordered
    ]
    transfer_values = [row.get("transfer_ratio_none_over_quantized") for row in ordered]
    if not any(
        _positive(value) is not None for value in [*runtime_values, *transfer_values]
    ):
        return "no_generic_quantization_ratios"
    fig, (runtime_ax, transfer_ax) = plt.subplots(
        2, 1, figsize=(max(8.0, len(labels) * 0.5), 7.0), constrained_layout=True
    )
    x = list(range(len(ordered)))
    for axis, values, title, ylabel in (
        (
            runtime_ax,
            runtime_values,
            "Host-side residual time ratio",
            "float32 host residual time / int8 host residual time",
        ),
        (
            transfer_ax,
            transfer_values,
            "Host/DPU transfer ratio",
            "float32 bytes / int8 bytes",
        ),
    ):
        axis.bar(
            x,
            [float(value) if _positive(value) is not None else 0.0 for value in values],
            color="#0f766e",
        )
        axis.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.3)
    transfer_ax.set_xticks(x)
    transfer_ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    transfer_ax.set_xlabel("Case")
    fig.suptitle("Strict generic UPMEM float32 vs int8 attribution (SDK simulator)")
    _save_plot(fig, path)
    return None


def _plot_quantization_transfer_bytes(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    """Render application-visible directional SDK transfer evidence.

    The runtime can observe payload-level H2D/D2H accounting, but not physical
    bus/DIMM traffic or unobservable SDK overhead. Keeping those boundaries in
    the figure prevents a byte-count plot from becoming a hardware claim.
    """
    usable = [
        row
        for row in rows
        if _plot_qubits(row) is not None
        and any(
            _positive(row.get(field)) is not None
            for field in (
                "unquantized_h2d_bytes",
                "quantized_h2d_bytes",
                "unquantized_d2h_bytes",
                "quantized_d2h_bytes",
            )
        )
    ]
    if not usable:
        return "no_directional_application_visible_transfer_bytes"
    ordered = sorted(
        usable,
        key=lambda row: (
            str(row["case_family"]),
            int(_plot_qubits(row) or 0),
            str(row["case_id"]),
        ),
    )
    labels = [f"{row['case_family']}_{_plot_qubits(row)}q" for row in ordered]
    x = list(range(len(ordered)))
    fig, (bytes_ax, ratio_ax) = plt.subplots(
        2, 1, figsize=(max(8.0, len(labels) * 0.6), 7.2), constrained_layout=True
    )
    width = 0.36
    float_h2d = [float(row.get("unquantized_h2d_bytes") or 0.0) for row in ordered]
    float_d2h = [float(row.get("unquantized_d2h_bytes") or 0.0) for row in ordered]
    int8_h2d = [float(row.get("quantized_h2d_bytes") or 0.0) for row in ordered]
    int8_d2h = [float(row.get("quantized_d2h_bytes") or 0.0) for row in ordered]
    left = [value - width / 2 for value in x]
    right = [value + width / 2 for value in x]
    bytes_ax.bar(left, float_h2d, width, label="float32 H2D", color="#2563eb")
    bytes_ax.bar(
        left, float_d2h, width, bottom=float_h2d, label="float32 D2H", color="#93c5fd"
    )
    bytes_ax.bar(right, int8_h2d, width, label="int8 H2D", color="#b45309")
    bytes_ax.bar(
        right, int8_d2h, width, bottom=int8_h2d, label="int8 D2H", color="#fdba74"
    )
    bytes_ax.set_ylabel("Application-visible SDK bytes")
    bytes_ax.set_title("Software-recorded application-visible H2D/D2H bytes")
    bytes_ax.legend(fontsize="small", ncol=2)
    bytes_ax.grid(True, axis="y", alpha=0.3)

    ratios = [row.get("transfer_ratio_none_over_quantized") for row in ordered]
    ratio_ax.bar(
        x,
        [float(value) if _positive(value) is not None else 0.0 for value in ratios],
        color="#0f766e",
    )
    ratio_ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ratio_ax.set_ylabel("float32 total bytes / int8 total bytes")
    ratio_ax.set_xlabel("Case")
    ratio_ax.set_xticks(x)
    ratio_ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ratio_ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("UPMEM SDK simulator transfer attribution (not physical bus traffic)")
    _save_plot(fig, path)
    return None


def _plot_physical_quantization_runtime(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [
        row
        for row in rows
        if _positive(row.get("float32_warm_runtime_s")) is not None
        and _positive(row.get("int8_warm_runtime_s")) is not None
        and row.get("float32_timing_class") == "measured_warm"
        and row.get("int8_timing_class") == "measured_warm"
    ]
    if not selected:
        return "no_matched_physical_measured_warm_runtime_rows"
    ordered = sorted(
        selected,
        key=lambda row: (str(row.get("case_id") or ""), int(row.get("repeat_id") or 0)),
    )
    labels = [f"{row.get('case_id')}_{row.get('repeat_id')}" for row in ordered]
    x = list(range(len(ordered)))
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.2), constrained_layout=True
    )
    ax.bar(
        [value - 0.18 for value in x],
        [float(row["float32_warm_runtime_s"]) for row in ordered],
        0.36,
        label="float32",
        color="#2563eb",
    )
    ax.bar(
        [value + 0.18 for value in x],
        [float(row["int8_warm_runtime_s"]) for row in ordered],
        0.36,
        label="int8",
        color="#b45309",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Measured warm route time (s, log)")
    ax.set_title("Physical UPMEM TaskGraph warm timing (not speedup)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, path)
    return None


def _plot_physical_quantization_transfer(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [
        row
        for row in rows
        if _positive(row.get("float32_transfer_bytes")) is not None
        and _positive(row.get("int8_transfer_bytes")) is not None
    ]
    if not selected:
        return "no_matched_physical_transfer_rows"
    ordered = sorted(
        selected,
        key=lambda row: (str(row.get("case_id") or ""), int(row.get("repeat_id") or 0)),
    )
    labels = [f"{row.get('case_id')}_{row.get('repeat_id')}" for row in ordered]
    x = list(range(len(ordered)))
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.2), constrained_layout=True
    )
    ax.bar(
        [value - 0.18 for value in x],
        [float(row["float32_transfer_bytes"]) for row in ordered],
        0.36,
        label="float32",
        color="#2563eb",
    )
    ax.bar(
        [value + 0.18 for value in x],
        [float(row["int8_transfer_bytes"]) for row in ordered],
        0.36,
        label="int8",
        color="#b45309",
    )
    ax.set_ylabel("Application-visible transfer bytes")
    ax.set_title("Physical UPMEM TaskGraph transfer attribution")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, path)
    return None


def _plot_physical_quantization_error(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [
        row
        for row in rows
        if _float_or_none(row.get("float32_max_abs_error")) is not None
        or _float_or_none(row.get("int8_max_abs_error")) is not None
    ]
    if not selected:
        return "no_physical_error_metadata"
    ordered = sorted(
        selected,
        key=lambda row: (str(row.get("case_id") or ""), int(row.get("repeat_id") or 0)),
    )
    labels = [f"{row.get('case_id')}_{row.get('repeat_id')}" for row in ordered]
    x = list(range(len(ordered)))
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.2), constrained_layout=True
    )
    ax.bar(
        [value - 0.18 for value in x],
        [float(row.get("float32_max_abs_error") or 0.0) for row in ordered],
        0.36,
        label="float32",
        color="#2563eb",
    )
    ax.bar(
        [value + 0.18 for value in x],
        [float(row.get("int8_max_abs_error") or 0.0) for row in ordered],
        0.36,
        label="int8",
        color="#b45309",
    )
    ax.set_ylabel("Recorded maximum absolute error")
    ax.set_title("Physical UPMEM TaskGraph validation error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.legend(fontsize="small")
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, path)
    return None


def _plot_physical_taskgraph_validation(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [row for row in rows if row.get("validation_passed") is not None]
    if not selected:
        return "no_physical_taskgraph_validation_rows"
    ordered = sorted(
        selected,
        key=lambda row: (
            str(row.get("case_id") or ""),
            int(row.get("repeat_id") or 0),
            str(row.get("quantization_mode") or ""),
        ),
    )
    labels = [
        f"{row.get('case_id')}_{row.get('repeat_id')}_{row.get('quantization_mode') or 'unknown'}"
        for row in ordered
    ]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.2), constrained_layout=True
    )
    ax.bar(
        range(len(ordered)),
        [1.0 if row.get("validation_passed") else 0.0 for row in ordered],
        color="#0f766e",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Validation passed (1/0)")
    ax.set_title("Physical UPMEM TaskGraph validation")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, path)
    return None


def _plot_physical_taskgraph_timing(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [row for row in rows if row.get("timing_class") == "measured_warm"]
    component_fields = (
        "allocation_time_s",
        "binary_load_time_s",
        "h2d_time_s",
        "d2h_time_s",
        "total_quantization_time_s",
        "total_dequantization_time_s",
        "total_bridge_time_s",
        "total_build_time_s",
        "validation_time_s",
        "output_materialization_time_s",
    )
    selected = [
        row
        for row in selected
        if any(_positive(row.get(field)) is not None for field in component_fields)
    ]
    if not selected:
        return "no_physical_measured_warm_timing_rows_bringup_excluded"
    ordered = sorted(
        selected,
        key=lambda row: (
            str(row.get("case_id") or ""),
            int(row.get("repeat_id") or 0),
            str(row.get("quantization_mode") or ""),
        ),
    )
    labels = [
        f"{row.get('case_id')}_{row.get('repeat_id')}_{row.get('quantization_mode') or 'unknown'}"
        for row in ordered
    ]
    x = list(range(len(ordered)))
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.55), 5.8), constrained_layout=True
    )
    bottom = [0.0] * len(ordered)
    colors = (
        "#64748b",
        "#94a3b8",
        "#2563eb",
        "#93c5fd",
        "#b45309",
        "#fdba74",
        "#0f766e",
        "#5eead4",
        "#7c3aed",
        "#c4b5fd",
    )
    labels_by_field = {
        "allocation_time_s": "allocation",
        "binary_load_time_s": "binary load",
        "h2d_time_s": "H2D",
        "d2h_time_s": "D2H",
        "total_quantization_time_s": "quantization",
        "total_dequantization_time_s": "dequantization",
        "total_bridge_time_s": "bridge",
        "total_build_time_s": "build",
        "validation_time_s": "validation",
        "output_materialization_time_s": "output",
    }
    for field, color in zip(component_fields, colors):
        values = [float(row.get(field) or 0.0) for row in ordered]
        ax.bar(x, values, 0.8, bottom=bottom, label=labels_by_field[field], color=color)
        bottom = [left + value for left, value in zip(bottom, values)]
    warm_values = [float(row.get("warm_runtime_s") or 0.0) for row in ordered]
    if any(warm_values):
        ax.plot(
            x,
            warm_values,
            color="#111827",
            marker="o",
            linewidth=1.5,
            label="warm route total",
        )
    ax.set_ylabel("Recorded measured warm time (s)")
    ax.set_title("Physical UPMEM TaskGraph timing breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.legend(fontsize="small", ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    _save_plot(fig, path)
    return None


def _plot_internal_parallelism(
    plt: Any, path: Path, rows: list[JsonDict]
) -> str | None:
    selected = [
        row
        for row in rows
        if row.get("route_id")
        in {"cpu_tn_frontier_exact", "cpu_tn_hybrid_sliced_frontier_exact"}
    ]
    if not selected:
        return "no_internal_parallelism_rows"
    selected = [row for row in selected if _plot_qubits(row) is not None]
    if not selected:
        return "no_actual_qubit_metadata"
    labels = [
        f"{row['route_id']}_{row['case_family']}_{_plot_qubits(row)}q"
        for row in selected
    ]
    widths = [float(row.get("max_frontier_width") or 0.0) for row in selected]
    fig, ax = plt.subplots(
        figsize=(max(8.0, len(labels) * 0.5), 4.8), constrained_layout=True
    )
    ax.bar(range(len(labels)), widths, color="#0891b2")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Max frontier width")
    ax.set_title("Internal diagnostic frontier width")
    _save_plot(fig, path)
    return None


def _save_plot(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover
        fig.clf()


def _caption(filename: str) -> str:
    return {
        "cpu_gpu_runtime_by_qubits.png": "Measured QuEST CPU and verified QuEST GPU compute time by circuit size.",
        "cpu_gpu_speedup_by_qubits.png": "Measured CPU/GPU compute ratio; values above 1 mean GPU compute was faster.",
        "cpu_gpu_energy_efficiency_by_qubits.png": "TODO until paired measured energy metadata is available.",
        "cpu_tn_runtime_by_qubits.png": "Measured Quimb unsliced and sliced tensor-network compute time by circuit size.",
        "full_state_vs_tn_runtime_by_qubits.png": "Cross-algorithm/backend measured QuEST full-state and Quimb TN compute time; not same-plan speedup.",
        "tn_planning_vs_contraction.png": "Measured Quimb planning and contraction compute time are reported separately.",
        "tn_path_flops_by_family_size.png": "Planner-estimated contraction FLOPs by circuit family and size.",
        "tn_path_peak_memory_by_family_size.png": "Planner-estimated largest intermediate tensor bytes by circuit family and size.",
        "cpu_tn_slicing_flop_ratio.png": "Slicing FLOP ratio = sliced cotengra plan reported FLOPs / unsliced cotengra plan reported FLOPs.",
        "cpu_tn_slicing_tradeoff.png": "Matched Quimb slicing trade-off ratios for runtime, planner-estimated FLOPs, and largest intermediate size (sliced / unsliced); not a speedup claim.",
        "upmem_supported_boundary.png": "Supported versus unsupported strict generic-only UPMEM SDK simulator rows.",
        "upmem_accuracy_error.png": "Strict generic UPMEM SDK simulator maximum absolute error where validation data exists.",
        "upmem_hardware_mvp_validation.png": "Physical UPMEM functionality MVP: exact CPU-reference validation counts by fixed dense case; not a performance figure.",
        "upmem_hardware_generic_mvp_validation.png": "Physical UPMEM generic TaskGraph MVP: exact CPU-reference validation counts for one fixed synthetic real contraction; not a quantum-circuit or performance figure.",
        "upmem_quantization_attribution.png": "Same-route float32 versus int8 attribution for strict generic UPMEM SDK simulator execution; not hardware speedup.",
        "quantization_runtime_by_executor.png": "Same-plan SDK simulator software-recorded host/control residual-time ratio; not hardware speedup.",
        "quantization_transfer_bytes.png": "Application-visible SDK H2D/D2H bytes and same-plan float32/int8 total-byte ratio; not physical bus traffic.",
        "quantization_error_by_family_size.png": "UPMEM SDK simulator int8 maximum absolute error against the full-precision TaskGraph reference.",
        "quantization_probability_error_by_family_size.png": "Recorded matched UPMEM SDK-simulator probability errors when validation records provide them; otherwise an explicit TODO figure.",
        "same_plan_cpu_upmem_runtime.png": "CPU replay and UPMEM SDK simulator rows share an identical contraction-plan hash; timing is not hardware speedup.",
        "upmem_physical_quantization_runtime.png": "Matched physical UPMEM TaskGraph float32/int8 warm route timing; bring-up-only wall time is excluded and this is not a speedup claim.",
        "upmem_physical_quantization_transfer.png": "Matched physical UPMEM TaskGraph application-visible float32/int8 transfer bytes; not physical bus traffic or a speedup claim.",
        "upmem_physical_quantization_error.png": "Matched physical UPMEM TaskGraph recorded float32/int8 validation error; missing error metadata is not inferred.",
        "upmem_physical_taskgraph_validation.png": "Physical UPMEM TaskGraph validation and task coverage, including bring-up rows; not a performance figure.",
        "upmem_physical_taskgraph_timing_breakdown.png": "Physical UPMEM TaskGraph measured warm timing components; bring-up-only timing is labeled and excluded.",
        "planner_flops_vs_upmem_pressure.png": "Planner-estimated FLOPs versus normalized modeled PIM objective for feasible candidates.",
        "planner_component_scores.png": "Normalized modeled PIM objective components; scenario weights are not measured hardware constants.",
        "planner_selection.png": "Selected feasible planner candidate per modeled PIM objective profile.",
        "planner_pareto_frontier.png": "Feasible planner candidates colored by modeled Pareto status.",
        "planner_sensitivity.png": "Selected planner candidates across modeled PIM weight profiles.",
        "internal_parallelism_metadata_by_qubits.png": "Diagnostic internal TaskGraph frontier metadata, not serious baseline performance.",
    }.get(filename, "")


def _stats(prefix: str, values: list[float]) -> JsonDict:
    if not values:
        return {
            f"{prefix}_median": None,
            f"{prefix}_p25": None,
            f"{prefix}_p75": None,
            f"{prefix}_iqr": None,
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_std": None,
        }
    if len(values) == 1:
        p25 = p75 = values[0]
    else:
        quartiles = statistics.quantiles(values, n=4, method="inclusive")
        p25, p75 = quartiles[0], quartiles[2]
    return {
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_p25": p25,
        f"{prefix}_p75": p75,
        f"{prefix}_iqr": p75 - p25,
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
        f"{prefix}_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _numbers(values: Iterable[Any]) -> list[float]:
    return [
        number
        for number in (_float_or_none(value) for value in values)
        if number is not None
    ]


def _positive(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or number <= 0:
        return None
    return number


def _reported_time(record: JsonDict) -> tuple[float | None, str]:
    """Select the recorded host residual without reconstructing it from totals."""
    residual = _float_or_none(record.get("total_host_residual_time_s"))
    if residual is not None and residual >= 0:
        return residual, "host_residual"
    total = _float_or_none(record.get("total_wall_time_s"))
    if total is not None and total >= 0:
        return total, "total_wall"
    return None, "unavailable"


def _reported_timing_basis(records: list[JsonDict]) -> str:
    bases = {_reported_time(record)[1] for record in records}
    if len(bases) == 1:
        return next(iter(bases))
    return "mixed"


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_value = _positive(numerator)
    denominator_value = _positive(denominator)
    if numerator_value is None or denominator_value is None:
        return None
    return numerator_value / denominator_value


def _difference(left: Any, right: Any) -> float | None:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _max_number(values: Iterable[Any]) -> float | None:
    numbers = _numbers(values)
    return max(numbers) if numbers else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validation_errors(record: JsonDict) -> JsonDict:
    metrics = record.get("validation_error_metrics")
    if isinstance(metrics, dict):
        return metrics
    if isinstance(metrics, str):
        try:
            parsed = json.loads(metrics)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _validation_metric(record: JsonDict, key: str) -> Any:
    """Read a normalized validation metric, including older direct-row records."""
    metrics = _validation_errors(record)
    if metrics.get(key) not in {None, ""}:
        return metrics[key]
    return record.get(key)


def _full_precision_errors(record: JsonDict) -> JsonDict:
    """Read full-precision accuracy fields without replacing execution validation."""
    candidates: list[Any] = [
        record,
        record.get("final_full_precision_accuracy"),
        record.get("full_precision_accuracy"),
        _notes(record).get("final_full_precision_accuracy"),
        _notes(record).get("full_precision_accuracy"),
    ]
    output: JsonDict = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for output_key, keys in {
            "max_abs_error": ("full_precision_max_abs_error", "max_abs_error"),
            "l2_error": ("full_precision_l2_error", "l2_error"),
            "norm_drift": ("full_precision_norm_drift", "norm_drift"),
        }.items():
            if output_key not in output:
                for key in keys:
                    if candidate.get(key) not in {None, ""}:
                        output[output_key] = candidate[key]
                        break
    return output


def _accuracy_errors_for_reporting(record: JsonDict) -> JsonDict:
    execution = _validation_errors(record)
    full_precision = _full_precision_errors(record)
    if (
        str(_record_value(record, "quantization_mode") or "")
        == "per_task_input_quantize"
    ):
        return {
            key: full_precision.get(key, execution.get(key))
            for key in ("max_abs_error", "l2_error", "norm_drift")
        }
    return execution


def _unsupported_reason(record: JsonDict) -> str:
    return str(
        _record_value(record, "resource_skip_reason")
        or _record_value(record, "reason")
        or record.get("validation_status")
        or record.get("status")
        or "unknown"
    )


def _tiling_status(record: JsonDict) -> bool | None:
    explicit = (
        record.get("tiling_implemented"),
        record.get("tiling_supported"),
        record.get("wram_output_tiled"),
        record.get("l2_tiled_execution"),
    )
    explicit_values = [_bool(value) for value in explicit if value is not None]
    if any(explicit_values):
        return True
    if explicit_values:
        return False
    for key in ("mram_tiled_task_count", "generic_output_tile_count"):
        value = _int_or_none(record.get(key))
        if value is not None:
            return value > 0 if key == "mram_tiled_task_count" else value > 1
    return None


def _notes(record: JsonDict) -> JsonDict:
    value = record.get("notes")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _record_value(record: JsonDict, key: str) -> Any:
    value = record.get(key)
    if value is not None and value != "":
        return value
    return _notes(record).get(key)


def _is_strict_generic_upmem_record(record: JsonDict) -> bool:
    return (
        record.get("contraction_execution_target") == "upmem"
        and str(_record_value(record, "policy") or "") == "generic-only"
        and str(_record_value(record, "quantization_mode") or "")
        in {"none", "per_task_input_quantize"}
    )


def _is_hardware_mvp_record(record: JsonDict) -> bool:
    return (
        record.get("contraction_execution_target") == "upmem"
        and record.get("benchmark_role") == "hardware_functionality_mvp"
    )


def _is_hardware_generic_mvp_record(record: JsonDict) -> bool:
    return (
        record.get("contraction_execution_target") == "upmem"
        and record.get("benchmark_role")
        == "hardware_generic_taskgraph_functionality_mvp"
    )


def _is_physical_hardware_mvp_record(record: JsonDict) -> bool:
    return _is_hardware_mvp_record(record) or _is_hardware_generic_mvp_record(record)


def _is_physical_upmem_taskgraph_record(record: JsonDict) -> bool:
    """Recognize physical TaskGraph evidence while keeping bring-up separate."""
    if record.get("contraction_execution_target") != "upmem":
        return False
    if _is_physical_hardware_mvp_record(record):
        return False
    mode = str(record.get("upmem_execution_mode") or "").lower()
    physical = (
        any(
            (
                record.get(field) is True
                for field in ("hardware_execution", "hardware_kernel_executed")
            )
        )
        or str(record.get("target_observed") or "").lower() == "hardware"
        or "hardware" in mode
    )
    if not physical:
        return False
    scope = " ".join(
        str(record.get(field) or "").lower()
        for field in (
            "execution_scope",
            "execution_plan_kind",
            "route_id",
            "backend_id",
        )
    )
    return (
        "taskgraph" in scope
        or _record_value(record, "task_count") is not None
        or _record_value(record, "upmem_task_count") is not None
    )


def _physical_taskgraph_dtype(record: JsonDict) -> str | None:
    mode = str(_record_value(record, "quantization_mode") or "").lower()
    dtype = str(
        _record_value(record, "input_dtype_on_dpu") or record.get("input_dtype") or ""
    ).lower()
    if (
        mode in {"none", "float32", "float32_no_quant", "no_quantization"}
        or dtype == "float32"
    ):
        return "float32"
    if (
        mode
        in {
            "per_task_input_quantize",
            "int8",
            "int8_scaled",
            "fixed_scale_identity_int8",
        }
        or dtype == "int8"
    ):
        return "int8"
    return None


def _physical_bool(record: JsonDict, field: str) -> bool | None:
    value = _record_value(record, field)
    return value if isinstance(value, bool) else None


def _physical_number(record: JsonDict, *fields: str) -> float | None:
    for field in fields:
        value = _float_or_none(_record_value(record, field))
        if value is not None:
            return value
    return None


def _physical_transfer_bytes(record: JsonDict) -> float | None:
    return _physical_number(
        record, "actual_transfer_bytes", "application_visible_transfer_bytes"
    )


def _physical_directional_transfer(record: JsonDict, direction: str) -> float | None:
    return _physical_number(
        record,
        f"actual_{direction}_bytes",
        f"application_visible_{direction}_bytes",
        f"{direction}_bytes",
    )


def _physical_timing_class(record: JsonDict) -> str:
    if (
        bool(_record_value(record, "timing_is_bringup_only"))
        or "bringup" in str(record.get("timing_scope") or "").lower()
    ):
        return "bringup_only"
    if (
        _record_value(record, "hardware_timing_available") is True
        and _physical_warm_runtime(record) is not None
    ):
        return "measured_warm"
    return "unavailable"


def _physical_warm_runtime(record: JsonDict) -> float | None:
    """Return only explicitly measured warm hardware timing, never bring-up wall time."""
    if _record_value(record, "hardware_timing_available") is not True:
        return None
    if bool(_record_value(record, "timing_is_bringup_only")):
        return None
    for field in (
        "warm_runtime_s",
        "warm_total_route_time_s",
        "warm_route_time_s",
        "measured_warm_runtime_s",
        "measured_warm_total_route_time_s",
        "warm_total_wall_time_s",
        "warm_runtime_wall_time_s",
        "upmem_runtime_warm_time_s",
        "total_route_time_s",
    ):
        value = _float_or_none(_record_value(record, field))
        if value is not None and value > 0:
            return value
    return None


def _physical_error(record: JsonDict) -> float | None:
    return _physical_number(
        record,
        "quantization_max_abs_error",
        "full_precision_max_abs_error",
        "validation_max_abs_error",
        "max_abs_error",
        "execution_max_abs_error",
    )


def _hardware_mvp_issues(record: JsonDict) -> list[str]:
    """Validate the narrow physical single-DPU functionality-evidence contract."""

    case = str(record.get("case_id") or "unknown")
    route = str(record.get("route_id") or "unknown")
    issues: list[str] = []
    if str(record.get("upmem_execution_mode") or "") != "sdk_hardware_single_dpu":
        issues.append(f"hardware MVP row has wrong execution mode: {case}/{route}")
    if str(record.get("target_requested") or "") != "hardware":
        issues.append(f"hardware MVP row did not request hardware: {case}/{route}")
    if bool(record.get("hardware_speedup_applicable", False)):
        issues.append(f"hardware MVP row marked speedup applicable: {case}/{route}")
    if bool(record.get("cpu_fallback_used", False)):
        issues.append(f"hardware MVP row used CPU fallback: {case}/{route}")
    if bool(record.get("simulator_kernel_executed", False)):
        issues.append(f"hardware MVP row executed simulator kernel: {case}/{route}")
    if str(record.get("status") or "") != "completed":
        return issues
    required_true = (
        "hardware_allocation_verified",
        "native_kernel_executed",
        "hardware_kernel_executed",
        "upmem_program_executed",
        "exact_integer_match",
    )
    for field in required_true:
        if record.get(field) is not True:
            issues.append(f"completed hardware MVP row lacks {field}: {case}/{route}")
    if str(record.get("target_observed") or "") != "hardware":
        issues.append(
            f"completed hardware MVP row did not observe hardware: {case}/{route}"
        )
    if str(record.get("validation_status") or "") != "passed":
        issues.append(
            f"completed hardware MVP row did not pass validation: {case}/{route}"
        )
    if _int_or_none(record.get("requested_dpu_count")) != 1:
        issues.append(f"hardware MVP row requested other than one DPU: {case}/{route}")
    if _int_or_none(record.get("allocated_dpu_count")) != 1:
        issues.append(f"hardware MVP row allocated other than one DPU: {case}/{route}")
    if _int_or_none(record.get("tasklets_per_dpu")) != 1:
        issues.append(f"hardware MVP row used other than one tasklet: {case}/{route}")
    return issues


def _strict_generic_upmem_issues(record: JsonDict) -> list[str]:
    """Reject non-generic UPMEM data from the research boundary group.

    Unsupported generic rows remain valid boundary evidence. Completed rows must
    prove the SDK generic loop executed every TaskGraph task without CPU fallback.
    """
    case = str(record.get("case_id") or "unknown")
    route = str(record.get("route_id") or "unknown")
    policy = str(_record_value(record, "policy") or "")
    mode = str(_record_value(record, "quantization_mode") or "")
    issues: list[str] = []
    if policy != "generic-only":
        issues.append(
            f"UPMEM research row is not generic-only: {case}/{route} policy={policy or 'missing'}"
        )
        return issues
    if mode not in {"none", "per_task_input_quantize"}:
        issues.append(
            f"UPMEM generic research row has unsupported quantization mode: {case}/{route} mode={mode or 'missing'}"
        )
        return issues
    if str(record.get("upmem_execution_mode") or "") != "sdk_simulator":
        issues.append(
            f"UPMEM generic research row is not SDK simulator execution: {case}/{route}"
        )
    if _bool(record.get("cpu_fallback_used")):
        issues.append(f"UPMEM generic research row used CPU fallback: {case}/{route}")
    if str(record.get("status") or "") != "completed":
        return issues
    if record.get("kernel_family") != "generic_loop_fallback":
        issues.append(
            f"completed UPMEM generic research row is not generic-loop evidence: {case}/{route}"
        )
    if _record_value(record, "generic_only_all_tasks_used_generic_backend") is not True:
        issues.append(
            f"completed UPMEM generic research row lacks all-task generic proof: {case}/{route}"
        )
    if _record_value(record, "valid_primary_upmem_codepath_result") is not True:
        issues.append(
            f"completed UPMEM generic research row lacks primary SDK-path proof: {case}/{route}"
        )
    if _record_value(record, "upmem_program_executed") is not True:
        issues.append(
            f"completed UPMEM generic research row lacks DPU program execution proof: {case}/{route}"
        )
    if (_int_or_none(_record_value(record, "dpu_program_invocations")) or 0) <= 0:
        issues.append(
            f"completed UPMEM generic research row lacks DPU invocations: {case}/{route}"
        )
    return issues


def _family_and_qubits(record: JsonDict) -> tuple[str, JsonDict]:
    case_id = str(record.get("case_id") or "")
    family = (
        case_id.split("_")[1]
        if case_id.startswith("quest_") and "_" in case_id
        else case_id.split("_")[0]
    )
    for key in (
        "actual_n_qubits",
        "benchmark_n_qubits",
        "case_n_qubits",
        "workload_n_qubits",
        "circuit_n_qubits",
        "n_qubits",
        "allocated_qubits",
    ):
        value = _int_or_none(record.get(key))
        if value is not None:
            return family, _qubit_metadata(value, key, None)
    workload_id = str(record.get("workload_id") or "")
    for source, text in (("case_id", case_id), ("workload_id", workload_id)):
        value = _qubits_from_identifier(text)
        if value is not None:
            return family, _qubit_metadata(value, source, None)
    warning = "actual_qubit_count_unresolved"
    return family, _qubit_metadata(None, None, warning)


def _qubit_metadata(
    value: int | None, source: str | None, warning: str | None
) -> JsonDict:
    return {
        "actual_n_qubits": value,
        "benchmark_n_qubits": value,
        "actual_n_qubits_source": source,
        "actual_n_qubits_warning": warning,
    }


def _qubits_from_identifier(value: str) -> int | None:
    match = re.search(r"(?:^|_)(\d+)q(?:_|$)", value)
    if not match:
        return None
    return _int_or_none(match.group(1))


def _plot_qubits(row: JsonDict) -> int | None:
    for key in ("benchmark_n_qubits", "actual_n_qubits", "n_qubits"):
        value = _int_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _plot_iqr_error(rows: list[JsonDict], prefix: str) -> list[list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        median = _float_or_none(row.get(f"{prefix}_median"))
        p25 = _float_or_none(row.get(f"{prefix}_p25"))
        p75 = _float_or_none(row.get(f"{prefix}_p75"))
        if median is None or p25 is None or p75 is None:
            lower.append(0.0)
            upper.append(0.0)
        else:
            lower.append(max(0.0, median - p25))
            upper.append(max(0.0, p75 - median))
    return [lower, upper]


def _first_present(group: list[JsonDict], key: str) -> Any:
    for row in group:
        value = row.get(key)
        if value not in {None, ""}:
            return value
    return None


def _transfer_invariant_status(group: list[JsonDict]) -> str | None:
    statuses = {
        str(row.get("actual_transfer_bytes_invariant"))
        for row in group
        if row.get("actual_transfer_bytes_invariant") not in {None, ""}
    }
    if not statuses:
        return None
    return next(iter(statuses)) if len(statuses) == 1 else "mixed"


def _same_or_mixed(left: Any, right: Any) -> Any:
    if left in {None, ""}:
        return right
    if right in {None, ""}:
        return left
    return left if left == right else "mixed"


def _paired_transfer_invariant_status(left: JsonDict, right: JsonDict) -> str | None:
    return _same_or_mixed(
        left.get("actual_transfer_bytes_invariant"),
        right.get("actual_transfer_bytes_invariant"),
    )


def _first_record_value(group: list[JsonDict], key: str) -> Any:
    for row in group:
        value = _record_value(row, key)
        if value is not None and value != "":
            return value
    return None


def _is_unsupported(record: JsonDict) -> bool:
    return (
        str(record.get("validation_status") or "") == "skipped"
        or str(record.get("status") or "") in {"unsupported", "failed"}
        or int(record.get("unsupported_task_count", 0) or 0) > 0
        or bool(record.get("resource_skip_reason"))
    )


def _valid_for_pair(record: JsonDict) -> bool:
    return str(record.get("validation_status") or "") in {
        "passed",
        "passed_native_status",
        "passed_runtime_only",
    }


def _selected_suites(suite_filter: list[str] | None) -> list[str]:
    return [
        key for key in SUITE_COMMAND_ORDER if not suite_filter or key in suite_filter
    ]


def _research_suite_argv(key: str, root: Path) -> list[str]:
    suite = RESEARCH_SUITES[key].relative_to(root).as_posix()
    if key in {"upmem_boundary", "upmem_quantization_stress"}:
        return [
            "upmem-mvp-benchmark",
            "--suite",
            suite,
            "--policies",
            "generic-only",
            "--quantization-modes",
            "none,per_task_input_quantize",
            "--execute-external",
            "--artifact-retention",
            "compact",
        ]
    if key in {
        "planner_paths",
        "planner_sensitivity",
        "planner_paths_v1",
        "planner_sensitivity_v1",
    }:
        return ["compare-planners", "--suite", suite]
    return [
        "simulation-backend-compare",
        "--suite",
        suite,
        "--artifact-retention",
        "compact",
    ]


def _pack_dir(root: Path, out: Path | None, *, label: str | None = None) -> Path:
    if out is not None:
        return out if out.is_absolute() else root / out
    namespace = _comparison_namespace(root, label)
    return namespace / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")


def _comparison_namespace(root: Path, label: str | None) -> Path:
    """Return a safe named comparison namespace, preserving the legacy default."""
    if label is None or not str(label).strip():
        return DEFAULT_COMPARISON_ROOT
    value = str(label).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(
            "comparison label must contain only letters, digits, '.', '_' or '-'"
        )
    return root / "runs" / "comparisons" / value


def _update_latest_link(parent: Path, target: Path) -> None:
    latest = parent / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(target.name)
    except OSError:
        return


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _latest_evidence_for_suite(root: Path, suite_id: str) -> Path | None:
    suite_root = root / "runs" / "evidence" / suite_id
    if not suite_root.exists():
        return None
    latest = suite_root / "latest"
    if (
        latest.is_symlink()
        and latest.exists()
        and (latest.resolve() / "normalized_records.jsonl").is_file()
    ):
        return latest.resolve()
    candidates = [
        path.parent for path in suite_root.glob("*/*/normalized_records.jsonl")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _write_csv(path: Path, rows: list[JsonDict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_optional_json(path: Path) -> JsonDict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"read_error": "invalid_json", "path": path.as_posix()}


def _evidence_source_provenance(evidence_inputs: list[Path]) -> JsonDict:
    commits: set[str] = set()
    source_dirty = False
    repository_dirty = False
    for evidence in evidence_inputs:
        root = evidence if evidence.is_dir() else evidence.parent
        manifest = _read_optional_json(root / "run_manifest.json") or {}
        commit = str(
            manifest.get("benchmark_source_commit") or manifest.get("git_commit") or ""
        )
        if commit:
            commits.add(commit)
        source_dirty = source_dirty or bool(
            manifest.get("benchmark_source_worktree_dirty")
            if manifest.get("benchmark_source_worktree_dirty") is not None
            else manifest.get("dirty_tree", manifest.get("dirty_worktree", False))
        )
        repository_dirty = repository_dirty or bool(
            manifest.get("repository_worktree_dirty", False)
        )
    ordered_commits = sorted(commits)
    return {
        "commit": ordered_commits[0] if len(ordered_commits) == 1 else None,
        "commits": ordered_commits,
        "worktree_dirty": source_dirty,
        "repository_worktree_dirty": repository_dirty,
    }


def _gpu_verification_passed(root: Path) -> bool:
    payload = _read_optional_json(
        root / "build" / "gpu_verification" / "quest_gpu_full_state_exact.json"
    )
    return bool(
        payload
        and payload.get("gpu_backend_verified") is True
        and payload.get("gpu_program_executed") is True
    )


def _gpu_blocker_reason(root: Path) -> str:
    payload = (
        _read_optional_json(
            root / "build" / "gpu_verification" / "quest_gpu_full_state_exact.json"
        )
        or {}
    )
    return str(
        payload.get("blocker_reason")
        or payload.get("status")
        or "gpu_verification_failed"
    )


def _skipped_group_result(group: str, reason: str) -> JsonDict:
    return {
        "command": f"skip research group {group}",
        "returncode": 0,
        "stdout": f"Skipped {group}: {reason}",
        "stderr": "",
        "skipped_group": group,
        "blocker_reason": reason,
        "benchmark_rows_emitted": False,
    }


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _cpu_metadata() -> JsonDict:
    model = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "model": model or platform.processor() or None,
        "logical_count": os.cpu_count(),
    }


def _git(root: Path, args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _run_capture(root: Path, argv: list[str]) -> JsonDict:
    result = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(argv),
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _bench_argv(args: list[str]) -> list[str]:
    return [sys.executable, "-m", "quantum_bench.bench", *args]


def _bench_command(args: list[str]) -> str:
    return f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python -m quantum_bench.bench {' '.join(args)}"


def _command(args: list[str], *, python_script: bool = False) -> str:
    if python_script:
        return f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../.venv/bin/python {' '.join(args)}"
    return " ".join(args)


if __name__ == "__main__":
    raise SystemExit(main())
