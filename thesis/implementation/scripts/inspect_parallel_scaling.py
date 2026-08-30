#!/usr/bin/env python3
"""Inspect the fixed one-rank tasklet and DPU scaling diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from quantum_bench.evidence import canonical_json, load_artifacts
from quantum_bench.report import verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
_ANALYSIS_VERSION = "parallel_scaling_diagnostic_v1"
_ROUTE_SPECS = {
    "upmem_float32_1dpu_t1": (1, 1),
    "upmem_float32_1dpu_t2": (1, 2),
    "upmem_float32_1dpu_t4": (1, 4),
    "upmem_float32_1dpu_t8": (1, 8),
    "upmem_float32_2dpu_t8": (2, 8),
    "upmem_float32_4dpu_t8": (4, 8),
}
_ROUTE_IDS = tuple(_ROUTE_SPECS)
_PRIMARY_COMPARISONS = (
    ("tasklet_scaling", "upmem_float32_1dpu_t1", "upmem_float32_1dpu_t2"),
    ("tasklet_scaling", "upmem_float32_1dpu_t1", "upmem_float32_1dpu_t4"),
    ("tasklet_scaling", "upmem_float32_1dpu_t1", "upmem_float32_1dpu_t8"),
    ("dpu_scaling", "upmem_float32_1dpu_t8", "upmem_float32_2dpu_t8"),
    ("dpu_scaling", "upmem_float32_1dpu_t8", "upmem_float32_4dpu_t8"),
)
_COMMON_IDENTITIES = (
    "environment_id",
    "problem_id",
    "tensor_network_structure_id",
    "logical_plan_id",
    "validation_policy_id",
)
_BINARY_HASHES = (
    "host_binary_sha256",
    "dpu_binary_sha256",
    "initialization_binary_sha256",
)
_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PLOT_LABELS = {
    "upmem_float32_1dpu_t1": "T1",
    "upmem_float32_1dpu_t2": "T2",
    "upmem_float32_1dpu_t4": "T4",
    "upmem_float32_1dpu_t8": "T8",
    "upmem_float32_2dpu_t8": "2D-T8",
    "upmem_float32_4dpu_t8": "4D-T8",
}
_ROUTE_STAT_COLUMNS = (
    "route",
    "dpus",
    "tasklets",
    "median_total_wall_s",
    "raw_mad_total_wall_s",
    "median_kernel_s",
    "raw_mad_kernel_s",
    "median_h2d_bytes",
    "median_d2h_bytes",
    "max_abs_error",
    "relative_l2_error",
    "norm_drift",
    "tasklet_utilization",
    "dpu_utilization",
    "dominant_wave_utilization",
)
_COMPARISON_COLUMNS = (
    "comparison_kind",
    "baseline_route",
    "candidate_route",
    "resource_ratio",
    "kernel_speedup",
    "total_wall_speedup",
    "kernel_parallel_efficiency",
    "total_wall_parallel_efficiency",
    "diagnostic_only",
)
_ACCURACY_COLUMNS = (
    "route",
    "max_abs_error",
    "relative_l2_error",
    "norm_drift",
    "float32_atol",
    "float32_relative_l2_max",
    "float32_norm_drift_max",
)


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _nonnegative(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _optional_nonnegative(
    values: list[float], source: Mapping[str, Any], field: str
) -> None:
    value = source.get(field)
    if value is not None:
        values.append(_nonnegative(value, field))


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a SHA-256 identity")
    return value


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode or len(commit) != 40:
        raise ValueError("cannot determine the current source commit")
    return commit


def _joined_facts(
    sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]
) -> Mapping[str, Any]:
    joined = dict(_mapping(sample.get("backend_facts"), "sample backend facts"))
    session = sessions.get(str(sample.get("session_instance_id")))
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        for field, value in terminal.items():
            joined.setdefault(field, value)
    return joined


def _validate_configuration(configuration: Mapping[str, Any]) -> None:
    collection = _mapping(configuration.get("collection"), "collection")
    expected_collection = {
        "claim_policy": "diagnostic_v1",
        "warmup_blocks": 1,
        "measurement_blocks": 5,
        "session_policy": "fresh_session_per_attempt_v1",
        "block_cooldown_s": 0.0,
    }
    if any(
        collection.get(field) != value
        for field, value in expected_collection.items()
    ):
        raise ValueError("parallel diagnostic collection contract changed")
    if set(configuration.get("cases", {})) != {"scaling_primary"}:
        raise ValueError("parallel diagnostic requires only scaling_primary")
    circuit = configuration["cases"]["scaling_primary"]["circuit"]
    if (
        circuit.get("kind") != "builtin"
        or circuit.get("name") != "quantization_stress"
        or dict(circuit.get("parameters", {}))
        != {"n_qubits": 18, "repeat_layers": 2}
    ):
        raise ValueError("parallel diagnostic Stress18 circuit changed")
    if set(configuration.get("plans", {})) != {"greedy"}:
        raise ValueError("parallel diagnostic requires only the greedy plan")
    plan = configuration["plans"]["greedy"]
    if (
        dict(plan.get("planner", {}))
        != {"engine": "opt_einsum", "mode": "greedy"}
        or plan.get("slicing") is not None
    ):
        raise ValueError("parallel diagnostic planner changed")
    routes = _mapping(configuration.get("routes"), "routes")
    if tuple(routes) != _ROUTE_IDS:
        raise ValueError("parallel diagnostic route set changed")
    for route_id, (dpu_count, tasklets) in _ROUTE_SPECS.items():
        route = _mapping(routes[route_id], route_id)
        options = _mapping(route.get("options"), f"{route_id} options")
        if (
            route.get("executor") != "upmem_physical"
            or route.get("numeric_policy") != "split_complex_float32_v1"
            or (
                options.get("dpu_count"),
                options.get("rank_count"),
                options.get("tasklets_per_dpu"),
            )
            != (dpu_count, 1, tasklets)
        ):
            raise ValueError(f"parallel diagnostic route changed: {route_id}")
    matrix = configuration.get("matrix")
    if not isinstance(matrix, Sequence) or len(matrix) != 1:
        raise ValueError("parallel diagnostic requires one matrix entry")
    item = matrix[0]
    if (
        item.get("case_id") != "scaling_primary"
        or item.get("plan_id") != "greedy"
        or tuple(item.get("route_ids", ())) != _ROUTE_IDS
    ):
        raise ValueError("parallel diagnostic matrix changed")


def _median_mad(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise ValueError("each route requires five measurement timings")
    center = float(median(values))
    return center, float(median(abs(value - center) for value in values))


def derive_summary(
    *,
    manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    expected_source_commit: str,
    reporting_tool_source_commit: str | None = None,
) -> Mapping[str, object]:
    """Validate the fixed protocol and derive descriptive scaling ratios."""

    configuration = _mapping(manifest.get("configuration"), "configuration")
    _validate_configuration(_mapping(configuration.get("experiment"), "experiment"))
    environment = _mapping(configuration.get("environment"), "environment")
    if tuple(environment.get("affinity", ())) != (0,):
        raise ValueError("parallel diagnostic requires CPU affinity [0]")
    if dict(environment.get("observed_cpu_governors", {})) != {"0": "powersave"}:
        raise ValueError("parallel diagnostic requires the powersave governor")
    if dict(environment.get("thread_environment", {})) != _THREAD_ENVIRONMENT:
        raise ValueError("parallel diagnostic requires one CPU thread per library")
    if manifest.get("source_commit") != expected_source_commit:
        raise ValueError("parallel diagnostic source commit does not match")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("parallel diagnostic source was dirty")
    if manifest.get("status") != "completed":
        raise ValueError("parallel diagnostic manifest is not completed")
    route_counts = {
        route_id: sum(sample.get("route_id") == route_id for sample in samples)
        for route_id in _ROUTE_IDS
    }
    if len(samples) != 36 or set(route_counts.values()) != {6}:
        raise ValueError("parallel diagnostic sample count is incomplete")
    session_map = {
        str(session.get("session_instance_id")): session for session in sessions
    }
    sample_session_ids = {str(sample.get("session_instance_id")) for sample in samples}
    if len(sessions) != 36 or sample_session_ids != set(session_map):
        raise ValueError("parallel diagnostic session matrix is incomplete")

    common_ids = {field: set() for field in _COMMON_IDENTITIES}
    plan_ids = {route_id: set() for route_id in _ROUTE_IDS}
    executable_ids = {route_id: set() for route_id in _ROUTE_IDS}
    binary_ids = {route_id: set() for route_id in _ROUTE_IDS}
    route_values = {
        route_id: {
            "total_wall_s": [],
            "kernel_s": [],
            "h2d_bytes": [],
            "d2h_bytes": [],
            "max_abs_error": [],
            "relative_l2_error": [],
            "norm_drift": [],
            "tasklet_utilization": set(),
            "dpu_utilization": set(),
            "dominant_wave_utilization": set(),
        }
        for route_id in _ROUTE_IDS
    }
    for sample in samples:
        route_id = str(sample["route_id"])
        dpu_count, tasklets = _ROUTE_SPECS[route_id]
        session = session_map[str(sample["session_instance_id"])]
        if session.get("route_id") != route_id:
            raise ValueError("sample and session route identities differ")
        identities = _mapping(sample.get("identities"), "sample identities")
        for field in _COMMON_IDENTITIES:
            common_ids[field].add(_sha256(identities.get(field), field))
        plan_ids[route_id].add(
            _sha256(identities.get("physical_plan_id"), "physical_plan_id")
        )
        executable_ids[route_id].add(
            _sha256(identities.get("executable_id"), "executable_id")
        )
        validation = _mapping(sample.get("validation"), "sample validation")
        if any(
            validation.get(field) is not True
            for field in (
                "accuracy_qualified",
                "policy_reference_passed",
                "full_precision_passed",
            )
        ):
            raise ValueError(f"float32 validation failed for {route_id}")
        _nonnegative(validation.get("relative_l2_error"), "relative_l2_error")
        _nonnegative(validation.get("norm_drift"), "norm_drift")
        facts = _joined_facts(sample, session_map)
        required = {
            "target_observed": "physical_hardware",
            "physical_target_verified": True,
            "hardware_kernel_executed": True,
            "simulator_kernel_executed": False,
            "cpu_fallback_used": False,
            "startup_resource_admission_passed": True,
            "execution_resource_admission_passed": True,
            "requested_dpus": dpu_count,
            "allocated_dpus": dpu_count,
            "active_dpus": dpu_count,
            "execution_active_dpu_count": dpu_count,
            "execution_active_rank_count": 1,
            "tasklets_per_dpu": tasklets,
            "observed_tasklets_per_dpu": tasklets,
            "dominant_work_wave_populated_dpu_slots": dpu_count,
            "dominant_work_wave_allocated_dpu_slots": dpu_count,
            "dominant_work_wave_tasklet_row_sufficiency_passed": True,
        }
        if any(facts.get(field) != value for field, value in required.items()):
            raise ValueError(f"resource or provenance mismatch for {route_id}")
        _sha256(facts.get("output_hash"), "output_hash")
        binary_ids[route_id].add(
            tuple(_sha256(facts.get(field), field) for field in _BINARY_HASHES)
        )
        measurement = _mapping(sample.get("measurement"), "measurement")
        if sample.get("attempt_kind") == "measurement":
            route_values[route_id]["total_wall_s"].append(
                _nonnegative(measurement.get("total_wall_s"), "total_wall_s")
            )
            route_values[route_id]["kernel_s"].append(
                _nonnegative(measurement.get("kernel_s"), "kernel_s")
            )
            for field in ("h2d_bytes", "d2h_bytes"):
                _optional_nonnegative(route_values[route_id][field], measurement, field)
            for field in ("max_abs_error", "relative_l2_error", "norm_drift"):
                _optional_nonnegative(route_values[route_id][field], validation, field)
        for source, target in (
            ("arithmetic_weighted_tasklet_utilization", "tasklet_utilization"),
            ("arithmetic_weighted_dpu_slot_utilization", "dpu_utilization"),
            ("dominant_work_wave_utilization", "dominant_wave_utilization"),
        ):
            value = _nonnegative(facts.get(source), source)
            if value > 1.0:
                raise ValueError(f"{source} must not exceed one")
            route_values[route_id][target].add(value)

    if any(len(values) != 1 for values in common_ids.values()):
        raise ValueError("logical identities differ across scaling routes")
    if any(len(values) != 1 for values in plan_ids.values()):
        raise ValueError("physical plan identity is inconsistent within a route")
    if len({next(iter(values)) for values in plan_ids.values()}) != 6:
        raise ValueError("physical plan identities do not distinguish topology")
    if any(len(values) != 1 for values in executable_ids.values()):
        raise ValueError("executable identity is inconsistent within a route")
    executable_by_route = {
        route_id: next(iter(values)) for route_id, values in executable_ids.items()
    }
    if len(
        {executable_by_route[route] for route in _ROUTE_IDS if route.endswith("t8")}
    ) != 1:
        raise ValueError("T8 routes do not share one executable identity")
    if len(
        {
            executable_by_route[f"upmem_float32_1dpu_t{tasklets}"]
            for tasklets in (1, 2, 4, 8)
        }
    ) != 4:
        raise ValueError("tasklet builds do not have distinct executable identities")
    if any(len(values) != 1 for values in binary_ids.values()):
        raise ValueError("binary hashes are inconsistent within a route")
    binary_by_route = {
        route_id: next(iter(values)) for route_id, values in binary_ids.items()
    }
    if len(
        {binary_by_route[route] for route in _ROUTE_IDS if route.endswith("t8")}
    ) != 1:
        raise ValueError("T8 routes do not share one binary triple")

    route_statistics: dict[str, Mapping[str, float | int]] = {}
    warnings: list[str] = []
    for route_id, (dpu_count, tasklets) in _ROUTE_SPECS.items():
        values = route_values[route_id]
        if any(
            len(values[field]) != 1
            for field in (
                "tasklet_utilization",
                "dpu_utilization",
                "dominant_wave_utilization",
            )
        ):
            raise ValueError(f"utilization facts vary within route {route_id}")
        total_median, total_mad = _median_mad(values["total_wall_s"])
        kernel_median, kernel_mad = _median_mad(values["kernel_s"])
        route_statistics[route_id] = {
            "dpu_count": dpu_count,
            "tasklets_per_dpu": tasklets,
            "median_total_wall_s": total_median,
            "mad_total_wall_s": total_mad,
            "median_kernel_s": kernel_median,
            "mad_kernel_s": kernel_mad,
            "median_h2d_bytes": (
                float(median(values["h2d_bytes"])) if values["h2d_bytes"] else None
            ),
            "median_d2h_bytes": (
                float(median(values["d2h_bytes"])) if values["d2h_bytes"] else None
            ),
            "max_abs_error": max(values["max_abs_error"])
            if values["max_abs_error"]
            else None,
            "relative_l2_error": max(values["relative_l2_error"])
            if values["relative_l2_error"]
            else None,
            "norm_drift": max(values["norm_drift"])
            if values["norm_drift"]
            else None,
            "arithmetic_weighted_tasklet_utilization": next(
                iter(values["tasklet_utilization"])
            ),
            "arithmetic_weighted_dpu_slot_utilization": next(
                iter(values["dpu_utilization"])
            ),
            "dominant_work_wave_utilization": next(
                iter(values["dominant_wave_utilization"])
            ),
        }
        for name, center, raw_mad in (
            ("total_wall", total_median, total_mad),
            ("kernel", kernel_median, kernel_mad),
        ):
            if center <= 0.0 or raw_mad / center > 0.15:
                warnings.append(f"runtime_variability:{route_id}:{name}")

    comparisons: list[Mapping[str, object]] = []
    for kind, baseline_id, candidate_id in _PRIMARY_COMPARISONS:
        baseline = route_statistics[baseline_id]
        candidate = route_statistics[candidate_id]
        resource_field = (
            "tasklets_per_dpu" if kind == "tasklet_scaling" else "dpu_count"
        )
        resource_ratio = float(candidate[resource_field]) / float(
            baseline[resource_field]
        )
        total_speedup = float(baseline["median_total_wall_s"]) / float(
            candidate["median_total_wall_s"]
        )
        kernel_speedup = float(baseline["median_kernel_s"]) / float(
            candidate["median_kernel_s"]
        )
        comparisons.append(
            {
                "comparison_kind": kind,
                "baseline_route_id": baseline_id,
                "candidate_route_id": candidate_id,
                "resource_ratio": resource_ratio,
                "total_wall_speedup": total_speedup,
                "total_wall_parallel_efficiency": total_speedup / resource_ratio,
                "kernel_speedup": kernel_speedup,
                "kernel_parallel_efficiency": kernel_speedup / resource_ratio,
            }
        )
    experiment = _mapping(configuration.get("experiment"), "experiment")
    collection = _mapping(experiment.get("collection"), "collection")
    environment = _mapping(configuration.get("environment"), "environment")
    route_config = _mapping(experiment.get("routes"), "routes")
    rank_paths = sorted(
        {
            str(path)
            for route_id in _ROUTE_IDS
            for path in _mapping(
                _mapping(route_config[route_id], route_id).get("options"),
                f"{route_id} options",
            ).get("rank_paths", ())
        }
    )
    return {
        "analysis_version": _ANALYSIS_VERSION,
        "source_commit": manifest["source_commit"],
        "physical_execution_source_commit": manifest["source_commit"],
        "reporting_tool_source_commit": reporting_tool_source_commit
        or _source_commit(),
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "expected_route_ids": list(_ROUTE_IDS),
        "expected_block_ids": list(range(6)),
        "sample_count": 36,
        "session_count": 36,
        "measurement_count_per_route": {route_id: 5 for route_id in _ROUTE_IDS},
        "rank_paths": rank_paths,
        "cpu_affinity": list(environment.get("affinity", ())),
        "observed_governor": environment.get("observed_cpu_governors", {}),
        "sdk_version": environment.get("upmem_sdk_version"),
        "claim_policy": collection.get("claim_policy"),
        "claim_eligible": False,
        "claim_ineligibility_reason": "diagnostic_claim_policy",
        "route_statistics": route_statistics,
        "primary_comparisons": comparisons,
        "measurement_warnings": warnings,
        "gate_passed": True,
    }


def inspect_artifacts(
    *,
    input_dir: Path,
    summary_output: Path,
    output_dir: Path | None = None,
    expected_source_commit: str | None = None,
) -> Mapping[str, object]:
    verification = verify_artifacts(input_dir)
    if verification.get("status") != "completed":
        raise ValueError("parallel diagnostic evidence is not completed")
    manifest, samples, sessions = load_artifacts(input_dir)
    summary = derive_summary(
        manifest=manifest,
        samples=samples,
        sessions=sessions,
        expected_source_commit=expected_source_commit or _source_commit(),
        reporting_tool_source_commit=_source_commit(),
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(_plain(summary), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if output_dir is not None:
        _write_parallel_outputs(
            output_dir=output_dir,
            summary=summary,
            manifest=manifest,
            samples=samples,
        )
    return summary


def _measurement_rows(
    samples: Sequence[Mapping[str, Any]], route_id: str
) -> list[Mapping[str, Any]]:
    rows = [
        sample
        for sample in samples
        if sample.get("route_id") == route_id
        and sample.get("attempt_kind") == "measurement"
        and sample.get("status") == "success"
    ]
    return sorted(rows, key=lambda row: int(row.get("block_id", 0)))


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    return value


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _route_table_rows(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = []
    statistics = _mapping(summary.get("route_statistics"), "route_statistics")
    for route_id in _ROUTE_IDS:
        values = _mapping(statistics[route_id], route_id)
        rows.append(
            {
                "route": route_id,
                "dpus": values.get("dpu_count"),
                "tasklets": values.get("tasklets_per_dpu"),
                "median_total_wall_s": values.get("median_total_wall_s"),
                "raw_mad_total_wall_s": values.get("mad_total_wall_s"),
                "median_kernel_s": values.get("median_kernel_s"),
                "raw_mad_kernel_s": values.get("mad_kernel_s"),
                "median_h2d_bytes": values.get("median_h2d_bytes"),
                "median_d2h_bytes": values.get("median_d2h_bytes"),
                "max_abs_error": values.get("max_abs_error"),
                "relative_l2_error": values.get("relative_l2_error"),
                "norm_drift": values.get("norm_drift"),
                "tasklet_utilization": values.get(
                    "arithmetic_weighted_tasklet_utilization"
                ),
                "dpu_utilization": values.get(
                    "arithmetic_weighted_dpu_slot_utilization"
                ),
                "dominant_wave_utilization": values.get(
                    "dominant_work_wave_utilization"
                ),
            }
        )
    return rows


def _comparison_table_rows(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    comparisons = summary.get("primary_comparisons")
    if not isinstance(comparisons, Sequence):
        raise ValueError("primary_comparisons must be a sequence")
    return [
        {
            "comparison_kind": comparison.get("comparison_kind"),
            "baseline_route": comparison.get("baseline_route_id"),
            "candidate_route": comparison.get("candidate_route_id"),
            "resource_ratio": comparison.get("resource_ratio"),
            "kernel_speedup": comparison.get("kernel_speedup"),
            "total_wall_speedup": comparison.get("total_wall_speedup"),
            "kernel_parallel_efficiency": comparison.get(
                "kernel_parallel_efficiency"
            ),
            "total_wall_parallel_efficiency": comparison.get(
                "total_wall_parallel_efficiency"
            ),
            "diagnostic_only": True,
        }
        for comparison in comparisons
    ]


def _accuracy_table_rows(
    summary: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    statistics = _mapping(summary.get("route_statistics"), "route_statistics")
    configuration = _mapping(manifest.get("configuration"), "configuration")
    policy_value = configuration.get("validation_policy", {})
    policy = (
        _mapping(policy_value, "validation_policy")
        if policy_value
        else {}
    )
    return [
        {
            "route": route_id,
            "max_abs_error": _mapping(statistics[route_id], route_id).get(
                "max_abs_error"
            ),
            "relative_l2_error": _mapping(statistics[route_id], route_id).get(
                "relative_l2_error"
            ),
            "norm_drift": _mapping(statistics[route_id], route_id).get("norm_drift"),
            "float32_atol": policy.get("float32_atol"),
            "float32_relative_l2_max": policy.get("float32_relative_l2_max"),
            "float32_norm_drift_max": policy.get("float32_norm_drift_max"),
        }
        for route_id in _ROUTE_IDS
    ]


def _plot_metadata(summary: Mapping[str, Any]) -> str:
    rank_paths = summary.get("rank_paths", ())
    rank = "rank1" if any("rank1" in str(path) for path in rank_paths) else "rank"
    return f"diagnostic_v1 | powersave | Stress18 | n=5 | {rank} | descriptive only"


def _plot_imports() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return matplotlib, plt


def _finish_plot(figure: Any, path: Path, metadata: str) -> None:
    from matplotlib import pyplot as plt

    figure.text(0.01, 0.01, metadata, fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=140, metadata={"Description": metadata})
    plt.close(figure)


def _runtime_plot(
    *,
    output: Path,
    summary: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    route_ids: Sequence[str],
    filename: str,
) -> None:
    _, plt = _plot_imports()
    statistics = _mapping(summary.get("route_statistics"), "route_statistics")
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    positions = list(range(len(route_ids)))
    for offset, (field, label, color) in enumerate(
        (("total_wall_s", "total wall", "#1565c0"), ("kernel_s", "kernel", "#ef6c00"))
    ):
        xs = [position + (offset - 0.5) * 0.16 for position in positions]
        for x, route_id in zip(xs, route_ids):
            values = [
                float(_mapping(row.get("measurement"), "measurement")[field])
                for row in _measurement_rows(samples, route_id)
            ]
            axis.scatter([x] * len(values), values, color=color, alpha=0.45, s=16)
            route_stats = _mapping(statistics[route_id], route_id)
            median_value = float(route_stats[f"median_{field}"])
            mad_value = float(route_stats[f"mad_{field}"])
            axis.errorbar(
                [x],
                [median_value],
                yerr=[mad_value],
                fmt="o",
                color=color,
                capsize=3,
                label=label if x == xs[0] else None,
            )
    axis.set_xticks(positions, [_PLOT_LABELS[route_id] for route_id in route_ids])
    axis.set_xlabel("tasklets per DPU" if "tasklet" in filename else "DPUs at T8")
    axis.set_ylabel("seconds")
    axis.set_title("Steady execution runtime")
    axis.legend()
    _finish_plot(figure, output / filename, _plot_metadata(summary))


def _speedup_plot(
    *, output: Path, summary: Mapping[str, Any], kind: str, filename: str
) -> None:
    _, plt = _plot_imports()
    comparisons = [
        comparison
        for comparison in summary["primary_comparisons"]
        if comparison["comparison_kind"] == kind
    ]
    resource = [1.0] + [float(row["resource_ratio"]) for row in comparisons]
    total = [1.0] + [float(row["total_wall_speedup"]) for row in comparisons]
    kernel = [1.0] + [float(row["kernel_speedup"]) for row in comparisons]
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    axis.plot(resource, total, "o-", label="total wall", color="#1565c0")
    axis.plot(resource, kernel, "o-", label="kernel", color="#ef6c00")
    axis.plot(resource, resource, "--", label="ideal linear", color="#616161")
    axis.set_xlabel("tasklets per DPU" if kind == "tasklet_scaling" else "DPUs")
    axis.set_ylabel("descriptive speedup")
    axis.set_title("Parallel speedup")
    axis.legend()
    _finish_plot(figure, output / filename, _plot_metadata(summary))


def _transfer_plot(
    *, output: Path, summary: Mapping[str, Any]
) -> None:
    _, plt = _plot_imports()
    statistics = _mapping(summary.get("route_statistics"), "route_statistics")
    labels = [_PLOT_LABELS[route_id] for route_id in _ROUTE_IDS]
    h2d = [statistics[route_id].get("median_h2d_bytes") or 0 for route_id in _ROUTE_IDS]
    d2h = [statistics[route_id].get("median_d2h_bytes") or 0 for route_id in _ROUTE_IDS]
    figure, axis = plt.subplots(figsize=(7.0, 4.4))
    positions = list(range(len(labels)))
    axis.bar(positions, h2d, label="H2D", color="#00897b")
    axis.bar(positions, d2h, bottom=h2d, label="D2H", color="#8e24aa")
    axis.set_xticks(positions, labels)
    axis.set_xlabel("route")
    axis.set_ylabel("bytes")
    axis.set_title("Transfer volume")
    axis.legend()
    _finish_plot(figure, output / "transfer_by_route.png", _plot_metadata(summary))


def _write_parallel_outputs(
    *,
    output_dir: Path,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "route_statistics.csv", _ROUTE_STAT_COLUMNS, _route_table_rows(summary))
    _write_csv(
        output_dir / "parallel_comparisons_descriptive.csv",
        _COMPARISON_COLUMNS,
        _comparison_table_rows(summary),
    )
    _write_csv(
        output_dir / "accuracy_summary.csv",
        _ACCURACY_COLUMNS,
        _accuracy_table_rows(summary, manifest),
    )
    _runtime_plot(
        output=output_dir,
        summary=summary,
        samples=samples,
        route_ids=_ROUTE_IDS[:4],
        filename="tasklet_runtime.png",
    )
    _speedup_plot(
        output=output_dir,
        summary=summary,
        kind="tasklet_scaling",
        filename="tasklet_speedup.png",
    )
    _runtime_plot(
        output=output_dir,
        summary=summary,
        samples=samples,
        route_ids=(_ROUTE_IDS[3], _ROUTE_IDS[4], _ROUTE_IDS[5]),
        filename="dpu_runtime.png",
    )
    _speedup_plot(
        output=output_dir,
        summary=summary,
        kind="dpu_scaling",
        filename="dpu_speedup.png",
    )
    _transfer_plot(output=output_dir, summary=summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args(argv)
    try:
        summary = inspect_artifacts(
            input_dir=args.input,
            summary_output=args.summary_output,
            output_dir=args.output_dir,
            expected_source_commit=args.expected_source_commit,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
