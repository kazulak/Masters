#!/usr/bin/env python3
"""Inspect and summarize the multi-circuit resource-sensitivity diagnostic."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import re
import subprocess
from statistics import median
from typing import Any

from quantum_bench.cli import _job, _plan_dag
from quantum_bench.evidence import canonical_json, load_artifacts, problem_id, tensor_network_structure_id
from quantum_bench.lowering import contraction_dag_hash
from quantum_bench.report import _TERMINAL_AUTHORITY_FIELDS, verify_artifacts
from quantum_bench.upmem.plan import UpmemTopology, collection_resource_admission, physical_plan_id, plan_upmem

try:
    from analyze_m7d_attribution import _sample_components
except ImportError:  # pragma: no cover - direct script execution puts scripts/ on sys.path
    _sample_components = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = ROOT / "configs" / "circuit_resource_sensitivity_selection.json"
ANALYSIS_VERSION = "circuit_resource_sensitivity_diagnostic_v1"
NUMERIC_POLICY = "split_complex_float32_v1"
TIMING_SCOPE = "steady_execution_v1"
ROUTE_SPECS = {
    "upmem_float32_1dpu_t1": (1, 1),
    "upmem_float32_1dpu_t4": (1, 4),
    "upmem_float32_1dpu_t8": (1, 8),
    "upmem_float32_1dpu_t12": (1, 12),
    "upmem_float32_2dpu_t8": (2, 8),
    "upmem_float32_3dpu_t8": (3, 8),
    "upmem_float32_4dpu_t8": (4, 8),
}
ROUTE_IDS = tuple(ROUTE_SPECS)
TASKLET_ROUTES = ROUTE_IDS[:4]
DPU_ROUTES = ROUTE_IDS[2:3] + ROUTE_IDS[4:]
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
EXPECTED_HOST = "safari-baguette1"
EXPECTED_RANK_PATH = "/dev/dpu_rank1"
DISJOINT_COMPONENT_FIELDS = ("preparation_s", "encode_s", "host_request_overhead_s", "native_request_overhead_s", "h2d_s", "kernel_s", "d2h_s", "assembly_s", "decode_s", "operation_other_s", "host_reduce_s", "coordinator_other_s", "accounting_residual_s")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _sha(value: object, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase hexadecimal identity")
    return value


def _source_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ValueError("cannot determine current source SHA")
    return _sha(result.stdout.strip(), "current source SHA", HEX40)


def _raw_mad(values: Sequence[float]) -> float:
    center = median(values)
    return float(median(abs(value - center) for value in values))


def _load_selection(path: Path, expected_source: str) -> Mapping[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "circuit_resource_sensitivity_selection_v1":
        raise ValueError("selection schema is not recognized")
    selected = selection.get("selected_case_ids")
    if selected != ["quantization_stress_18q_l2", "hs_18q_d1", "ghz_chain_18q"]:
        raise ValueError("selection is not the preregistered three-case matrix")
    if selection.get("selection_rule", {}).get("timing_used") is not False:
        raise ValueError("physical selection must not depend on timing")
    selection_source = _sha(selection.get("source_sha"), "selection source SHA", HEX40)
    lineage = subprocess.run(
        ["git", "merge-base", "--is-ancestor", selection_source, expected_source],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if lineage.returncode:
        raise ValueError("selection source is not an ancestor of the evidence source")
    for candidate in selection.get("candidates", ()):
        if candidate.get("candidate_id") in selected:
            _sha(candidate.get("tensor_network_structure_id"), "selection tensor-network structure ID", HEX64)
    return selection


def _validate_configuration(config: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
    collection = _mapping(config.get("collection"), "collection")
    expected = {
        "claim_policy": "diagnostic_v1",
        "warmup_blocks": 1,
        "measurement_blocks": 5,
        "session_policy": "fresh_session_per_attempt_v1",
        "block_cooldown_s": 0,
    }
    for field, value in expected.items():
        if collection.get(field) != value:
            raise ValueError(f"collection.{field} changed")
    selected = tuple(selection["selected_case_ids"])
    cases = _mapping(config.get("cases"), "cases")
    if set(cases) != set(selected):
        raise ValueError("case order or selection changed")
    plans = _mapping(config.get("plans"), "plans")
    greedy = _mapping(plans.get("greedy"), "plans.greedy")
    if tuple(plans) != ("greedy",) or greedy.get("planner") != {"engine": "opt_einsum", "mode": "greedy"} or greedy.get("slicing") is not None:
        raise ValueError("planner or slicing policy changed")
    routes = _mapping(config.get("routes"), "routes")
    if set(routes) != set(ROUTE_IDS):
        raise ValueError("route order changed")
    for route_id, (dpus, tasklets) in ROUTE_SPECS.items():
        route = _mapping(routes[route_id], route_id)
        options = _mapping(route.get("options"), f"{route_id}.options")
        if route.get("executor") != "upmem_physical" or route.get("numeric_policy") != NUMERIC_POLICY:
            raise ValueError(f"{route_id} is not the expected physical route")
        if (options.get("rank_count"), options.get("dpu_count"), options.get("tasklets_per_dpu")) != (1, dpus, tasklets):
            raise ValueError(f"{route_id} topology changed")
        if tuple(str(path) for path in options.get("rank_paths", ())) != (EXPECTED_RANK_PATH,):
            raise ValueError(f"{route_id} must have one rank path")
    expected_matrix = tuple(
        (case_id, "greedy", tuple(ROUTE_IDS)) for case_id in selected
    )
    actual_matrix = tuple(
        (
            item.get("case_id"),
            item.get("plan_id"),
            tuple(item.get("route_ids", ())),
        )
        for item in config.get("matrix", ())
    )
    if actual_matrix != expected_matrix:
        raise ValueError("case-route matrix changed")
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in selection.get("candidates", ())
        if candidate.get("candidate_id") in selected
    }
    if set(candidates) != set(selected):
        raise ValueError("selection is missing a selected candidate")
    for case_id in selected:
        circuit = _mapping(cases[case_id], f"cases.{case_id}")["circuit"]
        chosen = _mapping(candidates[case_id]["circuit"], f"selection.{case_id}.circuit")
        if (circuit.get("kind"), circuit.get("name"), dict(circuit.get("parameters", {}))) != (
            chosen.get("kind"), chosen.get("name"), dict(chosen.get("parameters", {}))
        ):
            raise ValueError(f"{case_id} differs from its preregistered circuit")


def _expected_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for case_id, case in config["cases"].items():
        job = _job(case)
        network, _, dag, _ = _plan_dag(job, config["plans"]["greedy"])
        routes: dict[str, Any] = {}
        for route_id, (dpus, tasklets) in ROUTE_SPECS.items():
            route = config["routes"][route_id]
            plan = plan_upmem(
                dag,
                numeric_policy=route["numeric_policy"],
                topology=UpmemTopology(dpu_count=dpus, rank_count=1, tasklets_per_dpu=tasklets),
            )
            routes[route_id] = {
                "physical_plan_id": physical_plan_id(plan),
                "kernel_policy": plan.kernel_policy,
                "admission": collection_resource_admission(plan),
            }
        result[case_id] = {
            "problem_id": problem_id(job),
            "tensor_network_structure_id": tensor_network_structure_id(network),
            "logical_plan_id": contraction_dag_hash(dag),
            "routes": routes,
        }
    return result


def _joined_facts(sample: Mapping[str, Any], sessions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    facts = dict(_mapping(sample.get("backend_facts"), "sample backend_facts"))
    session = sessions.get(str(sample.get("session_instance_id")))
    terminal = session.get("terminal_backend_facts") if session else None
    if isinstance(terminal, Mapping):
        conflicts = sorted(
            field
            for field in _TERMINAL_AUTHORITY_FIELDS
            if field in facts and field in terminal and facts[field] != terminal[field]
        )
        if conflicts:
            raise ValueError(f"terminal physical facts conflict for {', '.join(conflicts)}")
        for field, value in terminal.items():
            facts.setdefault(field, value)
    return facts


def _validate_terminal_facts(
    session: Mapping[str, Any], dpus: int, tasklets: int
) -> None:
    terminal = _mapping(session.get("terminal_backend_facts"), "terminal backend facts")
    expected = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "requested_dpu_count": dpus,
        "allocated_dpu_count": dpus,
        "observed_dpu_count": dpus,
        "observed_tasklets_per_dpu": tasklets,
        "startup_requested_dpu_count": dpus,
        "startup_allocated_dpu_count": dpus,
        "startup_requested_tasklets_per_dpu": tasklets,
        "allocation_verified": True,
        "hardware_allocation_verified": True,
        "binary_identity_verified": True,
        "native_identity_verified": True,
        "hardware_release_verified": True,
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise ValueError(f"terminal {field} does not match the physical contract")


def _validate_session_bijection(
    samples: Sequence[Mapping[str, Any]],
    sessions_by_id: Mapping[str, Mapping[str, Any]],
    expected_count: int,
) -> None:
    sample_session_ids = [sample.get("session_instance_id") for sample in samples]
    if len(sample_session_ids) != expected_count or any(not isinstance(session_id, str) for session_id in sample_session_ids):
        raise ValueError("every sample must reference a session")
    counts = Counter(sample_session_ids)
    if set(counts) != set(sessions_by_id) or any(count != 1 for count in counts.values()):
        raise ValueError("sample/session references are not a bijection")
    for sample in samples:
        session = sessions_by_id[sample["session_instance_id"]]
        if session.get("case_id") != sample.get("case_id") or session.get("route_id") != sample.get("route_id"):
            raise ValueError("sample/session case or route identity differs")


def _require_true(facts: Mapping[str, Any], field: str) -> None:
    if facts.get(field) is not True:
        raise ValueError(f"physical evidence requires {field}=true")


def _validate_sample(
    sample: Mapping[str, Any],
    case_id: str,
    route_id: str,
    expected: Mapping[str, Any],
    sessions: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], Mapping[str, float]]:
    if sample.get("status") != "success" or sample.get("case_id") != case_id or sample.get("route_id") != route_id or sample.get("plan_id") != "greedy":
        raise ValueError("sample has an unexpected case, route, plan, or status")
    if (sample.get("attempt_kind"), sample.get("block_id")) not in (("warmup", 0), *(('measurement', block) for block in range(1, 6))):
        raise ValueError("sample is outside the one-warmup/five-measurement schedule")
    session_id = sample.get("session_instance_id")
    if not isinstance(session_id, str) or session_id not in sessions:
        raise ValueError("sample session is missing")
    if list(sample.get("observed_affinity", ())) != [0]:
        raise ValueError("sample observed affinity is not [0]")
    measurement = _mapping(sample.get("measurement"), "sample measurement")
    if measurement.get("scope_id") != TIMING_SCOPE:
        raise ValueError("sample timing scope changed")
    identities = _mapping(sample.get("identities"), "sample identities")
    for field in ("problem_id", "tensor_network_structure_id", "logical_plan_id"):
        if identities.get(field) != expected[field]:
            raise ValueError(f"sample {field} does not match planned identity")
    if identities.get("physical_plan_id") != expected["routes"][route_id]["physical_plan_id"]:
        raise ValueError("sample physical plan identity changed")
    numeric = _mapping(sample.get("numeric_facts"), "sample numeric facts")
    if numeric.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError("sample numeric policy changed")
    validation = _mapping(sample.get("validation"), "sample validation")
    for field in ("policy_reference_applicable", "policy_reference_passed", "full_precision_threshold_applicable", "full_precision_passed", "accuracy_qualified"):
        _require_true(validation, field)
    _sha(sample.get("output_sha256"), "sample output hash", HEX64)
    facts = _joined_facts(sample, sessions)
    if facts.get("kernel_policy") != expected["routes"][route_id]["kernel_policy"]:
        raise ValueError("sample kernel policy changed")
    dpus, tasklets = ROUTE_SPECS[route_id]
    _validate_terminal_facts(sessions[session_id], dpus, tasklets)
    for field, value in (("requested_dpus", dpus), ("allocated_dpus", dpus), ("active_dpus", dpus), ("execution_active_dpu_count", dpus), ("tasklets_per_dpu", tasklets), ("startup_requested_dpu_count", dpus), ("startup_allocated_dpu_count", dpus), ("startup_requested_tasklets_per_dpu", tasklets), ("rank_count", 1)):
        if facts.get(field) != value:
            raise ValueError(f"sample {field} does not match {route_id}")
    for field in ("physical_target_verified", "hardware_kernel_executed", "startup_resource_admission_passed", "execution_resource_admission_passed", "binary_identity_verified", "native_identity_verified"):
        _require_true(facts, field)
    if facts.get("cpu_fallback_used") is not False or facts.get("simulator_kernel_executed") is not False:
        raise ValueError("physical sample used fallback or simulator execution")
    if facts.get("target_observed") != "physical_hardware":
        raise ValueError("sample target is not physical hardware")
    if not (facts.get("release_verified") is True or facts.get("hardware_release_verified") is True):
        raise ValueError("hardware release was not verified")
    if _sample_components is None:
        raise ValueError("request timing attribution helper is unavailable")
    components = _sample_components(sample)
    if components is None:
        raise ValueError("sample lacks operation timing attribution")
    for field in ("host_binary_sha256", "dpu_binary_sha256", "initialization_binary_sha256"):
        _sha(facts.get(field), field, HEX64)
    return facts, components


def _route_statistics(
    samples: Sequence[Mapping[str, Any]], facts: Sequence[Mapping[str, Any]], components: Sequence[Mapping[str, float]]
) -> dict[str, Any]:
    measurement_rows = [
        (sample, fact, component)
        for sample, fact, component in zip(samples, facts, components)
        if sample["attempt_kind"] == "measurement"
    ]
    if not measurement_rows:
        raise ValueError("route has no measurement samples")
    measurements = [row[0] for row in measurement_rows]
    measurement_facts = [row[1] for row in measurement_rows]
    measurement_components = [row[2] for row in measurement_rows]
    measurements_data = [_mapping(sample["measurement"], "measurement") for sample in measurements]
    totals = [_number(row["total_wall_s"], "total_wall_s") for row in measurements_data]
    kernels = [_number(row["kernel_s"], "kernel_s") for row in measurements_data]

    def med(values: Sequence[float]) -> float:
        return float(median(values))

    component_medians = {field: med([row[field] for row in measurement_components]) for field in DISJOINT_COMPONENT_FIELDS}
    component_shares = {field: med([row[field] / row["total_wall_s"] for row in measurement_components]) for field in component_medians}
    return {
        "measurement_count": len(measurements),
        "median_total_wall_s": med(totals),
        "raw_mad_total_wall_s": _raw_mad(totals),
        "median_kernel_s": med(kernels),
        "raw_mad_kernel_s": _raw_mad(kernels),
        "median_h2d_s": med([_number(row["h2d_s"], "h2d_s") for row in measurements_data]),
        "median_d2h_s": med([_number(row["d2h_s"], "d2h_s") for row in measurements_data]),
        "median_h2d_bytes": int(med([_number(row["h2d_bytes"], "h2d_bytes") for row in measurements_data])),
        "median_d2h_bytes": int(med([_number(row["d2h_bytes"], "d2h_bytes") for row in measurements_data])),
        "max_abs_error": max(_number(sample["validation"]["max_abs_error"], "max_abs_error") for sample in measurements),
        "max_relative_l2_error": max(_number(sample["validation"]["relative_l2_error"], "relative_l2_error") for sample in measurements),
        "max_norm_drift": max(_number(sample["validation"]["norm_drift"], "norm_drift") for sample in measurements),
        "tasklet_utilization": med([_number(f["arithmetic_weighted_tasklet_utilization"], "tasklet utilization") for f in measurement_facts]),
        "dpu_utilization": med([_number(f["arithmetic_weighted_dpu_slot_utilization"], "DPU utilization") for f in measurement_facts]),
        "dominant_wave_utilization": med([_number(f["dominant_work_wave_utilization"], "wave utilization") for f in measurement_facts]),
        "underfilled_diagnostic_count": sum(f.get("collection_resource_admission_passed") is False or f.get("dominant_work_wave_tasklet_row_sufficiency_passed") is False for f in measurement_facts),
        "component_medians_s": component_medians,
        "component_median_shares": component_shares,
        "request_build_medians_s": {field: med([row[field] for row in measurement_components]) for field in ("work_unit_materialization_s", "payload_record_staging_s", "manifest_sidecar_staging_s", "artifact_build_residual_s", "request_build_residual_s")},
        "request_build_parent_median_s": med([row["request_build_parent_s"] for row in measurement_components]),
        "request_build_parent_share": med([row["request_build_parent_s"] / row["total_wall_s"] for row in measurement_components]),
    }


def _comparisons(stats: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for kind, routes, field in (("tasklet", TASKLET_ROUTES, "tasklets_per_dpu"), ("dpu", DPU_ROUTES, "dpu_count")):
        base = stats[routes[0]]
        for route_id in routes[1:]:
            candidate = stats[route_id]
            ratio = float(candidate[field]) / float(base[field])
            kernel = base["median_kernel_s"] / candidate["median_kernel_s"]
            total = base["median_total_wall_s"] / candidate["median_total_wall_s"]
            result.append({"comparison_kind": kind, "baseline_route": routes[0], "candidate_route": route_id, "resource_ratio": ratio, "kernel_speedup": kernel, "total_wall_speedup": total, "kernel_parallel_efficiency": kernel / ratio, "total_wall_parallel_efficiency": total / ratio, "diagnostic_only": True})
    return result


def _environment(manifest: Mapping[str, Any]) -> dict[str, Any]:
    configuration = _mapping(manifest["configuration"], "manifest configuration")
    environment = _mapping(configuration["environment"], "environment")
    if list(environment.get("affinity", ())) != [0] or list(environment.get("selected_cpu_ids", ())) != [0]:
        raise ValueError("diagnostic CPU affinity is not [0]")
    governors = _mapping(environment.get("observed_cpu_governors"), "observed governors")
    if environment.get("host") != EXPECTED_HOST:
        raise ValueError("diagnostic host changed")
    if "0" not in governors or not isinstance(governors["0"], str) or not governors["0"]:
        raise ValueError("CPU 0 governor was not recorded")
    rank_paths = tuple(str(path) for path in environment.get("requested_rank_paths", ()))
    if rank_paths != (EXPECTED_RANK_PATH,):
        raise ValueError("diagnostic rank path changed")
    if dict(environment.get("thread_environment", {})) != THREAD_ENVIRONMENT:
        raise ValueError("BLAS/OpenMP environment is not single-threaded")
    return {"host": environment["host"], "affinity": [0], "observed_cpu_governors": dict(governors), "upmem_sdk_version": environment.get("upmem_sdk_version"), "numpy_version": environment.get("numpy_version"), "blas": environment.get("blas"), "thread_environment": dict(environment["thread_environment"]), "rank_paths": list(rank_paths) }


def _descriptor_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in selection["candidates"]:
        if candidate["candidate_id"] not in selection["selected_case_ids"]:
            continue
        circuit = candidate["circuit"]["descriptors"]
        graph = circuit["interaction_graph"]
        network = candidate["tensor_network"]
        rows.append({"case_id": candidate["candidate_id"], "circuit_name": candidate["circuit"]["name"], "n_qubits": circuit["n_qubits"], "depth": circuit["true_deterministic_depth"], "gate_count": circuit["gate_count"], "interaction_edge_count": graph["edge_count"], "interaction_density": graph["density"], "interaction_max_degree": graph["max_degree"], "tensor_count": network["tensor_count"], "index_count": network["index_count"], "contraction_count": network["contraction_count"], "planner_flops_estimate": network["planner_flops_estimate"], "largest_non_final_intermediate_elements": network["largest_non_final_intermediate_elements"], "largest_non_final_intermediate_bytes_complex128": network["largest_non_final_intermediate_bytes_complex128"], "max_intermediate_rank": network["max_intermediate_rank"], "logical_plan_id": network["logical_plan_id"]})
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plots(output_dir: Path, summary: Mapping[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    cases = list(summary["selected_case_ids"])
    metadata = "diagnostic_v1 | powersave/recorded governor | n=5 | descriptive only"
    for filename, routes, field, title in (("tasklet_scaling_by_circuit.png", TASKLET_ROUTES, "tasklets_per_dpu", "Tasklet scaling by circuit"), ("dpu_scaling_by_circuit.png", DPU_ROUTES, "dpu_count", "DPU scaling by circuit")):
        figure, axes = plt.subplots(len(cases), 1, figsize=(7.2, 2.2 * len(cases)), squeeze=False)
        for axis, case_id in zip(axes[:, 0], cases):
            stats = summary["cases"][case_id]["route_statistics"]
            base = stats[routes[0]]
            xs = [stats[route][field] for route in routes]
            kernel = [base["median_kernel_s"] / stats[route]["median_kernel_s"] for route in routes]
            total = [base["median_total_wall_s"] / stats[route]["median_total_wall_s"] for route in routes]
            axis.plot(xs, kernel, "o-", label="kernel")
            axis.plot(xs, total, "s-", label="total wall")
            axis.plot(xs, [x / xs[0] for x in xs], "--", color="0.5", label="ideal")
            axis.set_title(case_id)
            axis.set_xlabel(field.replace("_", " "))
            axis.set_ylabel("descriptive speedup")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle(title)
        figure.text(0.01, 0.01, metadata, fontsize=8)
        figure.tight_layout(rect=(0, 0.04, 1, 0.96))
        figure.savefig(output_dir / filename, dpi=140, metadata={"Description": metadata})
        plt.close(figure)
    figure, axes = plt.subplots(len(cases), 1, figsize=(8, 2.5 * len(cases)), squeeze=False)
    components = DISJOINT_COMPONENT_FIELDS
    labels = {"preparation_s": "preparation", "encode_s": "encode", "host_request_overhead_s": "host request", "native_request_overhead_s": "native request", "h2d_s": "H2D", "kernel_s": "kernel", "d2h_s": "D2H", "assembly_s": "assembly", "decode_s": "decode", "operation_other_s": "operation other", "host_reduce_s": "host reduce", "coordinator_other_s": "coordinator other", "accounting_residual_s": "residual"}
    for axis, case_id in zip(axes[:, 0], cases):
        stats = summary["cases"][case_id]["route_statistics"]
        bottom = [0.0] * len(ROUTE_IDS)
        for component in components:
            values = [stats[route]["component_medians_s"].get(component, 0.0) for route in ROUTE_IDS]
            axis.bar(range(len(ROUTE_IDS)), values, bottom=bottom, label=labels[component])
            bottom = [left + value for left, value in zip(bottom, values)]
        axis.set_title(case_id)
        axis.set_xticks(range(len(ROUTE_IDS)), [route.replace("upmem_float32_", "") for route in ROUTE_IDS], rotation=25, ha="right")
        axis.set_ylabel("seconds")
    axes[0, 0].legend(ncol=4, fontsize=8)
    figure.suptitle("Timing composition by circuit")
    figure.text(0.01, 0.01, metadata, fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output_dir / "timing_composition_by_circuit.png", dpi=140, metadata={"Description": metadata})
    plt.close(figure)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = ["# Circuit-Structure and Resource-Sensitivity Diagnostic", "", f"Source: `{summary['source_commit']}`", f"Experiment: `{summary['experiment_id']}`; run: `{summary['run_id']}`", f"Environment: `{summary['environment']['host']}`, CPU `{summary['environment']['affinity']}`, governor `{summary['environment']['observed_cpu_governors'].get('0')}`", "", "This is a diagnostic-only within-circuit comparison. Each route has one warmup and five measured blocks; values are not pooled across circuits.", "", "| Case | Gates | Contractions | FLOP estimate | Non-final peak |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in summary["circuit_descriptors"]:
        lines.append(f"| {row['case_id']} | {row['gate_count']} | {row['contraction_count']} | {row['planner_flops_estimate']} | {row['largest_non_final_intermediate_elements']} |")
    lines.extend(["", "| Case | Kind | Baseline | Candidate | Kernel speedup | Total-wall speedup |", "| --- | --- | --- | --- | ---: | ---: |"])
    for case_id in summary["selected_case_ids"]:
        for row in summary["cases"][case_id]["comparisons"]:
            lines.append(f"| {case_id} | {row['comparison_kind']} | {row['baseline_route']} | {row['candidate_route']} | {row['kernel_speedup']:.4f} | {row['total_wall_speedup']:.4f} |")
    lines.extend(["", "The result is diagnostic_v1 and does not support final performance, cross-circuit denominators, multi-rank claims, or machine-independent acceleration.", ""])
    return "\n".join(lines)


def inspect(*, input_dir: Path, output_dir: Path, selection_path: Path, expected_source_commit: str | None = None) -> Mapping[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("analysis output directory must be absent or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    verification = verify_artifacts(input_dir)
    manifest, samples, sessions = load_artifacts(input_dir)
    source_commit = _sha(manifest.get("source_commit"), "evidence source SHA", HEX40)
    expected_source = expected_source_commit or _source_sha()
    _sha(expected_source, "expected source SHA", HEX40)
    selection = _load_selection(selection_path, expected_source)
    expected_count = len(selection["selected_case_ids"]) * len(ROUTE_IDS) * 6
    if verification.get("status") != "completed":
        raise ValueError("evidence is not completed")
    for field, value in {"sample_count": expected_count, "session_count": expected_count, "success_count": expected_count, "failed_count": 0, "unsupported_count": 0}.items():
        if verification.get(field) != value:
            raise ValueError(f"evidence {field} is {verification.get(field)!r}, expected {value}")
    if source_commit != expected_source:
        raise ValueError("evidence source does not match expected source")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("evidence source worktree was dirty")
    config = _mapping(manifest["configuration"], "manifest configuration")
    experiment = _mapping(config.get("experiment"), "embedded experiment")
    _validate_configuration(experiment, selection)
    environment = _environment(manifest)
    expected_contract = _expected_contract(experiment)
    sessions_by_id = {str(session.get("session_instance_id")): session for session in sessions}
    if len(sessions_by_id) != expected_count:
        raise ValueError("fresh-session evidence does not have one unique session per sample")
    _validate_session_bijection(samples, sessions_by_id, expected_count)
    expected_keys = {
        (case_id, route_id, kind, block)
        for case_id in selection["selected_case_ids"]
        for route_id in ROUTE_IDS
        for kind, block in (("warmup", 0), *(('measurement', number) for number in range(1, 6)))
    }
    seen: set[tuple[str, str, str, int]] = set()
    by_route: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    facts_by_route: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    components_by_route: dict[tuple[str, str], list[Mapping[str, float]]] = {}
    binaries: dict[int, tuple[str, str, str]] = {}
    kernel_policies: set[str] = set()
    for sample in samples:
        case_id = str(sample.get("case_id"))
        route_id = str(sample.get("route_id"))
        key = (case_id, route_id, str(sample.get("attempt_kind")), int(sample.get("block_id")))
        if key in seen:
            raise ValueError("duplicate case-route-block sample")
        seen.add(key)
        if case_id not in expected_contract or route_id not in ROUTE_SPECS:
            raise ValueError("unexpected case or route in evidence")
        facts, components = _validate_sample(sample, case_id, route_id, expected_contract[case_id], sessions_by_id)
        by_route.setdefault((case_id, route_id), []).append(sample)
        facts_by_route.setdefault((case_id, route_id), []).append(facts)
        components_by_route.setdefault((case_id, route_id), []).append(components)
        kernel_policies.add(str(facts["kernel_policy"]))
        tasklets = ROUTE_SPECS[route_id][1]
        hashes = tuple(facts[field] for field in ("host_binary_sha256", "dpu_binary_sha256", "initialization_binary_sha256"))
        if tasklets in binaries and binaries[tasklets] != hashes:
            raise ValueError(f"binary identity drift for T{tasklets}")
        binaries[tasklets] = hashes
    if seen != expected_keys:
        raise ValueError("case-route-block matrix is incomplete or contains extras")
    if len(kernel_policies) != 1:
        raise ValueError("kernel policy drifted across the run")
    cases: dict[str, Any] = {}
    for case_id in selection["selected_case_ids"]:
        stats: dict[str, Any] = {}
        for route_id in ROUTE_IDS:
            values = _route_statistics(by_route[(case_id, route_id)], facts_by_route[(case_id, route_id)], components_by_route[(case_id, route_id)])
            values["dpu_count"], values["tasklets_per_dpu"] = ROUTE_SPECS[route_id]
            stats[route_id] = values
        cases[case_id] = {"route_statistics": stats, "comparisons": _comparisons(stats), "underfilled_measurement_count": sum(value["underfilled_diagnostic_count"] for value in stats.values())}
    summary: dict[str, Any] = {
        "analysis_version": ANALYSIS_VERSION,
        "source_commit": source_commit,
        "experiment_id": manifest["experiment_id"],
        "run_id": manifest["run_id"],
        "selected_case_ids": list(selection["selected_case_ids"]),
        "route_ids": list(ROUTE_IDS),
        "block_ids": list(range(6)),
        "measurement_count_per_route": 5,
        "sample_count": expected_count,
        "session_count": expected_count,
        "verification": _plain(verification),
        "environment": environment,
        "claim_policy": "diagnostic_v1",
        "claim_eligible": False,
        "claim_ineligibility_reason": "diagnostic_claim_policy",
        "kernel_policy": next(iter(kernel_policies)),
        "binary_hashes_by_tasklets": {str(tasklets): list(values) for tasklets, values in sorted(binaries.items())},
        "circuit_descriptors": _descriptor_rows(selection),
        "cases": cases,
        "gate_passed": True,
    }
    (output_dir / "circuit_resource_sensitivity_summary.json").write_text(json.dumps(_plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    route_columns = ("case_id", "route", "dpu_count", "tasklets_per_dpu", "measurement_count", "median_total_wall_s", "raw_mad_total_wall_s", "median_kernel_s", "raw_mad_kernel_s", "median_h2d_s", "median_d2h_s", "median_h2d_bytes", "median_d2h_bytes", "max_abs_error", "max_relative_l2_error", "max_norm_drift", "tasklet_utilization", "dpu_utilization", "dominant_wave_utilization", "underfilled_diagnostic_count")
    route_rows = [{"case_id": case_id, "route": route_id, **cases[case_id]["route_statistics"][route_id]} for case_id in summary["selected_case_ids"] for route_id in ROUTE_IDS]
    _write_csv(output_dir / "route_statistics.csv", route_columns, route_rows)
    comparison_columns = ("case_id", "comparison_kind", "baseline_route", "candidate_route", "resource_ratio", "kernel_speedup", "total_wall_speedup", "kernel_parallel_efficiency", "total_wall_parallel_efficiency", "diagnostic_only")
    comparison_rows = [{"case_id": case_id, **row} for case_id in summary["selected_case_ids"] for row in cases[case_id]["comparisons"]]
    _write_csv(output_dir / "within_circuit_scaling.csv", comparison_columns, comparison_rows)
    descriptor_columns = tuple(summary["circuit_descriptors"][0])
    _write_csv(output_dir / "circuit_descriptors.csv", descriptor_columns, summary["circuit_descriptors"])
    _plots(output_dir, summary)
    (output_dir / "circuit_resource_sensitivity.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args(argv)
    try:
        summary = inspect(input_dir=args.input.resolve(), output_dir=args.output_dir.resolve(), selection_path=args.selection.resolve(), expected_source_commit=args.expected_source_commit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "inspected", "output_dir": str(args.output_dir.resolve()), "sample_count": summary["sample_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
