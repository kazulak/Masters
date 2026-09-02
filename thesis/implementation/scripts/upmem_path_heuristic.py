#!/usr/bin/env python3
"""Generate and fit the finite UPMEM-aware path-heuristic dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from collections.abc import Mapping
import hashlib
from importlib import metadata
import json
import math
import multiprocessing
from pathlib import Path
import queue as queue_module
import subprocess
import time
from typing import Any

from quantum_bench.circuits import builtin_circuit
from quantum_bench.evidence import load_artifacts, problem_id, tensor_network_structure_id
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import ContractNode, make_simulation_job
from quantum_bench.planning import plan_cotengra, plan_opt_einsum
from quantum_bench.upmem.path_heuristic import (
    COST_MODEL_ID,
    ConventionalPathFeatures,
    FeatureModelDecision,
    PathCandidate,
    RawFeatureVector,
    RuntimeMeasurement,
    TrainingCell,
    WeightFitResult,
    WeightVector,
    choose_feature_model,
    equal_model_weights,
    explain_score,
    extract_conventional_features,
    extract_plan_features,
    feature_dependency_metadata,
    fit_weights,
    normalize_features,
    path_id,
    score_features,
    select_calibration_candidates,
    select_best_candidate,
)
from quantum_bench.upmem.plan import (
    UpmemTopology,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
    _canonical_dimensions,
)
from quantum_bench.upmem.tiling import _choose_tile_shape, tile_limits_for_numeric_mode


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "upmem_path_heuristic_v1.json"
NUMERIC_POLICY = "split_complex_float32_v1"
SCHEMA_VERSION = "upmem_path_candidate_dataset_v1"
FEATURE_COLUMNS = (
    "circuit_id", "split", "candidate_path_id", "source_kind", "source_seed",
    "is_greedy", "topology_id", "feasible", "infeasibility_reason",
    "logical_plan_id", "physical_plan_id", "flops", "macs",
    "peak_intermediate_elements", "peak_intermediate_bytes",
    "total_intermediate_writes", "maximum_intermediate_rank",
    "contraction_count", "B_host_dpu", "B_mram_wram", "I_dpu", "N_sync",
    "E_num", "P_wram", "h2d_bytes", "d2h_bytes", "work_unit_count",
    "wave_count", "packed_operation_count", "dpu_launch_count",
    "host_reduce_count", "barrier_events", "partial_wave_count",
    "tasklet_utilization", "dpu_utilization", "host_memory_estimate_bytes",
    "semantic_identity_expansion_units",
)
CALIBRATION_COLUMNS = (
    "split", "attempt_type", "cell_id", "circuit_id", "topology_id",
    "candidate_path_id", "plan_id", "route_id", "block", "sample_index",
    "order_index", "sample_id", "session_instance_id", "experiment_id", "run_id",
    "source_sha", "candidate_generation_source_sha", "physical_execution_source_sha",
    "candidate_set_sha256", "calibration_set_sha256", "problem_id",
    "tensor_network_structure_id", "logical_plan_id", "physical_plan_id",
    "executable_id", "validation_policy_id", "output_sha256", "status",
    "validation", "fallback", "max_abs_error", "relative_l2_error",
    "norm_drift", "phase_aligned_max_abs_error", "full_precision_passed",
    "policy_reference_passed",
    "timing_scope", "total_wall_s", "session_open_s", "session_close_s",
    "session_inclusive_s", "kernel_s", "h2d_s", "d2h_s", "h2d_bytes", "d2h_bytes",
    "preparation_s", "planning_s", "lowering_s", "mapping_s", "slicing_s",
    "host_reduce_s", "rank_work_s", "request_build_s", "request_wave_s",
    "request_artifact_build_s", "payload_record_staging_s",
    "request_work_unit_materialization_s", "request_payload_materialization_s",
    "request_payload_hashing_s", "request_payload_file_write_s",
    "request_manifest_sidecar_staging_s", "request_build_residual_s",
    "request_payload_record_count", "request_payload_bytes_staged",
    "request_payload_bytes_hashed", "request_payload_files_created",
    "requested_dpus", "allocated_dpus", "active_dpus", "tasklets_per_dpu",
    "rank_count", "active_rank_count", "target_observed", "request_transport",
    "collection_resource_admission_passed", "execution_resource_admission_passed",
    "startup_resource_admission_passed", "physical_target_verified",
    "hardware_kernel_executed", "simulator_kernel_executed", "cpu_fallback_used",
    "binary_identity_verified", "native_identity_verified", "hardware_release_verified",
    "tasklet_utilization", "dpu_utilization", "dominant_wave_utilization",
    "total_wave_count", "fully_populated_wave_count", "active_dpu_ids_json",
    "active_rank_indices_json", "requested_rank_paths_json",
)
CALIBRATION_SCHEMA_VERSION = "upmem_path_runtime_calibration_v1"
CALIBRATION_TIMING_SCOPE = "steady_execution_v1"
CALIBRATION_TRANSPORT = "packed_operation_v1"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    value = result.stdout.strip()
    if result.returncode or len(value) != 40:
        raise ValueError("cannot determine source SHA")
    return value


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "upmem_path_heuristic_preregistration_v1":
        raise ValueError("unrecognized path-heuristic preregistration")
    if record.get("numeric_policy") != NUMERIC_POLICY:
        raise ValueError("physical v1 requires split-complex float32")
    return record


def _topologies(config: dict[str, Any]) -> tuple[tuple[str, UpmemTopology], ...]:
    return tuple(
        (
            str(item["topology_id"]),
            UpmemTopology(
                dpu_count=int(item["dpu_count"]),
                rank_count=int(item["rank_count"]),
                tasklets_per_dpu=int(item["tasklets_per_dpu"]),
            ),
        )
        for item in config["topologies"]
    )


def _host_memory_estimate(inputs: dict[str, Any], dag: Any) -> int:
    input_bytes = sum(int(value.nbytes) for value in inputs.values())
    # Logical intermediates are complex128 before the physical float32 lowering.
    outputs = [int(_product(node.output.shape)) * 16 for node in dag.nodes]
    # Bound one packed float32 transport copy and one native input copy in
    # addition to the logical tensors retained by the host execution shell.
    packed_transport_bytes = input_bytes // 2 + sum(outputs) // 2
    native_copy_bytes = packed_transport_bytes
    return input_bytes + sum(outputs) + packed_transport_bytes + native_copy_bytes


def _estimated_work_unit_count(dag: Any) -> int:
    limits = tile_limits_for_numeric_mode("float32")
    result = 0
    for node in dag.nodes:
        if not isinstance(node, ContractNode):
            continue
        batch, m_size, n_size, k_size = _canonical_dimensions(node)
        tile_m, tile_k, tile_n = _choose_tile_shape(
            m_size, k_size, n_size, limits
        )
        result += (
            batch
            * ((m_size + tile_m - 1) // tile_m)
            * ((n_size + tile_n - 1) // tile_n)
            * ((k_size + tile_k - 1) // tile_k)
        )
    return result


def _semantic_identity_expansion_units(dag: Any, *, stop_after: int) -> int:
    """Bound recursive escaped-subtree expansion in the frozen DAG identity."""

    depths: dict[str, int] = {}
    total = 0
    for node in dag.nodes:
        depth = 1 + max((depths[item] for item in node.dependencies), default=0)
        depths[node.node_id] = depth
        total += 1 << depth
        if total > stop_after:
            return total
    return total


def _product(values: Any) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _cotengra_trial_worker(
    circuit_name: str,
    circuit_parameters: dict[str, Any],
    objective: str,
    methods: str,
    seed: int,
    queue: Any,
) -> None:
    try:
        circuit = builtin_circuit(circuit_name, circuit_parameters)
        network, _ = lower_tensor_network(make_simulation_job(circuit))
        path, provenance = plan_cotengra(
            network,
            objective=objective,
            methods=methods,
            max_repeats=1,
            seed=seed,
        )
        queue.put((path, provenance["planner_config_hash"], None))
    except BaseException as exc:  # child must return a finite failure record
        queue.put((None, None, f"{type(exc).__name__}:{exc}"))


def _isolated_cotengra_trial(
    *,
    circuit_name: str,
    circuit_parameters: dict[str, Any],
    objective: str,
    methods: str,
    seed: int,
) -> tuple[Any, dict[str, str]]:
    # The supported UPMEM hosts are Linux. Fork keeps imports and immutable
    # circuit metadata copy-on-write while still releasing each optimizer tree
    # when its one-trial child exits.
    context = multiprocessing.get_context("fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_cotengra_trial_worker,
        args=(circuit_name, circuit_parameters, objective, methods, seed, queue),
    )
    process.start()
    deadline = time.monotonic() + 300.0
    result = None
    while result is None:
        try:
            result = queue.get_nowait()
        except queue_module.Empty:
            if not process.is_alive():
                process.join()
                raise RuntimeError(
                    f"isolated cotengra trial {seed} returned no result: {process.exitcode}"
                )
            if time.monotonic() >= deadline:
                process.kill()
                process.join()
                raise RuntimeError(f"isolated cotengra trial {seed} timed out")
            time.sleep(0.01)
    path, config_hash, error = result
    process.join()
    queue.close()
    if process.exitcode != 0 or error is not None:
        raise RuntimeError(
            f"isolated cotengra trial {seed} failed: {error or process.exitcode}"
        )
    return path, {"planner_config_hash": str(config_hash)}


def _candidate_paths(
    network: Any,
    circuit_id: str,
    config: dict[str, Any],
    *,
    isolate_trials: bool = False,
    circuit_definition: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    generation = config["candidate_generation"]
    started = time.perf_counter()
    greedy_path, greedy_provenance = plan_opt_einsum(network, optimize="greedy")
    raw = [
        {
            "path": greedy_path,
            "source_kind": "opt_einsum_greedy",
            "source_seed": None,
            "is_greedy": True,
            "planner_config_hash": greedy_provenance["planner_config_hash"],
        }
    ]
    search_started = time.perf_counter()
    for trial in range(int(generation["one_trial_searches"])):
        seed = int(generation["master_seed"]) + trial
        if isolate_trials:
            if circuit_definition is None:
                raise ValueError("isolated trials require a circuit definition")
            candidate_path, provenance = _isolated_cotengra_trial(
                circuit_name=str(circuit_definition["name"]),
                circuit_parameters=dict(circuit_definition["parameters"]),
                objective=str(generation["cotengra_objective"]),
                methods=str(generation["cotengra_method"]),
                seed=seed,
            )
        else:
            candidate_path, provenance = plan_cotengra(
                network,
                objective=str(generation["cotengra_objective"]),
                methods=str(generation["cotengra_method"]),
                max_repeats=1,
                seed=seed,
            )
        raw.append(
            {
                "path": candidate_path,
                "source_kind": "cotengra_one_trial",
                "source_seed": seed,
                "is_greedy": False,
                "planner_config_hash": provenance["planner_config_hash"],
            }
        )
    search_elapsed = time.perf_counter() - search_started
    unique: dict[str, dict[str, Any]] = {}
    for item in raw:
        identifier = path_id(item["path"], circuit_id=circuit_id)
        item = {**item, "candidate_path_id": identifier}
        previous = unique.get(identifier)
        if previous is None or (
            previous["source_seed"] is not None
            and (item["source_seed"] is None or item["source_seed"] < previous["source_seed"])
        ):
            unique[identifier] = item
        elif item["is_greedy"]:
            previous["is_greedy"] = True
    candidates = sorted(unique.values(), key=lambda item: (not item["is_greedy"], item["candidate_path_id"]))
    return candidates, {
        "candidate_generation_s": time.perf_counter() - started,
        "cotengra_search_s": search_elapsed,
    }


def _serialize_candidate(
    *,
    circuit_id: str,
    split: str,
    network: Any,
    inputs: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], PathCandidate | None]:
    dag = build_contraction_dag(network, item["path"])
    conventional = extract_conventional_features(dag)
    logical_id = contraction_dag_hash(dag)
    memory_estimate = _host_memory_estimate(inputs, dag)
    memory_limit = int(config["host_memory_admission_bytes"])
    estimated_work_units = _estimated_work_unit_count(dag)
    work_unit_limit = int(config["candidate_generation"]["maximum_planned_work_units"])
    if estimated_work_units > work_unit_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="estimated_work_unit_count_exceeds_preregistered_bound",
            conventional=conventional,
            logical_plan_id=logical_id,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    feature_pairs: list[tuple[str, RawFeatureVector]] = []
    feasible_topologies: list[str] = []
    topology_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for topology_id, topology in _topologies(config):
        feasible = memory_estimate <= memory_limit
        reason = None if feasible else "host_memory_estimate_exceeds_preregistered_bound"
        plan = None
        facts = None
        admission = None
        try:
            if feasible:
                plan = plan_upmem(dag, numeric_policy=NUMERIC_POLICY, topology=topology)
                admission = collection_resource_admission(plan)
                facts = extract_plan_features(plan)
                if not admission["collection_resource_admission_passed"]:
                    feasible = False
                    reasons = admission["collection_resource_admission_reasons"]
                    reason = "collection_resource_admission_failed:" + ",".join(reasons)
        except Exception as exc:
            feasible = False
            reason = f"{type(exc).__name__}:{exc}"
        if plan is not None and facts is not None:
            physical_id = physical_plan_id(plan)
            fact_values = facts.as_mapping()
        else:
            physical_id = None
            fact_values = {}
        if feasible and plan is not None and facts is not None:
            feature_pairs.append((topology_id, facts.raw))
            feasible_topologies.append(topology_id)
        topology_records.append(
            {
                "topology_id": topology_id,
                "topology": asdict(topology),
                "feasible": feasible,
                "infeasibility_reason": reason,
                "physical_plan_id": physical_id,
                "resource_admission": admission,
                "features": fact_values,
                "host_memory_estimate_bytes": memory_estimate,
            }
        )
        row = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": item["candidate_path_id"],
            "source_kind": item["source_kind"],
            "source_seed": item["source_seed"],
            "is_greedy": item["is_greedy"],
            "topology_id": topology_id,
            "feasible": feasible,
            "infeasibility_reason": reason or "",
            "logical_plan_id": logical_id,
            "physical_plan_id": physical_id or "",
            **conventional.as_mapping(),
            **fact_values,
            "host_memory_estimate_bytes": memory_estimate,
        }
        rows.append({column: row.get(column, "") for column in FEATURE_COLUMNS})
    candidate = PathCandidate(
        path_id=item["candidate_path_id"],
        conventional=conventional,
        features_by_topology=tuple(feature_pairs),
        feasible_topologies=tuple(feasible_topologies),
        is_greedy=bool(item["is_greedy"]),
        source=str(item["source_kind"]),
    )
    record = {
        "candidate_path_id": item["candidate_path_id"],
        "path": [list(step) for step in item["path"]],
        "source_kind": item["source_kind"],
        "source_seed": item["source_seed"],
        "planner_config_hash": item["planner_config_hash"],
        "is_greedy": item["is_greedy"],
        "logical_plan_id": logical_id,
        "conventional_features": conventional.as_mapping(),
        "topologies": topology_records,
    }
    return record, rows, candidate


def _serialized_candidate_with_admission(
    *,
    circuit_id: str,
    split: str,
    definition: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], PathCandidate | None]:
    # Perform deterministic, target-neutral admission before entering native
    # planning in a child. This avoids paying or timing out physical lowering
    # for paths already known to violate the frozen campaign bounds.
    circuit = builtin_circuit(str(definition["name"]), dict(definition["parameters"]))
    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    dag = build_contraction_dag(network, item["path"])
    conventional = extract_conventional_features(dag)
    memory_estimate = _host_memory_estimate(inputs, dag)
    estimated_work_units = _estimated_work_unit_count(dag)
    work_unit_limit = int(config["candidate_generation"]["maximum_planned_work_units"])
    memory_limit = int(config["host_memory_admission_bytes"])
    identity_limit = int(
        config["candidate_generation"]["maximum_semantic_identity_expansion_units"]
    )
    identity_expansion = _semantic_identity_expansion_units(
        dag, stop_after=identity_limit
    )
    if estimated_work_units > work_unit_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="estimated_work_unit_count_exceeds_preregistered_bound",
            conventional=conventional,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    if memory_estimate > memory_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="host_memory_estimate_exceeds_preregistered_bound",
            conventional=conventional,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    if identity_expansion > identity_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="semantic_identity_expansion_exceeds_preregistered_bound",
            conventional=conventional,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
            semantic_identity_expansion_units=identity_expansion,
        )
    started = time.perf_counter()
    result = _serialize_candidate(
        circuit_id=circuit_id,
        split=split,
        network=network,
        inputs=inputs,
        item=item,
        config=config,
    )
    record, rows, candidate = result
    record["semantic_identity_expansion_units"] = identity_expansion
    for topology in record["topologies"]:
        topology["semantic_identity_expansion_units"] = identity_expansion
    for row in rows:
        row["semantic_identity_expansion_units"] = identity_expansion
    timeout_s = float(config["candidate_generation"]["physical_lowering_timeout_s"])
    elapsed = time.perf_counter() - started
    if elapsed > timeout_s:
        raise RuntimeError(
            "candidate lowering exceeded the generation guard; candidate "
            f"membership was not changed: {item['candidate_path_id']} "
            f"({elapsed:.3f}s > {timeout_s:g}s)"
        )
    return record, rows, candidate


def _infeasible_candidate_record(
    *,
    circuit_id: str,
    split: str,
    item: dict[str, Any],
    config: dict[str, Any],
    reason: str,
    conventional: ConventionalPathFeatures | None = None,
    logical_plan_id: str | None = None,
    host_memory_estimate_bytes: int | None = None,
    estimated_work_unit_count: int | None = None,
    semantic_identity_expansion_units: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], None]:
    topology_records = []
    rows = []
    for topology_id, topology in _topologies(config):
        topology_records.append(
            {
                "topology_id": topology_id,
                "topology": asdict(topology),
                "feasible": False,
                "infeasibility_reason": reason,
                "physical_plan_id": None,
                "resource_admission": None,
                "features": {},
                "host_memory_estimate_bytes": host_memory_estimate_bytes,
                "estimated_work_unit_count": estimated_work_unit_count,
                "semantic_identity_expansion_units": semantic_identity_expansion_units,
            }
        )
        row = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": item["candidate_path_id"],
            "source_kind": item["source_kind"],
            "source_seed": item["source_seed"],
            "is_greedy": item["is_greedy"],
            "topology_id": topology_id,
            "feasible": False,
            "infeasibility_reason": reason,
        }
        rows.append({column: row.get(column, "") for column in FEATURE_COLUMNS})
    return (
        {
            "candidate_path_id": item["candidate_path_id"],
            "path": [list(step) for step in item["path"]],
            "source_kind": item["source_kind"],
            "source_seed": item["source_seed"],
            "planner_config_hash": item["planner_config_hash"],
            "is_greedy": item["is_greedy"],
            "logical_plan_id": logical_plan_id,
            "conventional_features": (
                conventional.as_mapping() if conventional is not None else None
            ),
            "semantic_identity_expansion_units": semantic_identity_expansion_units,
            "topologies": topology_records,
        },
        rows,
        None,
    )


def build_dataset(
    config: dict[str, Any],
    *,
    candidate_partition: tuple[int, int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    source_sha = _source_sha()
    circuit_records = []
    feature_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    calibration_cells = []
    total_generation_s = 0.0
    total_feature_s = 0.0
    for circuit_spec in config["circuits"]:
        circuit_id = str(circuit_spec["circuit_id"])
        split = str(circuit_spec["split"])
        definition = circuit_spec["circuit"]
        circuit = builtin_circuit(str(definition["name"]), dict(definition["parameters"]))
        job = make_simulation_job(circuit)
        network, _ = lower_tensor_network(job)
        raw_candidates, timing = _candidate_paths(
            network,
            circuit_id,
            config,
            isolate_trials=True,
            circuit_definition=definition,
        )
        if candidate_partition is not None:
            partition_index, partition_count = candidate_partition
            raw_candidates = [raw_candidates[0]] + [
                item
                for index, item in enumerate(raw_candidates[1:])
                if index % partition_count == partition_index
            ]
        total_generation_s += timing["candidate_generation_s"]
        candidate_records = []
        path_candidates = []
        feature_started = time.perf_counter()
        for item in raw_candidates:
            record, rows, candidate = _serialized_candidate_with_admission(
                circuit_id=circuit_id,
                split=split,
                definition=definition,
                item=item,
                config=config,
            )
            candidate_records.append(record)
            feature_rows.extend(rows)
            if candidate is not None:
                path_candidates.append(candidate)
        total_feature_s += time.perf_counter() - feature_started
        for topology_id, _ in _topologies(config):
            feasible = [candidate for candidate in path_candidates if candidate.feasible_for(topology_id)]
            greedy = next((candidate for candidate in feasible if candidate.is_greedy), None)
            if greedy is None:
                raise ValueError(f"greedy candidate is infeasible for {circuit_id}/{topology_id}")
            normalized = tuple(
                normalize_features(candidate.raw_for(topology_id), greedy.raw_for(topology_id))
                for candidate in feasible
            )
            model = choose_feature_model(normalized)
            weights = equal_model_weights(model)
            ordered = sorted(
                feasible,
                key=lambda candidate: (
                    score_features(candidate.raw_for(topology_id), greedy.raw_for(topology_id), weights, model=model),
                    candidate.path_id,
                ),
            )
            for rank, candidate in enumerate(ordered, start=1):
                ranking_rows.append(
                    {
                        "circuit_id": circuit_id,
                        "split": split,
                        "topology_id": topology_id,
                        "candidate_path_id": candidate.path_id,
                        "equal_weight_rank": rank,
                        "equal_weight_score": score_features(
                            candidate.raw_for(topology_id), greedy.raw_for(topology_id), weights, model=model
                        ),
                        "feature_model": model.mode,
                    }
                )
            if split == "training":
                selected = select_calibration_candidates(
                    feasible,
                    topology_id,
                    limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                    model=model,
                    greedy_path_id=greedy.path_id,
                )
                calibration_cells.append(
                    {
                        "cell_id": f"{circuit_id}:{topology_id}",
                        "circuit_id": circuit_id,
                        "topology_id": topology_id,
                        "greedy_path_id": greedy.path_id,
                        "feature_model": asdict(model),
                        "candidate_path_ids": [candidate.path_id for candidate in selected],
                    }
                )
        circuit_records.append(
            {
                "circuit_id": circuit_id,
                "split": split,
                "circuit": definition,
                "problem_id": problem_id(job),
                "tensor_network_structure_id": tensor_network_structure_id(network),
                "requested_cotengra_trials": int(config["candidate_generation"]["one_trial_searches"]),
                "unique_candidate_count": len(candidate_records),
                "duplicate_count": 1 + int(config["candidate_generation"]["one_trial_searches"]) - len(candidate_records),
                "candidates": candidate_records,
            }
        )
    config_hash = _sha256_bytes(_canonical_bytes(config))
    dataset = {
        "schema_version": SCHEMA_VERSION,
        "source_sha": source_sha,
        "score_id": COST_MODEL_ID,
        "preregistration_sha256": config_hash,
        "dependency_versions": {
            "numpy": _version("numpy"),
            "opt_einsum": _version("opt_einsum"),
            "cotengra": _version("cotengra"),
            "quimb": _version("quimb"),
        },
        "feature_dependencies": [asdict(item) for item in feature_dependency_metadata()],
        "circuits": circuit_records,
    }
    calibration = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": source_sha,
        "candidate_set_sha256": _sha256_bytes(_canonical_bytes(dataset)),
        "timing_used_for_selection": False,
        "cells": calibration_cells,
    }
    return dataset, feature_rows, ranking_rows, calibration, {
        "candidate_generation_s": total_generation_s,
        "feature_extraction_s": total_feature_s,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _required_sha(value: object, field: str, length: int = 40) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{field} must be a {length}-character SHA-256 value")
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(f"{field} must be hexadecimal") from None
    return value


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _operation_timing_total(
    sample: Mapping[str, Any], field: str
) -> float | None:
    facts = _mapping(sample.get("backend_facts"), "sample backend_facts")
    operations = facts.get("operation_facts")
    if not isinstance(operations, list):
        return None
    values: list[float] = []
    for index, operation in enumerate(operations):
        operation_mapping = _mapping(operation, f"operation_facts[{index}]")
        timing = _mapping(
            operation_mapping.get("timing"), f"operation_facts[{index}].timing"
        )
        value = timing.get(field)
        if value is None:
            return None
        values.append(_finite_nonnegative(value, f"operation timing {field}"))
    return sum(values)


def _calibration_candidate_index(
    dataset: Mapping[str, Any], calibration: Mapping[str, Any]
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
    str,
]:
    if dataset.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate dataset has an invalid schema version")
    if calibration.get("schema_version") != "upmem_path_calibration_candidate_set_v1":
        raise ValueError("calibration candidate set has an invalid schema version")
    candidate_source = _required_sha(dataset.get("source_sha"), "candidate source_sha")
    if calibration.get("source_sha") != candidate_source:
        raise ValueError("calibration source_sha does not match candidate dataset")
    candidate_set_sha = _sha256_bytes(_canonical_bytes(dict(dataset)))
    if calibration.get("candidate_set_sha256") != candidate_set_sha:
        raise ValueError("calibration candidate-set identity does not match dataset")
    if calibration.get("timing_used_for_selection") is not False:
        raise ValueError("calibration candidate selection must not use timing")

    circuit_map: dict[str, dict[str, Any]] = {}
    candidate_map: dict[tuple[str, str], dict[str, Any]] = {}
    circuits = dataset.get("circuits")
    if not isinstance(circuits, list) or not circuits:
        raise ValueError("candidate dataset must contain circuits")
    for circuit in circuits:
        circuit_mapping = dict(_mapping(circuit, "candidate circuit"))
        circuit_id = str(circuit_mapping.get("circuit_id", ""))
        if not circuit_id or circuit_id in circuit_map:
            raise ValueError("candidate dataset has duplicate or empty circuit IDs")
        if circuit_mapping.get("split") != "training":
            continue
        circuit_map[circuit_id] = circuit_mapping
        candidates = circuit_mapping.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"candidate list is missing for {circuit_id}")
        for candidate in candidates:
            candidate_mapping = dict(_mapping(candidate, "candidate"))
            candidate_id = str(candidate_mapping.get("candidate_path_id", ""))
            key = (circuit_id, candidate_id)
            if not candidate_id or key in candidate_map:
                raise ValueError("candidate dataset has duplicate or empty path IDs")
            candidate_map[key] = candidate_mapping

    cells = calibration.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("calibration candidate set must contain cells")
    expected: dict[tuple[str, str, str], dict[str, Any]] = {}
    cell_map: dict[str, dict[str, Any]] = {}
    cell_topology_keys: set[tuple[str, str]] = set()
    for cell in cells:
        cell_mapping = dict(_mapping(cell, "calibration cell"))
        cell_id = str(cell_mapping.get("cell_id", ""))
        circuit_id = str(cell_mapping.get("circuit_id", ""))
        topology_id = str(cell_mapping.get("topology_id", ""))
        if not cell_id or cell_id in cell_map:
            raise ValueError("calibration cells must have unique nonempty IDs")
        if circuit_id not in circuit_map:
            raise ValueError(f"calibration cell references unknown training circuit: {circuit_id}")
        if (circuit_id, topology_id) in cell_topology_keys:
            raise ValueError("calibration cells must have unique circuit/topology pairs")
        cell_topology_keys.add((circuit_id, topology_id))
        path_ids = cell_mapping.get("candidate_path_ids")
        if not isinstance(path_ids, list) or not path_ids:
            raise ValueError(f"calibration cell has no candidate paths: {cell_id}")
        if len(set(path_ids)) != len(path_ids):
            raise ValueError(f"calibration cell repeats a candidate path: {cell_id}")
        greedy_id = cell_mapping.get("greedy_path_id")
        if greedy_id not in path_ids:
            raise ValueError(f"calibration cell greedy path is not selected: {cell_id}")
        cell_map[cell_id] = cell_mapping
        for candidate_id in path_ids:
            candidate = candidate_map.get((circuit_id, str(candidate_id)))
            if candidate is None:
                raise ValueError(
                    f"calibration cell references unknown candidate: {cell_id}/{candidate_id}"
                )
            topologies = candidate.get("topologies")
            if not isinstance(topologies, list):
                raise ValueError(f"candidate lacks topology records: {candidate_id}")
            topology = next(
                (item for item in topologies if item.get("topology_id") == topology_id),
                None,
            )
            if not isinstance(topology, Mapping):
                raise ValueError(f"candidate lacks topology {topology_id}: {candidate_id}")
            if topology.get("feasible") is not True:
                raise ValueError(
                    f"calibration candidate is infeasible: {cell_id}/{candidate_id}"
                )
            admission = _mapping(
                topology.get("resource_admission"),
                f"candidate resource admission {cell_id}/{candidate_id}",
            )
            if admission.get("collection_resource_admission_passed") is not True:
                raise ValueError(
                    f"calibration candidate lacks resource admission: {cell_id}/{candidate_id}"
                )
            physical_plan_id_value = topology.get("physical_plan_id")
            _required_sha(
                physical_plan_id_value,
                f"candidate physical_plan_id {cell_id}/{candidate_id}",
                64,
            )
            expected[(circuit_id, topology_id, str(candidate_id))] = {
                "cell_id": cell_id,
                "circuit": circuit_map[circuit_id],
                "candidate": candidate,
                "topology": dict(topology),
                "greedy_path_id": str(greedy_id),
            }
    return expected, cell_map, candidate_set_sha, candidate_source


def _manifest_calibration_contract(
    manifest: Mapping[str, Any],
    expected: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    if manifest.get("status") != "completed":
        raise ValueError("raw evidence manifest must be completed")
    if manifest.get("source_worktree_dirty") is not False:
        raise ValueError("physical execution source must be clean")
    physical_source = _required_sha(
        manifest.get("source_commit"), "physical execution source_commit"
    )
    experiment_id = _required_sha(manifest.get("experiment_id"), "experiment_id", 64)
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest run_id must be nonempty")
    configuration = _mapping(manifest.get("configuration"), "manifest configuration")
    experiment = _mapping(configuration.get("experiment"), "manifest experiment")
    collection = _mapping(experiment.get("collection"), "experiment collection")
    if collection.get("claim_policy") != "diagnostic_v1":
        raise ValueError("calibration evidence must use diagnostic_v1")
    if collection.get("warmup_blocks") != 1 or collection.get("measurement_blocks") != 3:
        raise ValueError("calibration evidence must use one warmup and three measurements")
    if collection.get("session_policy") != "fresh_session_per_attempt_v1":
        raise ValueError("calibration evidence must use fresh sessions")
    if experiment.get("experiment_id") != experiment_id:
        raise ValueError("manifest experiment identity is inconsistent")

    expected_matrix = {
        (circuit_id, f"path_{candidate_id}", topology_id)
        for circuit_id, topology_id, candidate_id in expected
    }
    actual_matrix: list[tuple[str, str, str]] = []
    matrix = experiment.get("matrix")
    if not isinstance(matrix, list):
        raise ValueError("manifest experiment matrix is missing")
    for item in matrix:
        matrix_item = _mapping(item, "experiment matrix item")
        case_id = matrix_item.get("case_id")
        plan_id = matrix_item.get("plan_id")
        route_ids = matrix_item.get("route_ids")
        if not isinstance(case_id, str) or not isinstance(plan_id, str):
            raise ValueError("experiment matrix identity is invalid")
        if not isinstance(route_ids, list) or not route_ids:
            raise ValueError("experiment matrix route list is invalid")
        for route_id in route_ids:
            if not isinstance(route_id, str):
                raise ValueError("experiment matrix route ID is invalid")
            actual_matrix.append((case_id, plan_id, route_id))
    if len(actual_matrix) != len(set(actual_matrix)) or set(actual_matrix) != expected_matrix:
        raise ValueError("manifest matrix does not match calibration cell/path set exactly")
    return physical_source, run_id, {
        "experiment_id": experiment_id,
        "collection": dict(collection),
        "environment": dict(_mapping(configuration.get("environment"), "environment")),
    }


def _joined_backend_facts(
    sample: Mapping[str, Any], session: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_facts = dict(_mapping(sample.get("backend_facts"), "sample backend_facts"))
    terminal = dict(
        _mapping(session.get("terminal_backend_facts"), "terminal backend facts")
    )
    # These identify different evidence scopes: the aggregate physical plan and
    # the native rank session. They are both retained in their original records.
    scope_specific_fields = {"backend_id", "execution_class"}
    for field in (set(sample_facts) & set(terminal)) - scope_specific_fields:
        if sample_facts[field] != terminal[field]:
            raise ValueError(f"sample/session backend fact conflict: {field}")
    joined = dict(sample_facts)
    for field, value in terminal.items():
        joined.setdefault(field, value)
    return joined, terminal


def _require_backend_contract(
    sample: Mapping[str, Any],
    session: Mapping[str, Any],
    facts: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> None:
    if session.get("status") != "success":
        raise ValueError("calibration contains a non-success session")
    for field in ("release_attempted", "release_succeeded", "release_verified"):
        if session.get(field) is not True:
            raise ValueError(f"session {field} must be true")
    expected_terminal = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "allocation_verified": True,
        "hardware_allocation_verified": True,
        "binary_identity_verified": True,
        "native_identity_verified": True,
        "hardware_release_verified": True,
        "startup_resource_admission_passed": True,
    }
    terminal = _mapping(session.get("terminal_backend_facts"), "terminal backend facts")
    for field, value in expected_terminal.items():
        if terminal.get(field) != value:
            raise ValueError(f"terminal physical fact {field} is not qualified")
    expected_topology = _mapping(topology.get("topology"), "candidate topology")
    dpu_count = int(expected_topology["dpu_count"])
    rank_count = int(expected_topology["rank_count"])
    tasklets = int(expected_topology["tasklets_per_dpu"])
    expected_facts = {
        "target_observed": "physical_hardware",
        "physical_target_verified": True,
        "hardware_kernel_executed": True,
        "simulator_kernel_executed": False,
        "cpu_fallback_used": False,
        "collection_resource_admission_passed": True,
        "execution_resource_admission_passed": True,
        "startup_resource_admission_passed": True,
        "requested_dpus": dpu_count,
        "allocated_dpus": dpu_count,
        "active_dpus": dpu_count,
        "tasklets_per_dpu": tasklets,
        "rank_count": rank_count,
        "request_transport": CALIBRATION_TRANSPORT,
    }
    for field, value in expected_facts.items():
        if facts.get(field) != value:
            raise ValueError(f"sample physical fact {field} is not qualified")
    for field, value in {
        "requested_dpu_count": dpu_count,
        "allocated_dpu_count": dpu_count,
        "observed_dpu_count": dpu_count,
        "observed_tasklets_per_dpu": tasklets,
        "startup_requested_dpu_count": dpu_count,
        "startup_allocated_dpu_count": dpu_count,
        "startup_requested_tasklets_per_dpu": tasklets,
    }.items():
        if terminal.get(field) != value:
            raise ValueError(f"terminal resource fact {field} is not qualified")
    if sample.get("validation", {}).get("accuracy_qualified") is not True:
        raise ValueError("calibration sample accuracy is not qualified")
    validation = _mapping(sample.get("validation"), "sample validation")
    for applicable, passed in (
        ("full_precision_threshold_applicable", "full_precision_passed"),
        ("policy_reference_applicable", "policy_reference_passed"),
    ):
        if validation.get(applicable) is True and validation.get(passed) is not True:
            raise ValueError(f"calibration sample validation {passed} is not true")
    if not isinstance(sample.get("output_sha256"), str):
        raise ValueError("calibration sample lacks output hash")


def _calibration_row(
    *,
    sample: Mapping[str, Any],
    session: Mapping[str, Any],
    facts: Mapping[str, Any],
    terminal: Mapping[str, Any],
    expected_item: Mapping[str, Any],
    physical_source: str,
    candidate_source: str,
    candidate_set_sha: str,
    calibration_set_sha: str,
    raw_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    measurement = _mapping(sample.get("measurement"), "sample measurement")
    if measurement.get("scope_id") != CALIBRATION_TIMING_SCOPE:
        raise ValueError("calibration sample timing scope is not steady_execution_v1")
    total_wall = _finite_nonnegative(measurement.get("total_wall_s"), "total_wall_s")
    open_s = _finite_nonnegative(session.get("open_s"), "session open_s")
    close_s = _finite_nonnegative(session.get("session_close_s"), "session_close_s")
    row: dict[str, Any] = {
        "split": "training",
        "attempt_type": sample["attempt_kind"],
        "cell_id": expected_item["cell_id"],
        "circuit_id": sample["case_id"],
        "topology_id": sample["route_id"],
        "candidate_path_id": str(sample["plan_id"])[len("path_"):],
        "plan_id": sample["plan_id"],
        "route_id": sample["route_id"],
        "block": sample["block_id"],
        "sample_index": sample["sample_index"],
        "order_index": sample["order_index"],
        "sample_id": sample["sample_id"],
        "session_instance_id": sample["session_instance_id"],
        "experiment_id": sample["experiment_id"],
        "run_id": sample["run_id"],
        "source_sha": candidate_source,
        "candidate_generation_source_sha": candidate_source,
        "physical_execution_source_sha": physical_source,
        "candidate_set_sha256": candidate_set_sha,
        "calibration_set_sha256": calibration_set_sha,
        "problem_id": sample["identities"]["problem_id"],
        "tensor_network_structure_id": sample["identities"]["tensor_network_structure_id"],
        "logical_plan_id": sample["identities"]["logical_plan_id"],
        "physical_plan_id": sample["identities"]["physical_plan_id"],
        "executable_id": sample["identities"]["executable_id"],
        "validation_policy_id": sample["identities"]["validation_policy_id"],
        "output_sha256": sample["output_sha256"],
        "status": sample["status"],
        "validation": "passed",
        "fallback": "false",
        "max_abs_error": sample["validation"].get("max_abs_error"),
        "relative_l2_error": sample["validation"].get("relative_l2_error"),
        "norm_drift": sample["validation"].get("norm_drift"),
        "phase_aligned_max_abs_error": sample["validation"].get(
            "phase_aligned_max_abs_error"
        ),
        "full_precision_passed": sample["validation"].get("full_precision_passed"),
        "policy_reference_passed": sample["validation"].get(
            "policy_reference_passed"
        ),
        "timing_scope": measurement["scope_id"],
        "total_wall_s": total_wall,
        "session_open_s": open_s,
        "session_close_s": close_s,
        "session_inclusive_s": open_s + total_wall + close_s,
        "kernel_s": measurement.get("kernel_s"),
        "h2d_s": measurement.get("h2d_s"),
        "d2h_s": measurement.get("d2h_s"),
        "h2d_bytes": measurement.get("h2d_bytes"),
        "d2h_bytes": measurement.get("d2h_bytes"),
        "preparation_s": measurement.get("preparation_s"),
        "planning_s": measurement.get("planning_s"),
        "lowering_s": measurement.get("lowering_s"),
        "mapping_s": measurement.get("mapping_s"),
        "slicing_s": measurement.get("slicing_s"),
        "host_reduce_s": measurement.get("host_reduce_s"),
        "rank_work_s": measurement.get("rank_work_s"),
        "request_build_s": _operation_timing_total(sample, "request_build_sum_s"),
        "request_wave_s": _operation_timing_total(sample, "request_wave_wall_sum_s"),
        "request_artifact_build_s": _operation_timing_total(sample, "request_artifact_build_sum_s"),
        "payload_record_staging_s": _operation_timing_total(sample, "request_payload_record_staging_sum_s"),
        "request_work_unit_materialization_s": _operation_timing_total(sample, "request_work_unit_materialization_sum_s"),
        "request_payload_materialization_s": _operation_timing_total(sample, "request_payload_materialization_sum_s"),
        "request_payload_hashing_s": _operation_timing_total(sample, "request_payload_hashing_sum_s"),
        "request_payload_file_write_s": _operation_timing_total(sample, "request_payload_file_write_sum_s"),
        "request_manifest_sidecar_staging_s": _operation_timing_total(sample, "request_manifest_sidecar_staging_sum_s"),
        "request_build_residual_s": _operation_timing_total(sample, "request_build_residual_sum_s"),
        "request_payload_record_count": facts.get("request_payload_record_count"),
        "request_payload_bytes_staged": facts.get("request_payload_bytes_staged"),
        "request_payload_bytes_hashed": facts.get("request_payload_bytes_hashed"),
        "request_payload_files_created": facts.get("request_payload_files_created"),
        "requested_dpus": facts["requested_dpus"],
        "allocated_dpus": facts["allocated_dpus"],
        "active_dpus": facts["active_dpus"],
        "tasklets_per_dpu": facts["tasklets_per_dpu"],
        "rank_count": facts["rank_count"],
        "active_rank_count": facts.get("execution_active_rank_count"),
        "target_observed": facts["target_observed"],
        "request_transport": facts["request_transport"],
        "collection_resource_admission_passed": facts["collection_resource_admission_passed"],
        "execution_resource_admission_passed": facts["execution_resource_admission_passed"],
        "startup_resource_admission_passed": facts["startup_resource_admission_passed"],
        "physical_target_verified": facts["physical_target_verified"],
        "hardware_kernel_executed": facts["hardware_kernel_executed"],
        "simulator_kernel_executed": facts["simulator_kernel_executed"],
        "cpu_fallback_used": facts["cpu_fallback_used"],
        "binary_identity_verified": facts.get("binary_identity_verified"),
        "native_identity_verified": facts.get("native_identity_verified"),
        "hardware_release_verified": terminal["hardware_release_verified"],
        "tasklet_utilization": facts.get("arithmetic_weighted_tasklet_utilization"),
        "dpu_utilization": facts.get("arithmetic_weighted_dpu_slot_utilization"),
        "dominant_wave_utilization": facts.get("dominant_work_wave_utilization"),
        "total_wave_count": facts.get("total_wave_count"),
        "fully_populated_wave_count": facts.get("fully_populated_wave_count"),
        "active_dpu_ids_json": _json_text(facts.get("active_dpu_ids")),
        "active_rank_indices_json": _json_text(facts.get("active_rank_indices")),
        "requested_rank_paths_json": _json_text(raw_hashes["rank_paths"]),
    }
    for field in (
        "kernel_s", "h2d_s", "d2h_s", "preparation_s", "planning_s", "lowering_s",
        "mapping_s", "slicing_s", "host_reduce_s", "rank_work_s",
        "max_abs_error", "relative_l2_error", "norm_drift",
        "phase_aligned_max_abs_error",
    ):
        if row[field] is not None:
            row[field] = _finite_nonnegative(row[field], field)
    for field in (
        "h2d_bytes", "d2h_bytes", "request_payload_record_count",
        "request_payload_bytes_staged", "request_payload_bytes_hashed",
        "request_payload_files_created", "requested_dpus", "allocated_dpus",
        "active_dpus", "tasklets_per_dpu", "rank_count", "active_rank_count",
        "total_wave_count", "fully_populated_wave_count",
    ):
        if row[field] is not None and (
            isinstance(row[field], bool) or not isinstance(row[field], int)
        ):
            raise ValueError(f"{field} must be an integer when present")
    return row


def extract_calibration(
    raw_dir: Path,
    candidate_path: Path,
    calibration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract a strict physical calibration table from canonical evidence."""

    dataset = _mapping(
        json.loads(candidate_path.read_text(encoding="utf-8")), "candidate dataset"
    )
    calibration = _mapping(
        json.loads(calibration_path.read_text(encoding="utf-8")),
        "calibration candidate set",
    )
    expected, cells, candidate_set_sha, candidate_source = _calibration_candidate_index(
        dataset, calibration
    )
    manifest, samples, sessions = load_artifacts(raw_dir)
    physical_source, run_id, manifest_contract = _manifest_calibration_contract(
        manifest, expected
    )
    experiment_id = manifest_contract["experiment_id"]
    sessions_by_id: dict[str, Mapping[str, Any]] = {}
    for session in sessions:
        session_id = session.get("session_instance_id")
        if not isinstance(session_id, str) or session_id in sessions_by_id:
            raise ValueError("sessions must have unique session_instance_id values")
        sessions_by_id[session_id] = session
    expected_observations = len(expected) * 4
    if len(samples) != expected_observations or len(sessions) != expected_observations:
        raise ValueError(
            "canonical evidence count does not match calibration set and block schedule"
        )
    calibration_set_sha = _sha256_bytes(_canonical_bytes(dict(calibration)))
    raw_root = Path(raw_dir)
    raw_hashes = {
        "manifest": _file_sha256(raw_root / "manifest.json"),
        "samples": _file_sha256(raw_root / "samples.jsonl"),
        "sessions": _file_sha256(raw_root / "sessions.jsonl"),
        "rank_paths": manifest_contract["environment"].get("requested_rank_paths", []),
    }
    seen: set[tuple[str, str, int, str]] = set()
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for sample in samples:
        if sample.get("experiment_id") != experiment_id or sample.get("run_id") != run_id:
            raise ValueError("sample experiment/run identity does not match manifest")
        case_id = sample.get("case_id")
        route_id = sample.get("route_id")
        plan_id = sample.get("plan_id")
        if not isinstance(case_id, str) or not isinstance(route_id, str):
            raise ValueError("sample case/route identity is invalid")
        if not isinstance(plan_id, str) or not plan_id.startswith("path_"):
            raise ValueError("sample plan_id is not a candidate path plan")
        candidate_id = plan_id.removeprefix("path_")
        expected_item = expected.get((case_id, route_id, candidate_id))
        if expected_item is None:
            raise ValueError(f"sample is outside the exact calibration set: {plan_id}")
        attempt_kind = sample.get("attempt_kind")
        block_id = sample.get("block_id")
        if (attempt_kind, block_id) not in {
            ("warmup", 0), ("measurement", 1), ("measurement", 2), ("measurement", 3)
        }:
            raise ValueError("sample is outside blocks 0..3 with one warmup")
        key = (
            str(expected_item["cell_id"]),
            candidate_id,
            int(block_id),
            str(attempt_kind),
        )
        if key in seen:
            raise ValueError(f"duplicate calibration observation: {key}")
        seen.add(key)
        if sample.get("status") != "success":
            raise ValueError("calibration contains a non-success sample")
        session_id = sample.get("session_instance_id")
        if not isinstance(session_id, str) or session_id not in sessions_by_id:
            raise ValueError("sample references a missing session")
        session = sessions_by_id[session_id]
        for field in ("experiment_id", "run_id", "case_id", "plan_id", "route_id"):
            if session.get(field) != sample.get(field):
                raise ValueError(f"sample/session {field} identity mismatch")
        facts, terminal = _joined_backend_facts(sample, session)
        _require_backend_contract(sample, session, facts, expected_item["topology"])
        identities = _mapping(sample.get("identities"), "sample identities")
        circuit = _mapping(expected_item["circuit"], "candidate circuit")
        candidate = _mapping(expected_item["candidate"], "candidate")
        topology = _mapping(expected_item["topology"], "candidate topology")
        identity_expected = {
            "problem_id": circuit["problem_id"],
            "tensor_network_structure_id": circuit["tensor_network_structure_id"],
            "logical_plan_id": candidate["logical_plan_id"],
            "physical_plan_id": topology["physical_plan_id"],
        }
        for field, value in identity_expected.items():
            if identities.get(field) != value:
                raise ValueError(f"sample identity {field} does not match candidate")
        row = _calibration_row(
            sample=sample,
            session=session,
            facts=facts,
            terminal=terminal,
            expected_item=expected_item,
            physical_source=physical_source,
            candidate_source=candidate_source,
            candidate_set_sha=candidate_set_sha,
            calibration_set_sha=calibration_set_sha,
            raw_hashes=raw_hashes,
        )
        rows.append(row)
        observations.append(dict(row))
    expected_keys = {
        (str(item["cell_id"]), candidate_id, block, attempt)
        for (_case_id, _topology_id, candidate_id), item in expected.items()
        for block, attempt in (
            (0, "warmup"), (1, "measurement"), (2, "measurement"), (3, "measurement")
        )
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)
        extra = sorted(seen - expected_keys)
        raise ValueError(f"calibration observations are not exact (missing={missing}, extra={extra})")
    if set(sessions_by_id) != {str(sample["session_instance_id"]) for sample in samples}:
        raise ValueError("sessions are not in a one-to-one relation with samples")
    rows.sort(key=lambda row: (row["cell_id"], row["candidate_path_id"], row["block"]))
    observations.sort(key=lambda row: (row["cell_id"], row["candidate_path_id"], row["block"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "path_runtime_calibration.csv", rows, CALIBRATION_COLUMNS)
    result = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "source_sha": candidate_source,
        "source_sha_semantics": "candidate_generation_source_sha",
        "candidate_generation_source_sha": candidate_source,
        "physical_execution_source_sha": physical_source,
        "candidate_set_sha256": candidate_set_sha,
        "calibration_set_sha256": calibration_set_sha,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "claim_policy": "diagnostic_v1",
        "timing_scope": CALIBRATION_TIMING_SCOPE,
        "numeric_policy": "split_complex_float32_v1",
        "request_transport": CALIBRATION_TRANSPORT,
        "collection": {
            "warmup_blocks": 1,
            "measurement_blocks": 3,
            "blocks": [0, 1, 2, 3],
            "attempts_per_candidate_cell": 4,
        },
        "expected_cell_count": len(cells),
        "expected_candidate_cell_count": len(expected),
        "sample_count": len(samples),
        "session_count": len(sessions),
        "all_successful_physical_sessions": True,
        "all_resource_admission_passed": True,
        "all_accuracy_qualified": True,
        "fallback_used": False,
        "raw_artifact_sha256": {
            "manifest.json": raw_hashes["manifest"],
            "samples.jsonl": raw_hashes["samples"],
            "sessions.jsonl": raw_hashes["sessions"],
        },
        "environment": manifest_contract["environment"],
        "cells": [dict(value) for _, value in sorted(cells.items())],
        "observations": observations,
    }
    (output_dir / "path_runtime_calibration.json").write_bytes(_canonical_bytes(result))
    return result


def generate(
    config_path: Path,
    output_dir: Path,
    *,
    check: bool = False,
    circuit_ids: tuple[str, ...] = (),
    candidate_partition: tuple[int, int] | None = None,
) -> dict[str, float]:
    full_config = load_config(config_path)
    config = full_config
    if circuit_ids:
        requested = set(circuit_ids)
        known = {str(item["circuit_id"]) for item in full_config["circuits"]}
        if not requested <= known:
            raise ValueError(f"unknown circuit shard IDs: {sorted(requested - known)!r}")
        config = {
            **full_config,
            "circuits": [
                item for item in full_config["circuits"]
                if item["circuit_id"] in requested
            ],
        }
    dataset, features, rankings, calibration, timings = build_dataset(
        config, candidate_partition=candidate_partition
    )
    preregistration_sha = _sha256_bytes(_canonical_bytes(full_config))
    dataset["preregistration_sha256"] = preregistration_sha
    calibration["candidate_set_sha256"] = _sha256_bytes(_canonical_bytes(dataset))
    outputs: dict[str, bytes] = {
        "candidate_paths.json": _canonical_bytes(dataset),
        "calibration_candidate_set.json": _canonical_bytes(calibration),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in outputs.items():
        path = output_dir / filename
        if check:
            if path.read_bytes() != contents:
                raise ValueError(f"{filename} differs from deterministic recomputation")
        else:
            path.write_bytes(contents)
    feature_path = output_dir / "candidate_features.csv"
    ranking_path = output_dir / "candidate_rankings.csv"
    ranking_columns = (
        "circuit_id", "split", "topology_id", "candidate_path_id",
        "equal_weight_rank", "equal_weight_score", "feature_model",
    )
    if check:
        import io

        for path, rows, columns in (
            (feature_path, features, FEATURE_COLUMNS),
            (ranking_path, rankings, ranking_columns),
        ):
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            if path.read_bytes() != stream.getvalue().encode("utf-8"):
                raise ValueError(f"{path.name} differs from deterministic recomputation")
    else:
        _write_csv(feature_path, features, FEATURE_COLUMNS)
        _write_csv(ranking_path, rankings, ranking_columns)
        (output_dir / "planning_timing.json").write_bytes(_canonical_bytes(timings))
    return timings


def merge_shards(
    config_path: Path,
    shard_dirs: tuple[Path, ...],
    output_dir: Path,
) -> dict[str, float]:
    """Merge exact-source circuit shards into one canonical frozen dataset."""

    if not shard_dirs:
        raise ValueError("at least one candidate shard is required")
    config = load_config(config_path)
    expected_order = [str(item["circuit_id"]) for item in config["circuits"]]
    expected_preregistration = _sha256_bytes(_canonical_bytes(config))
    datasets = [json.loads((path / "candidate_paths.json").read_text()) for path in shard_dirs]
    calibrations = [
        json.loads((path / "calibration_candidate_set.json").read_text())
        for path in shard_dirs
    ]
    base = {key: value for key, value in datasets[0].items() if key != "circuits"}
    if base["preregistration_sha256"] != expected_preregistration:
        raise ValueError("candidate shard does not match preregistration")
    circuit_parts: dict[str, list[dict[str, Any]]] = {}
    feature_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    calibration_cells: list[dict[str, Any]] = []
    timings = {"candidate_generation_s": 0.0, "feature_extraction_s": 0.0}
    for shard_dir, dataset, calibration in zip(
        shard_dirs, datasets, calibrations, strict=True
    ):
        if {key: value for key, value in dataset.items() if key != "circuits"} != base:
            raise ValueError("candidate shards have mixed source or dependency provenance")
        if calibration["source_sha"] != dataset["source_sha"]:
            raise ValueError("candidate shard calibration source mismatch")
        if calibration["candidate_set_sha256"] != _sha256_bytes(
            _canonical_bytes(dataset)
        ):
            raise ValueError("candidate shard checksum mismatch")
        for circuit in dataset["circuits"]:
            circuit_id = str(circuit["circuit_id"])
            circuit_parts.setdefault(circuit_id, []).append(circuit)
        with (shard_dir / "candidate_features.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            feature_rows.extend(csv.DictReader(stream))
        with (shard_dir / "candidate_rankings.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            ranking_rows.extend(csv.DictReader(stream))
        shard_timing = json.loads((shard_dir / "planning_timing.json").read_text())
        for key in timings:
            timings[key] += float(shard_timing[key])
    if set(circuit_parts) != set(expected_order):
        raise ValueError("candidate shards do not cover the frozen circuit set exactly")
    circuits: dict[str, dict[str, Any]] = {}
    for circuit_id, parts in circuit_parts.items():
        identity = {
            key: value for key, value in parts[0].items()
            if key not in {"candidates", "unique_candidate_count", "duplicate_count"}
        }
        candidates: dict[str, dict[str, Any]] = {}
        for part in parts:
            if {
                key: value for key, value in part.items()
                if key not in {"candidates", "unique_candidate_count", "duplicate_count"}
            } != identity:
                raise ValueError(f"mixed circuit shard identity: {circuit_id}")
            for candidate in part["candidates"]:
                candidate_id = candidate["candidate_path_id"]
                if candidate_id in candidates and candidates[candidate_id] != candidate:
                    raise ValueError(f"mixed duplicate candidate: {candidate_id}")
                candidates[candidate_id] = candidate
        ordered = sorted(
            candidates.values(),
            key=lambda item: (not item["is_greedy"], item["candidate_path_id"]),
        )
        circuits[circuit_id] = {
            **identity,
            "unique_candidate_count": len(ordered),
            "duplicate_count": 1
            + int(identity["requested_cotengra_trials"])
            - len(ordered),
            "candidates": ordered,
        }
    dataset = {**base, "circuits": [circuits[item] for item in expected_order]}
    circuit_order = {value: index for index, value in enumerate(expected_order)}
    topology_order = {
        str(item["topology_id"]): index
        for index, item in enumerate(config["topologies"])
    }
    candidate_order = {
        (circuit_id, candidate["candidate_path_id"]): index
        for circuit_id, circuit in circuits.items()
        for index, candidate in enumerate(circuit["candidates"])
    }
    feature_by_key = {
        (row["circuit_id"], row["candidate_path_id"], row["topology_id"]): row
        for row in feature_rows
    }
    feature_rows = list(feature_by_key.values())
    feature_rows.sort(
        key=lambda row: (
            circuit_order[row["circuit_id"]],
            candidate_order[(row["circuit_id"], row["candidate_path_id"])],
            topology_order[row["topology_id"]],
        )
    )
    ranking_rows = []
    calibration_cells = []
    for circuit in dataset["circuits"]:
        candidates = tuple(_candidate_from_record(item) for item in circuit["candidates"])
        for topology_id in topology_order:
            feasible = tuple(item for item in candidates if item.feasible_for(topology_id))
            greedy = next(item for item in feasible if item.is_greedy)
            normalized = tuple(
                normalize_features(item.raw_for(topology_id), greedy.raw_for(topology_id))
                for item in feasible
            )
            model = choose_feature_model(normalized)
            weights = equal_model_weights(model)
            ordered = sorted(
                feasible,
                key=lambda item: (
                    score_features(
                        item.raw_for(topology_id), greedy.raw_for(topology_id),
                        weights, model=model,
                    ),
                    item.path_id,
                ),
            )
            for rank, candidate in enumerate(ordered, start=1):
                ranking_rows.append({
                    "circuit_id": circuit["circuit_id"],
                    "split": circuit["split"],
                    "topology_id": topology_id,
                    "candidate_path_id": candidate.path_id,
                    "equal_weight_rank": rank,
                    "equal_weight_score": score_features(
                        candidate.raw_for(topology_id), greedy.raw_for(topology_id),
                        weights, model=model,
                    ),
                    "feature_model": model.mode,
                })
            if circuit["split"] == "training":
                selected = select_calibration_candidates(
                    feasible,
                    topology_id,
                    limit=int(config["calibration"]["candidates_per_cell_maximum"]),
                    model=model,
                    greedy_path_id=greedy.path_id,
                )
                calibration_cells.append({
                    "cell_id": f"{circuit['circuit_id']}:{topology_id}",
                    "circuit_id": circuit["circuit_id"],
                    "topology_id": topology_id,
                    "greedy_path_id": greedy.path_id,
                    "feature_model": asdict(model),
                    "candidate_path_ids": [item.path_id for item in selected],
                })
    calibration = {
        "schema_version": "upmem_path_calibration_candidate_set_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": _sha256_bytes(_canonical_bytes(dataset)),
        "timing_used_for_selection": False,
        "cells": calibration_cells,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_paths.json").write_bytes(_canonical_bytes(dataset))
    (output_dir / "calibration_candidate_set.json").write_bytes(
        _canonical_bytes(calibration)
    )
    _write_csv(output_dir / "candidate_features.csv", feature_rows, FEATURE_COLUMNS)
    _write_csv(
        output_dir / "candidate_rankings.csv",
        ranking_rows,
        (
            "circuit_id", "split", "topology_id", "candidate_path_id",
            "equal_weight_rank", "equal_weight_score", "feature_model",
        ),
    )
    (output_dir / "planning_timing.json").write_bytes(_canonical_bytes(timings))
    return timings


def _candidate_from_record(record: dict[str, Any]) -> PathCandidate:
    feasible = []
    pairs = []
    for item in record["topologies"]:
        if item["feasible"]:
            raw = RawFeatureVector.from_mapping(item["features"])
            pairs.append((str(item["topology_id"]), raw))
            feasible.append(str(item["topology_id"]))
    return PathCandidate(
        path_id=str(record["candidate_path_id"]),
        conventional=ConventionalPathFeatures(**record["conventional_features"]),
        features_by_topology=tuple(pairs),
        feasible_topologies=tuple(feasible),
        is_greedy=bool(record["is_greedy"]),
        source=str(record["source_kind"]),
    )


def fit(candidate_path: Path, calibration_path: Path, runtime_path: Path, output_dir: Path, *, samples: int, seed: int) -> WeightFitResult:
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    candidate_set_sha = _sha256_bytes(_canonical_bytes(dataset))
    if calibration.get("candidate_set_sha256") != candidate_set_sha:
        raise ValueError("calibration candidate-set identity does not match dataset")
    if calibration.get("source_sha") != dataset.get("source_sha"):
        raise ValueError("calibration source identity does not match dataset")
    by_circuit = {
        circuit["circuit_id"]: {
            candidate["candidate_path_id"]: _candidate_from_record(candidate)
            for candidate in circuit["candidates"]
        }
        for circuit in dataset["circuits"]
    }
    cells = []
    for item in calibration["cells"]:
        candidates = tuple(by_circuit[item["circuit_id"]][path] for path in item["candidate_path_ids"])
        cells.append(
            TrainingCell(
                cell_id=item["cell_id"],
                topology=item["topology_id"],
                candidates=candidates,
                greedy_path_id=item["greedy_path_id"],
            )
        )
    measurements = []
    physical_execution_sources: set[str] = set()
    expected_cells = {cell.cell_id: cell for cell in cells}
    expected_physical_plans = {
        (item["cell_id"], candidate_id): next(
            topology["physical_plan_id"]
            for topology in next(
                candidate
                for candidate in next(
                    circuit for circuit in dataset["circuits"]
                    if circuit["circuit_id"] == item["circuit_id"]
                )["candidates"]
                if candidate["candidate_path_id"] == candidate_id
            )["topologies"]
            if topology["topology_id"] == item["topology_id"]
        )
        for item in calibration["cells"]
        for candidate_id in item["candidate_path_ids"]
    }
    with runtime_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") != "training":
                raise ValueError("calibration runtime table contains a non-training row")
            if row.get("attempt_type") not in {"warmup", "measurement"}:
                raise ValueError("calibration runtime table has an invalid attempt type")
            if row.get("candidate_generation_source_sha") != dataset["source_sha"]:
                raise ValueError(
                    "calibration runtime candidate source does not match candidate dataset"
                )
            physical_source = str(row.get("physical_execution_source_sha", ""))
            if len(physical_source) != 40 or any(
                character not in "0123456789abcdef" for character in physical_source
            ):
                raise ValueError("calibration runtime physical source is not a full SHA")
            physical_execution_sources.add(physical_source)
            if row.get("timing_scope") != "steady_execution_v1":
                raise ValueError("calibration runtime table has an invalid timing scope")
            if row.get("status") != "success":
                raise ValueError("calibration runtime table contains a failed attempt")
            if row.get("validation") not in {"true", "passed", "1"}:
                raise ValueError("calibration runtime table contains an invalid output")
            if row.get("fallback") not in {"false", "0"}:
                raise ValueError("calibration runtime table contains fallback execution")
            cell = expected_cells.get(str(row.get("cell_id")))
            if cell is None:
                raise ValueError("calibration runtime table references an unknown cell")
            key = (cell.cell_id, str(row.get("candidate_path_id")))
            if row.get("physical_plan_id") != expected_physical_plans.get(key):
                raise ValueError("calibration runtime physical-plan identity mismatch")
            if row["attempt_type"] == "warmup":
                continue
            block = int(row["block"])
            if block not in {1, 2, 3}:
                raise ValueError("calibration measurement block must be 1, 2, or 3")
            measurements.append(
                RuntimeMeasurement(
                    cell_id=str(row["cell_id"]),
                    candidate_id=str(row["candidate_path_id"]),
                    runtime_s=float(row["total_wall_s"]),
                    split="train",
                    source_sha=physical_source,
                    timing_scope=str(row["timing_scope"]),
                    status=str(row["status"]),
                    observation_id=str(block),
                )
            )
    if len(physical_execution_sources) != 1:
        raise ValueError("calibration runtime table has mixed physical execution sources")
    physical_execution_source = next(iter(physical_execution_sources))
    output_dir.mkdir(parents=True, exist_ok=True)
    search_path = output_dir / "weight_search_candidates.csv"
    columns = (
        "weights_json", "selected_path_ids_json", "cell_speedups_json",
        "geometric_mean_speedup", "minimum_cell_speedup", "improved_cell_count",
        "equivalent_weight_vector_count",
    )
    representatives: dict[tuple[tuple[str, str], ...], WeightFitResult] = {}
    outcome_counts: dict[tuple[tuple[str, str], ...], int] = {}

    def result_order(item: WeightFitResult) -> tuple[float, float, int, tuple[float, ...]]:
        return (
            round(item.geometric_mean_speedup, 12),
            round(item.minimum_cell_speedup, 12),
            item.improved_cell_count,
            tuple(-value for value in item.weights.as_tuple()),
        )

    def record(item: WeightFitResult) -> None:
        key = item.selected_path_ids
        outcome_counts[key] = outcome_counts.get(key, 0) + 1
        current = representatives.get(key)
        if current is None or result_order(item) > result_order(current):
            representatives[key] = item

    result = fit_weights(
        tuple(cells), tuple(measurements), seed=seed,
        random_sample_count=samples, evaluation_callback=record,
    )
    ordered_representatives = sorted(
        representatives.values(),
        key=lambda item: (
            tuple(item.selected_path_ids),
            tuple(item.weights.as_tuple()),
        ),
    )
    with search_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for item in ordered_representatives:
            writer.writerow(
                {
                    "weights_json": json.dumps(item.weights.as_mapping(), sort_keys=True, separators=(",", ":")),
                    "selected_path_ids_json": json.dumps(dict(item.selected_path_ids), sort_keys=True, separators=(",", ":")),
                    "cell_speedups_json": json.dumps(dict(item.cell_speedups), sort_keys=True, separators=(",", ":")),
                    "geometric_mean_speedup": item.geometric_mean_speedup,
                    "minimum_cell_speedup": item.minimum_cell_speedup,
                    "improved_cell_count": item.improved_cell_count,
                    "equivalent_weight_vector_count": outcome_counts[item.selected_path_ids],
                }
            )
    profile = {
        "schema_version": "physical_speedup_fit_v1",
        "score_id": COST_MODEL_ID,
        "source_sha": dataset["source_sha"],
        "source_sha_semantics": "candidate_generation_source_sha",
        "candidate_generation_source_sha": dataset["source_sha"],
        "physical_execution_source_sha": physical_execution_source,
        "reporting_tool_source_sha": _source_sha(),
        "candidate_set_sha256": candidate_set_sha,
        "weights": result.weights.as_mapping(),
        "feature_model": asdict(result.model),
        "training_cell_ids": [cell.cell_id for cell in cells],
        "selected_path_ids": dict(result.selected_path_ids),
        "cell_speedups": dict(result.cell_speedups),
        "geometric_mean_speedup": result.geometric_mean_speedup,
        "minimum_cell_speedup": result.minimum_cell_speedup,
        "improved_cell_count": result.improved_cell_count,
        "weight_search_seed": seed,
        "random_weight_samples": samples,
        "evaluated_weight_vectors": result.evaluated_weight_vectors,
        "weight_search_candidate_rows": len(ordered_representatives),
        "weight_search_candidates_semantics": (
            "best_lexicographic_weight_vector_per_distinct_training_path_selection"
        ),
        "normalization": "log((candidate+1)/(greedy+1))",
        "primary_objective": "geometric_mean_greedy_relative_speedup",
    }
    (output_dir / "physical_speedup_fit_v1.json").write_bytes(_canonical_bytes(profile))
    (output_dir / "weight_search_summary.json").write_bytes(_canonical_bytes(profile))
    return result


def _model_from_profile(profile: dict[str, Any]) -> FeatureModelDecision:
    value = profile["feature_model"]
    return FeatureModelDecision(
        mode=value["mode"],
        active_features=tuple(value["active_features"]),
        zero_range_features=tuple(value["zero_range_features"]),
        correlated_pairs=tuple(tuple(pair) for pair in value["correlated_pairs"]),
        matrix_rank=int(value["matrix_rank"]),
        rank_tolerance=float(value["rank_tolerance"]),
        reason=str(value["reason"]),
    )


def evaluate_frozen_profile(
    candidate_path: Path,
    profile_path: Path,
    output_path: Path,
    *,
    split: str,
) -> dict[str, Any]:
    """Select held-out paths without consulting physical timing."""

    if split not in {"validation", "test"}:
        raise ValueError("frozen-profile evaluation is limited to validation or test")
    dataset = json.loads(candidate_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    candidate_sha = _sha256_bytes(_canonical_bytes(dataset))
    if profile["candidate_set_sha256"] != candidate_sha:
        raise ValueError("fitted profile does not match candidate dataset")
    if profile["source_sha"] != dataset["source_sha"]:
        raise ValueError("fitted profile source does not match candidate dataset")
    weights = WeightVector.from_values(profile["weights"])
    model = _model_from_profile(profile)
    selections = []
    for circuit in dataset["circuits"]:
        if circuit["split"] != split:
            continue
        candidates = tuple(_candidate_from_record(item) for item in circuit["candidates"])
        for topology_id, _ in _topologies(load_config()):
            feasible = tuple(item for item in candidates if item.feasible_for(topology_id))
            greedy = next(item for item in feasible if item.is_greedy)
            flop_best = min(
                feasible,
                key=lambda item: (item.conventional.flops, item.path_id),
            )
            selected = select_best_candidate(
                feasible,
                topology_id,
                weights,
                model=model,
                greedy_path_id=greedy.path_id,
            )
            selections.append(
                {
                    "circuit_id": circuit["circuit_id"],
                    "split": split,
                    "topology_id": topology_id,
                    "greedy_path_id": greedy.path_id,
                    "minimum_flops_path_id": flop_best.path_id,
                    "upmem_selected_path_id": selected.path_id,
                    "upmem_score": score_features(
                        selected.raw_for(topology_id),
                        greedy.raw_for(topology_id),
                        weights,
                        model=model,
                    ),
                    "explanation": [
                        row.as_mapping()
                        for row in explain_score(
                            selected.raw_for(topology_id),
                            greedy.raw_for(topology_id),
                            weights,
                            model=model,
                        )
                    ],
                }
            )
    if not selections:
        raise ValueError(f"candidate dataset contains no {split} circuits")
    result = {
        "schema_version": "upmem_path_frozen_selection_v1",
        "score_id": COST_MODEL_ID,
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": candidate_sha,
        "fitted_profile_sha256": _sha256_bytes(_canonical_bytes(profile)),
        "split": split,
        "timing_used_for_selection": False,
        "weights": weights.as_mapping(),
        "feature_model": asdict(model),
        "selections": selections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--check", action="store_true")
    generate_parser.add_argument("--circuit-id", action="append", default=[])
    generate_parser.add_argument("--candidate-part-index", type=int)
    generate_parser.add_argument("--candidate-part-count", type=int)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    merge_parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    merge_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--candidate-paths", type=Path, required=True)
    fit_parser.add_argument("--calibration-set", type=Path, required=True)
    fit_parser.add_argument("--runtime-table", type=Path, required=True)
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    fit_parser.add_argument("--samples", type=int, default=100_000)
    fit_parser.add_argument("--seed", type=int, default=20260903)
    extract_parser = subparsers.add_parser(
        "extract-calibration", aliases=("extract",)
    )
    extract_parser.add_argument("--raw-dir", type=Path, required=True)
    extract_parser.add_argument("--candidate-paths", type=Path, required=True)
    extract_parser.add_argument("--calibration-set", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--candidate-paths", type=Path, required=True)
    evaluate_parser.add_argument("--profile", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        partition = None
        if args.candidate_part_index is not None or args.candidate_part_count is not None:
            if args.candidate_part_index is None or args.candidate_part_count is None:
                raise ValueError("candidate partition requires both index and count")
            if not 0 <= args.candidate_part_index < args.candidate_part_count:
                raise ValueError("candidate partition index is outside its count")
            partition = (args.candidate_part_index, args.candidate_part_count)
        timings = generate(
            args.config,
            args.output_dir,
            check=args.check,
            circuit_ids=tuple(args.circuit_id),
            candidate_partition=partition,
        )
        print(json.dumps(timings, sort_keys=True))
    elif args.command == "merge":
        timings = merge_shards(
            args.config, tuple(args.shard_dir), args.output_dir
        )
        print(json.dumps(timings, sort_keys=True))
    elif args.command == "fit":
        result = fit(
            args.candidate_paths, args.calibration_set, args.runtime_table,
            args.output_dir, samples=args.samples, seed=args.seed,
        )
        print(json.dumps({"weights": result.weights.as_mapping(), "geometric_mean_speedup": result.geometric_mean_speedup}, sort_keys=True))
    elif args.command in {"extract-calibration", "extract"}:
        result = extract_calibration(
            args.raw_dir,
            args.candidate_paths,
            args.calibration_set,
            args.output_dir,
        )
        print(json.dumps({"observation_count": result["sample_count"]}, sort_keys=True))
    else:
        result = evaluate_frozen_profile(
            args.candidate_paths,
            args.profile,
            args.output,
            split=args.split,
        )
        print(json.dumps({"selection_count": len(result["selections"])}))


if __name__ == "__main__":
    main()
