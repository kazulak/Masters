#!/usr/bin/env python3
"""Generate and fit the finite UPMEM-aware path-heuristic dataset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import multiprocessing
from pathlib import Path
import queue as queue_module
import resource
import subprocess
import time
from typing import Any

from quantum_bench.circuits import builtin_circuit
from quantum_bench.evidence import problem_id, tensor_network_structure_id
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
)


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
        except Exception as exc:
            feasible = False
            reason = f"{type(exc).__name__}:{exc}"
        if feasible and plan is not None and facts is not None:
            feature_pairs.append((topology_id, facts.raw))
            feasible_topologies.append(topology_id)
            physical_id = physical_plan_id(plan)
            fact_values = facts.as_mapping()
        else:
            physical_id = None
            fact_values = {}
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


def _serialize_candidate_worker(
    circuit_id: str,
    split: str,
    definition: dict[str, Any],
    item: dict[str, Any],
    config: dict[str, Any],
    connection: Any,
) -> None:
    try:
        worker_limit = int(
            config["candidate_generation"][
                "physical_lowering_worker_address_space_bytes"
            ]
        )
        resource.setrlimit(resource.RLIMIT_AS, (worker_limit, worker_limit))
        circuit = builtin_circuit(str(definition["name"]), dict(definition["parameters"]))
        network, inputs = lower_tensor_network(make_simulation_job(circuit))
        record, rows, candidate = _serialize_candidate(
            circuit_id=circuit_id,
            split=split,
            network=network,
            inputs=inputs,
            item=item,
            config=config,
        )
        result = (record, rows, candidate, None)
    except BaseException as exc:
        result = (None, None, None, f"{type(exc).__name__}:{exc}")
    try:
        connection.send(result)
    finally:
        connection.close()


def _isolated_serialized_candidate(
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
    logical_id = contraction_dag_hash(dag)
    memory_estimate = _host_memory_estimate(inputs, dag)
    estimated_work_units = _estimated_work_unit_count(dag)
    work_unit_limit = int(config["candidate_generation"]["maximum_planned_work_units"])
    memory_limit = int(config["host_memory_admission_bytes"])
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
    if memory_estimate > memory_limit:
        return _infeasible_candidate_record(
            circuit_id=circuit_id,
            split=split,
            item=item,
            config=config,
            reason="host_memory_estimate_exceeds_preregistered_bound",
            conventional=conventional,
            logical_plan_id=logical_id,
            host_memory_estimate_bytes=memory_estimate,
            estimated_work_unit_count=estimated_work_units,
        )
    # Candidate lowering calls NumPy/BLAS after cotengra has initialized native
    # worker state in the parent. Forking at that point can inherit locked
    # runtime state and deadlock before deterministic admission is emitted.
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_serialize_candidate_worker,
        args=(circuit_id, split, definition, item, config, send),
    )
    process.start()
    send.close()
    timeout_s = float(config["candidate_generation"]["physical_lowering_timeout_s"])
    deadline = time.monotonic() + timeout_s
    result = None
    while result is None:
        if receive.poll():
            result = receive.recv()
        else:
            if not process.is_alive():
                process.join()
                receive.close()
                raise RuntimeError(
                    f"candidate lowering {item['candidate_path_id']} returned no result: "
                    f"{process.exitcode}"
                )
            if time.monotonic() >= deadline:
                process.kill()
                process.join()
                receive.close()
                raise RuntimeError(
                    "candidate lowering exceeded the generation guard; "
                    "candidate membership was not changed: "
                    f"{item['candidate_path_id']} ({timeout_s:g}s)"
                )
            time.sleep(0.01)
    record, rows, candidate, error = result
    process.join()
    receive.close()
    if process.exitcode != 0 or error is not None:
        raise RuntimeError(
            f"candidate lowering {item['candidate_path_id']} failed: "
            f"{error or process.exitcode}"
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
            "topologies": topology_records,
        },
        rows,
        None,
    )


def build_dataset(config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, float]]:
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
        total_generation_s += timing["candidate_generation_s"]
        candidate_records = []
        path_candidates = []
        feature_started = time.perf_counter()
        for item in raw_candidates:
            record, rows, candidate = _isolated_serialized_candidate(
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


def generate(
    config_path: Path,
    output_dir: Path,
    *,
    check: bool = False,
    circuit_ids: tuple[str, ...] = (),
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
    dataset, features, rankings, calibration, timings = build_dataset(config)
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
    circuits: dict[str, dict[str, Any]] = {}
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
            if circuit_id in circuits:
                raise ValueError(f"duplicate circuit shard: {circuit_id}")
            circuits[circuit_id] = circuit
        with (shard_dir / "candidate_features.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            feature_rows.extend(csv.DictReader(stream))
        with (shard_dir / "candidate_rankings.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            ranking_rows.extend(csv.DictReader(stream))
        calibration_cells.extend(calibration["cells"])
        shard_timing = json.loads((shard_dir / "planning_timing.json").read_text())
        for key in timings:
            timings[key] += float(shard_timing[key])
    if set(circuits) != set(expected_order):
        raise ValueError("candidate shards do not cover the frozen circuit set exactly")
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
    feature_rows.sort(
        key=lambda row: (
            circuit_order[row["circuit_id"]],
            candidate_order[(row["circuit_id"], row["candidate_path_id"])],
            topology_order[row["topology_id"]],
        )
    )
    ranking_rows.sort(
        key=lambda row: (
            circuit_order[row["circuit_id"]],
            topology_order[row["topology_id"]],
            int(row["equal_weight_rank"]),
        )
    )
    calibration_cells.sort(
        key=lambda item: (
            circuit_order[item["circuit_id"]],
            topology_order[item["topology_id"]],
        )
    )
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
            if row.get("source_sha") != dataset["source_sha"]:
                raise ValueError("calibration runtime source does not match candidate dataset")
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
                    source_sha=str(row["source_sha"]),
                    timing_scope=str(row["timing_scope"]),
                    status=str(row["status"]),
                    observation_id=str(block),
                )
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    search_path = output_dir / "weight_search_candidates.csv"
    columns = (
        "weights_json", "selected_path_ids_json", "cell_speedups_json",
        "geometric_mean_speedup", "minimum_cell_speedup", "improved_cell_count",
    )
    with search_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()

        def record(result: WeightFitResult) -> None:
            writer.writerow(
                {
                    "weights_json": json.dumps(result.weights.as_mapping(), sort_keys=True, separators=(",", ":")),
                    "selected_path_ids_json": json.dumps(dict(result.selected_path_ids), sort_keys=True, separators=(",", ":")),
                    "cell_speedups_json": json.dumps(dict(result.cell_speedups), sort_keys=True, separators=(",", ":")),
                    "geometric_mean_speedup": result.geometric_mean_speedup,
                    "minimum_cell_speedup": result.minimum_cell_speedup,
                    "improved_cell_count": result.improved_cell_count,
                }
            )

        result = fit_weights(
            tuple(cells), tuple(measurements), seed=seed,
            random_sample_count=samples, evaluation_callback=record,
        )
    profile = {
        "schema_version": "physical_speedup_fit_v1",
        "score_id": COST_MODEL_ID,
        "source_sha": dataset["source_sha"],
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
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--candidate-paths", type=Path, required=True)
    evaluate_parser.add_argument("--profile", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        timings = generate(
            args.config,
            args.output_dir,
            check=args.check,
            circuit_ids=tuple(args.circuit_id),
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
