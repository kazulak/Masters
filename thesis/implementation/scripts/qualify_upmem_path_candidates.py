#!/usr/bin/env python3
"""Qualify frozen path candidates and prepare canonical experiment configs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quantum_bench.circuits import builtin_circuit
from quantum_bench.cpu import run_complex128_reference, run_cpu_once
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import make_simulation_job
from quantum_bench.planning import plan_cotengra, plan_opt_einsum
from quantum_bench.upmem.path_heuristic import path_id


ROOT = Path(__file__).resolve().parents[1]
FLOAT32 = "split_complex_float32_v1"


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _candidate_set_sha256(dataset: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(dataset)).hexdigest()


def _candidate_map(dataset: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (circuit["circuit_id"], candidate["candidate_path_id"]): candidate
        for circuit in dataset["circuits"]
        for candidate in circuit["candidates"]
    }


def _circuit_map(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {circuit["circuit_id"]: circuit for circuit in dataset["circuits"]}


def _planner_config(candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate["source_kind"] == "opt_einsum_greedy":
        return {"engine": "opt_einsum", "mode": "greedy"}
    if candidate["source_kind"] == "cotengra_one_trial":
        return {
            "engine": "cotengra",
            "mode": "greedy",
            "max_repeats": 1,
            "seed": int(candidate["source_seed"]),
        }
    raise ValueError(f"unsupported candidate source: {candidate['source_kind']}")


def _regenerate(circuit: dict[str, Any], candidate: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    definition = circuit["circuit"]
    spec = builtin_circuit(definition["name"], dict(definition["parameters"]))
    network, inputs = lower_tensor_network(make_simulation_job(spec))
    planner = _planner_config(candidate)
    if planner["engine"] == "opt_einsum":
        path, provenance = plan_opt_einsum(network, optimize="greedy")
    else:
        path, provenance = plan_cotengra(
            network,
            methods="greedy",
            max_repeats=1,
            seed=planner["seed"],
        )
    identifier = path_id(path, circuit_id=circuit["circuit_id"])
    if identifier != candidate["candidate_path_id"]:
        raise ValueError(
            f"candidate regeneration mismatch for {circuit['circuit_id']}/{candidate['candidate_path_id']}"
        )
    if provenance["planner_config_hash"] != candidate["planner_config_hash"]:
        raise ValueError(
            f"planner provenance mismatch for {circuit['circuit_id']}/{identifier}"
        )
    dag = build_contraction_dag(network, path)
    if contraction_dag_hash(dag) != candidate["logical_plan_id"]:
        raise ValueError(
            f"logical-plan mismatch for {circuit['circuit_id']}/{identifier}"
        )
    return dag, inputs


def _ranking_best(rankings_path: Path) -> dict[tuple[str, str], str]:
    result = {}
    with rankings_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["equal_weight_rank"]) == 1:
                result[(row["circuit_id"], row["topology_id"])] = row["candidate_path_id"]
    return result


def _representative_ids_for_topology(
    circuit: dict[str, Any], rankings: dict[tuple[str, str], str], topology_id: str
) -> tuple[str, ...]:
    candidates = tuple(
        candidate
        for candidate in circuit["candidates"]
        if candidate.get("conventional_features") is not None
        and next(
            item for item in candidate["topologies"]
            if item["topology_id"] == topology_id
        ).get("feasible") is True
    )
    greedy = next(candidate for candidate in candidates if candidate["is_greedy"])
    selected = {
        greedy["candidate_path_id"],
        min(
            candidates,
            key=lambda item: (
                item["conventional_features"]["flops"], item["candidate_path_id"]
            ),
        )["candidate_path_id"],
    }
    selected.add(rankings[(circuit["circuit_id"], topology_id)])
    selected.add(
        min(
            candidates,
            key=lambda candidate: (
                next(
                    item for item in candidate["topologies"]
                    if item["topology_id"] == topology_id
                )["features"]["B_host_dpu"],
                candidate["candidate_path_id"],
            ),
        )["candidate_path_id"]
    )
    return tuple(sorted(selected))


def _representative_ids(
    circuit: dict[str, Any], rankings: dict[tuple[str, str], str]
) -> tuple[str, ...]:
    return tuple(sorted({
        candidate_id
        for topology_id in ("1dpu_t8", "4dpu_t8")
        for candidate_id in _representative_ids_for_topology(
            circuit, rankings, topology_id
        )
    }))


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _cpu_reference_errors(
    actual: np.ndarray, reference: np.ndarray
) -> tuple[float | None, float | None, float | None, bool]:
    if actual.shape != reference.shape:
        return None, None, None, False
    difference = np.asarray(actual, dtype=np.complex128) - reference
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = float(np.linalg.norm(reference.reshape(-1)))
    relative_l2 = (
        float(np.linalg.norm(difference.reshape(-1))) / denominator
        if denominator
        else float(np.linalg.norm(difference.reshape(-1)))
    )
    norm_drift = abs(
        float(np.linalg.norm(np.asarray(actual, dtype=np.complex128).reshape(-1)))
        - denominator
    )
    passed = bool(
        np.allclose(actual, reference, rtol=1.0e-5, atol=1.0e-5)
    )
    return maximum, relative_l2, norm_drift, passed


def qualify_frozen_selection(
    dataset_path: Path,
    selection_path: Path,
    output_path: Path,
    *,
    split: str,
) -> dict[str, Any]:
    """CPU-qualify every unique candidate named by a frozen selection artifact."""

    dataset = _load(dataset_path)
    selection = _load(selection_path)
    dataset_hash = _candidate_set_sha256(dataset)
    candidates = _candidate_map(dataset)
    circuits = _circuit_map(dataset)
    selected, selection_roles = _evaluation_selection(
        dataset=dataset,
        dataset_hash=dataset_hash,
        candidate_map=candidates,
        circuit_map=circuits,
        selection_path=selection_path,
        split=split,
    )

    unique: dict[tuple[str, str], dict[str, set[str]]] = {}
    for circuit_id, topology_id, candidate_id in selected:
        entry = unique.setdefault(
            (circuit_id, candidate_id),
            {"topology_ids": set(), "roles": set()},
        )
        entry["topology_ids"].add(topology_id)
        entry["roles"].update(
            selection_roles[(circuit_id, topology_id, candidate_id)]
        )

    rows: list[dict[str, Any]] = []
    references: dict[str, tuple[np.ndarray, str]] = {}
    for circuit_id, candidate_id in sorted(unique):
        circuit = circuits[circuit_id]
        candidate = candidates[(circuit_id, candidate_id)]
        if circuit_id not in references:
            definition = circuit["circuit"]
            spec = builtin_circuit(definition["name"], dict(definition["parameters"]))
            network, reference_inputs = lower_tensor_network(make_simulation_job(spec))
            greedy_path, _ = plan_opt_einsum(network, optimize="greedy")
            reference = np.asarray(
                run_complex128_reference(
                    build_contraction_dag(network, greedy_path), reference_inputs
                ),
                dtype=np.complex128,
            )
            references[circuit_id] = (reference, _array_sha256(reference))
        reference, reference_hash = references[circuit_id]
        row: dict[str, Any] = {
            "circuit_id": circuit_id,
            "split": split,
            "candidate_path_id": candidate_id,
            "roles": sorted(unique[(circuit_id, candidate_id)]["roles"]),
            "topology_ids": sorted(
                unique[(circuit_id, candidate_id)]["topology_ids"]
            ),
            "logical_plan_id": candidate["logical_plan_id"],
            "reference_output_sha256": reference_hash,
            "output_sha256": None,
            "max_absolute_error": None,
            "relative_l2_error": None,
            "norm_drift": None,
            "errors": {
                "max_absolute_error": None,
                "relative_l2_error": None,
                "norm_drift": None,
            },
            "error": None,
            "passed": False,
        }
        try:
            dag, inputs = _regenerate(circuit, candidate)
            actual = np.asarray(run_cpu_once(dag, inputs, FLOAT32).output)
            row["output_sha256"] = _array_sha256(actual)
            maximum, relative_l2, norm_drift, passed = _cpu_reference_errors(
                actual, reference
            )
            row["max_absolute_error"] = maximum
            row["relative_l2_error"] = relative_l2
            row["norm_drift"] = norm_drift
            row["errors"] = {
                "max_absolute_error": maximum,
                "relative_l2_error": relative_l2,
                "norm_drift": norm_drift,
            }
            row["passed"] = passed
            if actual.shape != reference.shape:
                row["error"] = (
                    f"output shape mismatch: {actual.shape!r} != {reference.shape!r}"
                )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    result = {
        "schema_version": "upmem_path_frozen_selection_cpu_qualification_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": dataset_hash,
        "selection_sha256": hashlib.sha256(_canonical_bytes(selection)).hexdigest(),
        "selection_schema_version": selection["schema_version"],
        "split": split,
        "numeric_policy": FLOAT32,
        "reference_numeric_policy": "complex128",
        "rtol": 1.0e-5,
        "atol": 1.0e-5,
        "selection_contract_passed": True,
        "selected_cell_count": len(
            {(circuit, topology) for circuit, topology, _ in selected}
        ),
        "qualified_candidate_count": len(rows),
        "all_passed": bool(rows) and all(row["passed"] for row in rows),
        "candidates": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(result))
    return result


def qualify_cpu(
    dataset_path: Path,
    rankings_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    candidates = _candidate_map(dataset)
    rankings = _ranking_best(rankings_path)
    rows = []
    for circuit in dataset["circuits"]:
        definition = circuit["circuit"]
        spec = builtin_circuit(definition["name"], dict(definition["parameters"]))
        network, reference_inputs = lower_tensor_network(make_simulation_job(spec))
        greedy_path, _ = plan_opt_einsum(network, optimize="greedy")
        reference = run_complex128_reference(
            build_contraction_dag(network, greedy_path), reference_inputs
        )
        for candidate_id in _representative_ids(circuit, rankings):
            candidate = candidates[(circuit["circuit_id"], candidate_id)]
            dag, inputs = _regenerate(circuit, candidate)
            actual = run_cpu_once(dag, inputs, FLOAT32)
            actual_output = np.asarray(actual.output)
            maximum, relative_l2, _, passed = _cpu_reference_errors(
                actual_output, np.asarray(reference, dtype=np.complex128)
            )
            if not passed:
                raise ValueError(
                    f"CPU candidate validation failed for {circuit['circuit_id']}/{candidate_id}"
                )
            output_bytes = np.ascontiguousarray(actual.output).tobytes()
            rows.append(
                {
                    "circuit_id": circuit["circuit_id"],
                    "split": circuit["split"],
                    "candidate_path_id": candidate_id,
                    "logical_plan_id": contraction_dag_hash(dag),
                    "max_absolute_error": maximum,
                    "relative_l2_error": relative_l2,
                    "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                    "passed": True,
                }
            )
    record = {
        "schema_version": "upmem_path_cpu_candidate_qualification_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": _candidate_set_sha256(dataset),
        "numeric_policy": FLOAT32,
        "qualified_candidate_count": len(rows),
        "all_passed": True,
        "candidates": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_canonical_bytes(record))
    return record


def _collection(*, warmups: int, measurements: int, seed: int) -> dict[str, Any]:
    return {
        "claim_policy": "diagnostic_v1",
        "base_seed": seed,
        "warmup_blocks": warmups,
        "measurement_blocks": measurements,
        "session_policy": "fresh_session_per_attempt_v1",
        "block_cooldown_s": 0.0,
        "machine_policy": {
            "machine_exclusivity": {"mode": "observed_v1"},
            "cpu_governor": {"mode": "observed_v1"},
            "affinity": {"mode": "exact_required_v1", "expected_cpus": [0]},
            "numa_policy": {"mode": "observed_v1"},
            "background_load": {
                "mode": "observed_v1",
                "max_load1_per_online_cpu": None,
            },
        },
    }


def _route(topology_id: str, *, simulator: bool) -> dict[str, Any]:
    dpus = 1 if topology_id == "1dpu_t8" else 4
    options = {
        "dpu_count": dpus,
        "rank_count": 1,
        "tasklets_per_dpu": 8,
        "session_root": f"../runs/upmem_sessions/path_heuristic_{'sdk' if simulator else 'physical'}_{topology_id}",
        "host_binary": "../native/upmem/runtime/bin/host_upmem_execution_plan_v4_t8",
        "dpu_binary": "../native/upmem/runtime/bin/dpu_gemm_tile_v4_t8",
        "initialization_binary": "../native/upmem/runtime/bin/dpu_simplepim_management_init_t8",
    }
    if not simulator:
        options["rank_paths"] = ["/dev/dpu_rank1"]
    return {
        "executor": "upmem_sdk_simulator" if simulator else "upmem_physical",
        "numeric_policy": FLOAT32,
        "options": options,
    }


_EVALUATION_TOPOLOGIES = ("1dpu_t8", "4dpu_t8")


def _topology_record(candidate: dict[str, Any], topology_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in candidate.get("topologies", [])
        if item.get("topology_id") == topology_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"candidate {candidate.get('candidate_path_id')} must have exactly "
            f"one {topology_id} topology record"
        )
    return matches[0]


def _require_evaluation_candidate(
    candidate: dict[str, Any], circuit_id: str, topology_id: str
) -> dict[str, Any]:
    topology = _topology_record(candidate, topology_id)
    if topology.get("feasible") is not True:
        raise ValueError(
            f"evaluation candidate is infeasible for {circuit_id}/{topology_id}/"
            f"{candidate.get('candidate_path_id')}"
        )
    admission = topology.get("resource_admission")
    if not isinstance(admission, dict) or admission.get(
        "collection_resource_admission_passed"
    ) is not True:
        raise ValueError(
            f"evaluation candidate lacks passed resource admission for "
            f"{circuit_id}/{topology_id}/{candidate.get('candidate_path_id')}"
        )
    if not isinstance(topology.get("physical_plan_id"), str) or not topology[
        "physical_plan_id"
    ]:
        raise ValueError(
            f"evaluation candidate lacks physical-plan identity for "
            f"{circuit_id}/{topology_id}/{candidate.get('candidate_path_id')}"
        )
    return topology


def _evaluation_selection(
    *,
    dataset: dict[str, Any],
    dataset_hash: str,
    candidate_map: dict[tuple[str, str], dict[str, Any]],
    circuit_map: dict[str, dict[str, Any]],
    selection_path: Path,
    split: str,
) -> tuple[list[tuple[str, str, str]], dict[tuple[str, str, str], tuple[str, ...]]]:
    if split not in {"validation", "test"}:
        raise ValueError("evaluation split must be validation or test")
    selection = _load(selection_path)
    if selection.get("schema_version") != "upmem_path_frozen_selection_v1":
        raise ValueError("evaluation selection has an invalid schema")
    if selection.get("candidate_set_sha256") != dataset_hash:
        raise ValueError("evaluation candidate-set identity does not match dataset")
    if selection.get("source_sha") != dataset.get("source_sha"):
        raise ValueError("evaluation source identity does not match dataset")
    if selection.get("split") != split:
        raise ValueError("evaluation selection split does not match requested split")
    if selection.get("timing_used_for_selection") is not False:
        raise ValueError("evaluation selection must be timing-independent")
    rows = selection.get("selections")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evaluation selection must contain selections")

    expected_circuits = {
        circuit_id
        for circuit_id, circuit in circuit_map.items()
        if circuit.get("split") == split
    }
    if not expected_circuits:
        raise ValueError(f"candidate dataset contains no {split} circuits")
    expected_keys = {
        (circuit_id, topology_id)
        for circuit_id in expected_circuits
        for topology_id in _EVALUATION_TOPOLOGIES
    }
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("evaluation selection rows must be objects")
        if row.get("split") != split:
            raise ValueError("evaluation selection contains another split")
        circuit_id = row.get("circuit_id")
        topology_id = row.get("topology_id")
        if not isinstance(circuit_id, str) or not isinstance(topology_id, str):
            raise ValueError("evaluation selection cell identities must be strings")
        key = (circuit_id, topology_id)
        if key in rows_by_key:
            raise ValueError(
                f"evaluation selection contains duplicate cell {circuit_id}/{topology_id}"
            )
        if key not in expected_keys:
            raise ValueError(
                f"evaluation selection contains an unexpected cell {circuit_id}/{topology_id}"
            )
        rows_by_key[key] = row
    if set(rows_by_key) != expected_keys:
        missing = sorted(expected_keys - set(rows_by_key))
        raise ValueError(f"evaluation selection is missing cells: {missing}")

    selected: set[tuple[str, str, str]] = set()
    roles: dict[tuple[str, str, str], set[str]] = {}
    for (circuit_id, topology_id), row in sorted(rows_by_key.items()):
        circuit_candidates = [
            candidate
            for (candidate_circuit_id, _), candidate in candidate_map.items()
            if candidate_circuit_id == circuit_id
        ]
        feasible = []
        for candidate in circuit_candidates:
            topology = _topology_record(candidate, topology_id)
            admission = topology.get("resource_admission")
            if (
                topology.get("feasible") is True
                and isinstance(admission, dict)
                and admission.get("collection_resource_admission_passed") is True
            ):
                feasible.append(candidate)
        if not feasible:
            raise ValueError(f"no admitted candidates for {circuit_id}/{topology_id}")
        greedy_candidates = [
            candidate for candidate in feasible if candidate.get("is_greedy") is True
        ]
        if len(greedy_candidates) != 1:
            raise ValueError(
                f"evaluation requires exactly one feasible greedy candidate for "
                f"{circuit_id}/{topology_id}"
            )
        expected_greedy = greedy_candidates[0]["candidate_path_id"]
        expected_flops = min(
            feasible,
            key=lambda candidate: (
                candidate["conventional_features"]["flops"],
                candidate["candidate_path_id"],
            ),
        )["candidate_path_id"]
        expected_roles = {
            "greedy_path_id": expected_greedy,
            "minimum_flops_path_id": expected_flops,
        }
        for field, expected in expected_roles.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"evaluation {field} is not deterministic for "
                    f"{circuit_id}/{topology_id}"
                )
        upmem_selected = row.get("upmem_selected_path_id")
        if not isinstance(upmem_selected, str) or not upmem_selected:
            raise ValueError(
                f"evaluation selection has no UPMEM-selected path for "
                f"{circuit_id}/{topology_id}"
            )
        role_values = (*expected_roles.items(), ("upmem_selected_path_id", upmem_selected))
        for field, candidate_id in role_values:
            candidate = candidate_map.get((circuit_id, candidate_id))
            if candidate is None:
                raise ValueError(
                    f"evaluation selection references unknown candidate "
                    f"{circuit_id}/{candidate_id}"
                )
            _require_evaluation_candidate(candidate, circuit_id, topology_id)
            selected_key = (circuit_id, topology_id, candidate_id)
            selected.add(selected_key)
            roles.setdefault(selected_key, set()).add(field.removesuffix("_path_id"))
    return sorted(selected), {
        key: tuple(sorted(value)) for key, value in roles.items()
    }


def prepare_config(
    *,
    dataset_path: Path,
    output_path: Path,
    mode: str,
    calibration_path: Path | None = None,
    rankings_path: Path | None = None,
    selection_path: Path | None = None,
    split: str | None = None,
    execution_target: str = "physical",
) -> dict[str, Any]:
    if execution_target not in {"physical", "sdk"}:
        raise ValueError("execution target must be physical or sdk")
    dataset = _load(dataset_path)
    dataset_hash = _candidate_set_sha256(dataset)
    candidate_map = _candidate_map(dataset)
    circuit_map = _circuit_map(dataset)
    selected: list[tuple[str, str, str]] = []
    selection_roles: dict[tuple[str, str, str], tuple[str, ...]] = {}
    selection_provenance: list[tuple[str, str, str]] = []
    if mode == "calibration":
        if calibration_path is None:
            raise ValueError("calibration mode requires a calibration set")
        calibration = _load(calibration_path)
        if calibration.get("candidate_set_sha256") != dataset_hash:
            raise ValueError("calibration candidate-set identity does not match dataset")
        if calibration.get("source_sha") != dataset.get("source_sha"):
            raise ValueError("calibration source identity does not match dataset")
        for cell in calibration["cells"]:
            for candidate_id in cell["candidate_path_ids"]:
                selected.append((cell["circuit_id"], cell["topology_id"], candidate_id))
        warmups, measurements, seed, simulator = 1, 3, 20260910, False
        topology_ids = ("1dpu_t8", "4dpu_t8")
    elif mode == "sdk":
        if rankings_path is None:
            raise ValueError("sdk mode requires rankings")
        rankings = _ranking_best(rankings_path)
        topology_ids = ("1dpu_t8",)
        for circuit in dataset["circuits"]:
            for topology_id in topology_ids:
                ids = _representative_ids_for_topology(
                    circuit, rankings, topology_id
                )
                selected.extend((circuit["circuit_id"], topology_id, path) for path in ids)
        warmups, measurements, seed, simulator = 0, 1, 20260909, True
    elif mode == "evaluation":
        if selection_path is None:
            raise ValueError("evaluation mode requires a frozen selection")
        if split is None:
            raise ValueError("evaluation mode requires validation or test split")
        topology_ids = _EVALUATION_TOPOLOGIES
        selected, selection_roles = _evaluation_selection(
            dataset=dataset,
            dataset_hash=dataset_hash,
            candidate_map=candidate_map,
            circuit_map=circuit_map,
            selection_path=selection_path,
            split=split,
        )
        selection_provenance = list(selected)
        if execution_target == "sdk":
            selected = sorted({
                (circuit_id, "1dpu_t8", candidate_id)
                for circuit_id, _topology_id, candidate_id in selected
            })
            topology_ids = ("1dpu_t8",)
            warmups, measurements, seed, simulator = 0, 1, 20260912, True
        else:
            warmups, measurements, seed, simulator = 1, 5, 20260911, False
    else:
        raise ValueError("mode must be sdk, calibration, or evaluation")
    if mode != "evaluation" and execution_target != "physical":
        raise ValueError(
            "execution target is only configurable for evaluation mode"
        )
    selected = sorted(set(selected))
    cases = {
        circuit_id: {"circuit": {**circuit_map[circuit_id]["circuit"], "path": None}}
        for circuit_id in sorted({item[0] for item in selected})
    }
    plans = {}
    matrix = []
    for circuit_id, topology_id, candidate_id in selected:
        candidate = candidate_map[(circuit_id, candidate_id)]
        topology = _topology_record(candidate, topology_id)
        if topology.get("feasible") is not True:
            raise ValueError(
                f"selected candidate is infeasible for {circuit_id}/{topology_id}/{candidate_id}"
            )
        _regenerate(circuit_map[circuit_id], candidate)
        plan_id = f"path_{candidate_id}"
        plans[plan_id] = {"planner": _planner_config(candidate), "slicing": None}
        matrix.append(
            {
                "case_id": circuit_id,
                "plan_id": plan_id,
                "route_ids": [f"upmem_{topology_id}"],
            }
        )
    if mode == "evaluation":
        grouped: dict[tuple[str, str], set[str]] = {}
        for circuit_id, topology_id, candidate_id in selected:
            grouped.setdefault((circuit_id, candidate_id), set()).add(topology_id)
        topology_order = {
            topology_id: index for index, topology_id in enumerate(topology_ids)
        }
        matrix = [
            {
                "case_id": circuit_id,
                "plan_id": f"path_{candidate_id}",
                "route_ids": sorted(
                    route_ids, key=lambda route_id: topology_order[route_id]
                ),
            }
            for (circuit_id, candidate_id), route_ids in sorted(grouped.items())
        ]
    config = {
        "schema_version": "tn_benchmark_v3",
        "experiment_id": (
            f"upmem-path-heuristic-evaluation-{split}"
            f"{'-sdk' if execution_target == 'sdk' else ''}-v1"
            if mode == "evaluation"
            else f"upmem-path-heuristic-{mode}-v1"
        ),
        "defaults": {"timeout_s": 120.0},
        "collection": _collection(warmups=warmups, measurements=measurements, seed=seed),
        "cases": cases,
        "plans": plans,
        "routes": {
            topology_id: _route(topology_id, simulator=simulator)
            for topology_id in topology_ids
        },
        "matrix": matrix,
    }
    # Matrix route names use the canonical mapping keys above.
    for item in config["matrix"]:
        item["route_ids"] = [
            route_id.removeprefix("upmem_") for route_id in item["route_ids"]
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    provenance = {
        "schema_version": "upmem_path_experiment_provenance_v1",
        "source_sha": dataset["source_sha"],
        "candidate_set_sha256": dataset_hash,
        "preregistration_sha256": dataset["preregistration_sha256"],
        "mode": mode,
        **(
            {
                "selection_split": split,
                "execution_target": execution_target,
                "selection_path": str(selection_path),
                "selection_roles": [
                    {
                        "circuit_id": circuit_id,
                        "topology_id": topology_id,
                        "candidate_path_id": candidate_id,
                        "roles": list(
                            selection_roles[(circuit_id, topology_id, candidate_id)]
                        ),
                    }
                    for circuit_id, topology_id, candidate_id in selection_provenance
                ],
            }
            if mode == "evaluation"
            else {}
        ),
        "selected": [
            {
                "circuit_id": circuit_id,
                "topology_id": topology_id,
                "candidate_path_id": candidate_id,
                "logical_plan_id": candidate_map[(circuit_id, candidate_id)]["logical_plan_id"],
                "physical_plan_id": next(
                    item["physical_plan_id"]
                    for item in candidate_map[(circuit_id, candidate_id)]["topologies"]
                    if item["topology_id"] == topology_id
                ),
            }
            for circuit_id, topology_id, candidate_id in selected
        ],
    }
    output_path.with_suffix(output_path.suffix + ".provenance.json").write_bytes(
        _canonical_bytes(provenance)
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cpu = subparsers.add_parser("cpu")
    cpu.add_argument("--candidate-paths", type=Path, required=True)
    cpu.add_argument("--rankings", type=Path, required=True)
    cpu.add_argument("--output", type=Path, required=True)
    frozen_cpu = subparsers.add_parser(
        "qualify-frozen-selection",
        aliases=(
            "cpu-selection",
            "evaluation-cpu",
            "cpu-evaluate",
            "qualify-evaluation",
        ),
    )
    frozen_cpu.add_argument("--candidate-paths", type=Path, required=True)
    frozen_cpu.add_argument(
        "--selection", "--selection-artifact", dest="selection", type=Path, required=True
    )
    frozen_cpu.add_argument("--split", choices=("validation", "test"), required=True)
    frozen_cpu.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidate-paths", type=Path, required=True)
    prepare.add_argument("--calibration-set", type=Path)
    prepare.add_argument("--selection", "--selection-artifact", dest="selection", type=Path)
    prepare.add_argument("--rankings", type=Path)
    prepare.add_argument(
        "--mode", choices=("sdk", "calibration", "evaluation"), required=True
    )
    prepare.add_argument("--split", choices=("validation", "test"))
    prepare.add_argument(
        "--execution-target",
        "--evaluation-target",
        "--target",
        dest="execution_target",
        choices=("physical", "sdk"),
        default="physical",
    )
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cpu":
        record = qualify_cpu(args.candidate_paths, args.rankings, args.output)
        print(json.dumps({"qualified_candidate_count": record["qualified_candidate_count"]}))
    elif args.command in {
        "qualify-frozen-selection",
        "cpu-selection",
        "evaluation-cpu",
        "cpu-evaluate",
        "qualify-evaluation",
    }:
        record = qualify_frozen_selection(
            args.candidate_paths,
            args.selection,
            args.output,
            split=args.split,
        )
        print(
            json.dumps(
                {
                    "all_passed": record["all_passed"],
                    "qualified_candidate_count": record[
                        "qualified_candidate_count"
                    ],
                },
                sort_keys=True,
            )
        )
    else:
        config = prepare_config(
            dataset_path=args.candidate_paths,
            output_path=args.output,
            mode=args.mode,
            calibration_path=args.calibration_set,
            rankings_path=args.rankings,
            selection_path=args.selection,
            split=args.split,
            execution_target=args.execution_target,
        )
        print(json.dumps({"matrix_count": len(config["matrix"])}))


if __name__ == "__main__":
    main()
