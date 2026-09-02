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
        path, _ = plan_opt_einsum(network, optimize="greedy")
    else:
        path, _ = plan_cotengra(
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
    return build_contraction_dag(network, path), inputs


def _ranking_best(rankings_path: Path) -> dict[tuple[str, str], str]:
    result = {}
    with rankings_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["equal_weight_rank"]) == 1:
                result[(row["circuit_id"], row["topology_id"])] = row["candidate_path_id"]
    return result


def _representative_ids(
    circuit: dict[str, Any], rankings: dict[tuple[str, str], str]
) -> tuple[str, ...]:
    candidates = tuple(
        candidate
        for candidate in circuit["candidates"]
        if candidate.get("conventional_features") is not None
        and any(item.get("feasible") is True for item in candidate["topologies"])
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
    for topology_id in ("1dpu_t8", "4dpu_t8"):
        selected.add(rankings[(circuit["circuit_id"], topology_id)])
        feasible = [
            candidate
            for candidate in candidates
            if next(item for item in candidate["topologies"] if item["topology_id"] == topology_id)["feasible"]
        ]
        selected.add(
            min(
                feasible,
                key=lambda candidate: (
                    next(item for item in candidate["topologies"] if item["topology_id"] == topology_id)["features"]["B_host_dpu"],
                    candidate["candidate_path_id"],
                ),
            )["candidate_path_id"]
        )
    return tuple(sorted(selected))


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
            difference = np.asarray(actual.output, dtype=np.complex128) - np.asarray(
                reference, dtype=np.complex128
            )
            maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
            denominator = float(np.linalg.norm(reference.reshape(-1)))
            relative_l2 = (
                float(np.linalg.norm(difference.reshape(-1))) / denominator
                if denominator
                else float(np.linalg.norm(difference.reshape(-1)))
            )
            passed = bool(np.allclose(actual.output, reference, rtol=1.0e-5, atol=1.0e-5))
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
        "candidate_set_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
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


def prepare_config(
    *,
    dataset_path: Path,
    calibration_path: Path,
    rankings_path: Path,
    output_path: Path,
    mode: str,
) -> dict[str, Any]:
    dataset = _load(dataset_path)
    candidate_map = _candidate_map(dataset)
    circuit_map = _circuit_map(dataset)
    rankings = _ranking_best(rankings_path)
    selected: list[tuple[str, str, str]] = []
    if mode == "calibration":
        calibration = _load(calibration_path)
        for cell in calibration["cells"]:
            for candidate_id in cell["candidate_path_ids"]:
                selected.append((cell["circuit_id"], cell["topology_id"], candidate_id))
        warmups, measurements, seed, simulator = 1, 3, 20260910, False
    elif mode == "sdk":
        for circuit in dataset["circuits"]:
            ids = _representative_ids(circuit, rankings)
            for topology_id in ("1dpu_t8", "4dpu_t8"):
                selected.extend((circuit["circuit_id"], topology_id, path) for path in ids)
        warmups, measurements, seed, simulator = 0, 1, 20260909, True
    else:
        raise ValueError("mode must be sdk or calibration")
    selected = sorted(set(selected))
    cases = {
        circuit_id: {"circuit": {**circuit_map[circuit_id]["circuit"], "path": None}}
        for circuit_id in sorted({item[0] for item in selected})
    }
    plans = {}
    matrix = []
    for circuit_id, topology_id, candidate_id in selected:
        candidate = candidate_map[(circuit_id, candidate_id)]
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
    config = {
        "schema_version": "tn_benchmark_v3",
        "experiment_id": f"upmem-path-heuristic-{mode}-v1",
        "defaults": {"timeout_s": 120.0},
        "collection": _collection(warmups=warmups, measurements=measurements, seed=seed),
        "cases": cases,
        "plans": plans,
        "routes": {
            topology_id: _route(topology_id, simulator=simulator)
            for topology_id in ("1dpu_t8", "4dpu_t8")
        },
        "matrix": matrix,
    }
    # Matrix route names use the canonical mapping keys above.
    for item in config["matrix"]:
        item["route_ids"] = [item["route_ids"][0].removeprefix("upmem_")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cpu = subparsers.add_parser("cpu")
    cpu.add_argument("--candidate-paths", type=Path, required=True)
    cpu.add_argument("--rankings", type=Path, required=True)
    cpu.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--candidate-paths", type=Path, required=True)
    prepare.add_argument("--calibration-set", type=Path, required=True)
    prepare.add_argument("--rankings", type=Path, required=True)
    prepare.add_argument("--mode", choices=("sdk", "calibration"), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cpu":
        record = qualify_cpu(args.candidate_paths, args.rankings, args.output)
        print(json.dumps({"qualified_candidate_count": record["qualified_candidate_count"]}))
    else:
        config = prepare_config(
            dataset_path=args.candidate_paths,
            calibration_path=args.calibration_set,
            rankings_path=args.rankings,
            output_path=args.output,
            mode=args.mode,
        )
        print(json.dumps({"matrix_count": len(config["matrix"])}))


if __name__ == "__main__":
    main()
