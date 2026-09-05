#!/usr/bin/env python3
"""Fixed-path, software-only census; never generates paths or opens hardware."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys

from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import ContractNode, make_simulation_job
from quantum_bench.upmem.plan import UpmemTopology, collection_resource_admission, physical_plan_id, plan_upmem
from quantum_bench.upmem.protocol import COMPLETION_BYTES, CONTROL_BYTES
from quantum_bench.upmem.runtime import _wram_panel_operation_facts
from quantum_bench.upmem.tiling import canonical_label_geometry

ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "b921b8804e324da75222354ee2f4df41e770b75c"
POLICIES = ("split_complex_float32_v1", "complex_int8_shared_scale_v1")
MEMORY_LIMIT = 512 * 1024 * 1024
WORK_LIMIT = 400
LOWERING_TIMEOUT = 60.0
SCHEMA = "upmem_execution_census_v1"
POOL_FILES = {
    "generalization": ("thesis_results/upmem_path_heuristic_generalization_v1/software/candidate_paths.json",
                       "d95150ddf89f6aafa861000b0db2d8447d64456a035c5404463a878c3a319049"),
    "pilot": ("thesis_results/upmem_path_heuristic_v1/software/candidate_paths.json",
              "5269bfea2a1777b041e10edf9618ba613d44021ce89b3ff0d95093a81c095b65"),
}
CIRCUITS = ("quantization_stress_16q_l2", "hs_20q_d1", "edc_14q", "bv_18q")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def select_paths(circuit: dict) -> list[dict]:
    candidates = circuit["candidates"]
    greedy = [p for p in candidates if p["is_greedy"]]
    if len(greedy) != 1:
        raise ValueError("exactly one retained greedy reference is required")
    selections = [("greedy", greedy[0])]
    for role, feature in (("minimum_flops", "flops"), ("minimum_peak", "peak_intermediate_elements")):
        selections.append((role, min(candidates, key=lambda p: (
            p["conventional_features"][feature], p["candidate_path_id"]))))
    result: dict[str, dict] = {}
    for role, candidate in selections:
        key = candidate["candidate_path_id"]
        if key not in result:
            result[key] = {
                "candidate_path_id": key, "path": candidate["path"],
                "logical_plan_id": candidate["logical_plan_id"], "path_roles": [],
                "retained_infeasibility_reasons": sorted({
                    t["infeasibility_reason"] for t in candidate.get("topologies", [])
                    if t.get("infeasibility_reason")
                }),
            }
        result[key]["path_roles"].append(role)
    return [result[key] for key in sorted(result)]


def frozen_cells() -> tuple[list[dict], dict]:
    pools = {}
    provenance = {}
    for name, (relative, expected) in POOL_FILES.items():
        data = (ROOT / relative).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"frozen candidate pool hash mismatch: {relative}")
        pools[name] = json.loads(data)
        provenance[name] = {"path": relative, "sha256": expected}
    cells = []
    for circuit_id in CIRCUITS:
        source = pools["pilot" if circuit_id == "bv_18q" else "generalization"]
        circuit = next(c for c in source["circuits"] if c["circuit_id"] == circuit_id)
        definition = {key: circuit["circuit"][key] for key in ("name", "parameters")}
        for selection in select_paths(circuit):
            for policy in POLICIES:
                for dpus in (1, 4):
                    cell = {
                        **selection, "circuit_id": circuit_id, "circuit": definition,
                        "numeric_policy": policy,
                        "topology": {"dpu_count": dpus, "rank_count": 1, "tasklets_per_dpu": 8},
                    }
                    cell["cell_id"] = digest(canonical_bytes(cell))
                    cells.append(cell)
    return cells, provenance


def operation_facts(node: ContractNode, units: tuple, policy: str, topology: UpmemTopology) -> dict:
    b, m, k, n = canonical_label_geometry(
        node.left.labels, node.left.shape, node.right.labels, node.right.shape, node.output.labels)
    waves = sorted({unit.wave for unit in units})
    occupied = Counter(unit.wave for unit in units)
    requests = 4 * len(waves)
    return {
        "node_id": node.node_id, "b": b, "m": m, "n": n, "k": k,
        "left_elements": prod(node.left.shape), "right_elements": prod(node.right.shape),
        "output_elements": prod(node.output.shape), "work_unit_count": len(units),
        "output_tile_count": len({(u.batch_start, u.m_start, u.n_start) for u in units}),
        "k_chunk_count": len({(u.k_start, u.k_size) for u in units}),
        "wave_count": len(waves), "request_count": requests,
        "lane_envelope_submissions": 4, "batched_envelope_submissions": 1,
        "dpu_launch_count": requests, "descriptor_count": requests,
        "dpu_record_count": requests * topology.dpu_count,
        "output_file_count": 4 * len(units),
        "idle_dpu_slots": sum(topology.dpu_count - occupied[w] for w in waves),
        "partial_wave_count": sum(occupied[w] < topology.dpu_count for w in waves),
        "planned_h2d_operand_bytes_estimate": 4 * sum(u.estimated_input_bytes for u in units),
        "planned_d2h_output_bytes_estimate": 4 * sum(u.estimated_output_bytes for u in units),
        "h2d_control_bytes": requests * topology.dpu_count * CONTROL_BYTES,
        "d2h_completion_bytes": requests * topology.dpu_count * COMPLETION_BYTES,
        "maximum_work_unit_aligned_mram_bytes": max((u.aligned_mram_bytes for u in units), default=0),
        "kernel_facts": _wram_panel_operation_facts(units, numeric_policy=policy,
                                                    tasklets_per_dpu=topology.tasklets_per_dpu),
        "work_units": [asdict(u) for u in units],
        "measured_timing": None, "measured_timing_reason": "source_only_no_execution",
        "iram_bytes": None, "iram_reason": "not_measured_by_source_census",
    }


def characterize_cell(cell: dict, *, memory_limit: int = MEMORY_LIMIT,
                      work_limit: int = WORK_LIMIT) -> dict:
    result = {**cell, "status": "rejected", "rejection_reasons": [], "operations": []}
    if cell["logical_plan_id"] is None:
        result["rejection_reasons"] = ["retained_candidate_has_no_logical_identity"]
        return result
    circuit = builtin_circuit(cell["circuit"]["name"], cell["circuit"]["parameters"])
    network, inputs = lower_tensor_network(make_simulation_job(circuit))
    dag = build_contraction_dag(network, tuple(tuple(pair) for pair in cell["path"]))
    observed_identity = contraction_dag_hash(dag)
    if observed_identity != cell["logical_plan_id"]:
        result["observed_logical_plan_id"] = observed_identity
        result["rejection_reasons"] = ["logical_plan_identity_mismatch"]
        return result
    # Preserve the preregistered conservative retained-tensor/transport estimate.
    input_bytes = sum(int(value.nbytes) for value in inputs.values())
    output_bytes = sum(prod(node.output.shape) * 16 for node in dag.nodes)
    memory = input_bytes + output_bytes + 2 * (input_bytes // 2 + output_bytes // 2)
    result["estimated_host_memory_bytes"] = memory
    if memory > memory_limit:
        result["rejection_reasons"] = ["host_memory_limit"]
        return result
    topology = UpmemTopology(**cell["topology"])
    plan = plan_upmem(dag, numeric_policy=cell["numeric_policy"], topology=topology)
    units = tuple(u for stage in plan.stages for u in stage.work_units)
    result["planned_work_unit_count"] = len(units)
    if len(units) > work_limit:
        result["rejection_reasons"] = ["planned_work_unit_limit"]
        return result
    result["physical_plan_id"] = physical_plan_id(plan)
    result["resource_admission"] = collection_resource_admission(plan)
    result["eligibility_scope"] = "host_only_preparation_not_scaling_admission"
    result["host_reduce_stage_count"] = sum(s.kind == "host_reduce" for s in plan.stages)
    for node in dag.nodes:
        if isinstance(node, ContractNode):
            assigned = tuple(u for u in units if u.node_id == node.node_id)
            if not assigned:
                raise ValueError("contract node has no planned work")
            result["operations"].append(operation_facts(node, assigned, cell["numeric_policy"], topology))
    result["status"] = "eligible"
    result["totals"] = {
        key: sum(op[key] for op in result["operations"])
        for key in ("work_unit_count", "wave_count", "request_count", "lane_envelope_submissions",
                    "batched_envelope_submissions", "dpu_launch_count", "descriptor_count", "dpu_record_count",
                    "output_file_count", "idle_dpu_slots", "partial_wave_count",
                    "planned_h2d_operand_bytes_estimate", "planned_d2h_output_bytes_estimate",
                    "h2d_control_bytes", "d2h_completion_bytes")
    }
    return result


def isolated_cell(cell: dict, *, timeout: float = LOWERING_TIMEOUT) -> dict:
    try:
        run = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(cell), text=True, capture_output=True, timeout=timeout, cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
    except subprocess.TimeoutExpired:
        return {**cell, "status": "rejected", "rejection_reasons": ["lowering_timeout"], "operations": []}
    if run.returncode:
        return {**cell, "status": "rejected", "rejection_reasons": ["planning_error"],
                "diagnostic": run.stderr, "operations": []}
    return json.loads(run.stdout)


def write_census(output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty")
    cells, pools = frozen_cells()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    output.mkdir(parents=True, exist_ok=True)
    selection = {"runtime_base_sha": BASE_SHA, "source_sha": source, "source_dirty": dirty,
                 "candidate_pools": pools, "cells": cells}
    (output / "selected_paths.json").write_bytes(canonical_bytes(selection))
    results = []
    for index, cell in enumerate(cells):
        record = isolated_cell(cell)
        results.append(record)
        print(f"{index + 1}/{len(cells)} {cell['circuit_id']} {cell['numeric_policy']} "
              f"D{cell['topology']['dpu_count']} {record['status']}", file=sys.stderr, flush=True)
    report = {
        "schema_version": SCHEMA, "source_sha": source, "source_dirty": dirty,
        "runtime_base_sha": BASE_SHA, "candidate_pools": pools,
        "selection_sha256": digest((output / "selected_paths.json").read_bytes()),
        "dependency_versions": {n: metadata.version(n) for n in ("numpy", "opt_einsum", "cotengra", "quimb")},
        "limits": {"host_memory_bytes": MEMORY_LIMIT, "work_units": WORK_LIMIT, "lowering_timeout_s": LOWERING_TIMEOUT},
        "execution": {"physical": False, "sdk": False, "timing_measured": False},
        "cells": results, "counts": dict(Counter(r["status"] for r in results)),
    }
    (output / "execution_census.json").write_bytes(canonical_bytes(report))
    rows = []
    for cell in results:
        common = {key: cell[key] for key in ("cell_id", "circuit_id", "candidate_path_id", "numeric_policy", "status")}
        common.update(dpu_count=cell["topology"]["dpu_count"], rejection_reasons=";".join(cell["rejection_reasons"]))
        for operation in cell["operations"] or [{}]:
            rows.append({**common, **{k: v for k, v in operation.items() if not isinstance(v, (dict, list))}})
    columns = sorted({key for row in rows for key in row})
    with (output / "execution_census.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (output / "boundary_report.md").write_text(
        "# Source-only execution boundary census\n\n"
        "No hardware or SDK execution; timing and IRAM facts are unavailable.\n"
        "Python owns DAG order, encoding, four lane submissions and reconstruction.\n"
        "The persistent C host owns sequential embedded-request SDK transfers/launches.\n"
        "There are four lane envelopes per contract node and four launches per wave.\n"
        "Idle DPU descriptors are retained but do not create output files.\n"
        "Envelope descriptor_count counts requests; dpu_record_count includes idle output paths.\n"
        "Scaling admission is reported separately; legal underfilled waves remain in this host-only census.\n"
        "MRAM/WRAM traffic and host-memory figures are algorithmic estimates.\n"
        "The historical heuristic packed_operation_count aliases waves, not submissions.\n"
        "Rejected cells are retained; paths are never replaced.\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text("".join(
        f"{digest(p.read_bytes())}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file()), encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(characterize_cell(json.load(sys.stdin)), allow_nan=False))
    elif args.output_dir is not None:
        print(json.dumps(write_census(args.output_dir)["counts"], sort_keys=True))
    else:
        parser.error("--output-dir is required")


if __name__ == "__main__":
    main()
