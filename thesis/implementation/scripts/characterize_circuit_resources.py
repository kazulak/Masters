#!/usr/bin/env python3
"""Build deterministic circuit and UPMEM resource descriptors without execution."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from math import comb, dist, log2, prod
from pathlib import Path
import subprocess

from quantum_bench.circuits import builtin_circuit
from quantum_bench.evidence import problem_id, tensor_network_structure_id
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import ContractNode, make_simulation_job
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.plan import UpmemPlan, UpmemTopology, collection_resource_admission, physical_plan_id, plan_upmem
from quantum_bench.upmem.runtime import _wram_panel_operation_facts


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "circuit_structure_resource_sensitivity_v1"
SELECTION_SCHEMA_VERSION = "circuit_resource_sensitivity_selection_v1"
NUMERIC_POLICY = "split_complex_float32_v1"
COMPLEX_LANE_PASS_COUNT = 4
COMPLEX128_BYTES = 16
JSON_FILENAME = "characterization.json"
CSV_FILENAME = "characterization.csv"


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    circuit_name: str
    parameters: Mapping[str, int]


CANDIDATES = (
    Candidate("quantization_stress_18q_l2", "quantization_stress", {"n_qubits": 18, "repeat_layers": 2}),
    Candidate("ghz_chain_18q", "ghz_chain", {"n_qubits": 18}),
    Candidate("bv_18q", "bv", {"n_qubits": 18}),
    Candidate("hs_18q_d1", "hs", {"n_qubits": 18, "depth": 1}),
)
CSV_COLUMNS = (
    "candidate_id", "circuit_name", "n_qubits", "circuit_depth", "gate_count",
    "interaction_edge_count", "interaction_density", "interaction_max_degree",
    "tensor_count", "index_count", "contraction_count", "planner_flops_estimate",
    "planner_peak_elements_including_final_output_estimate", "largest_non_final_intermediate_elements",
    "largest_non_final_intermediate_bytes_complex128", "max_intermediate_rank", "logical_plan_id",
    "dpu_count", "tasklets_per_dpu", "physical_plan_id", "work_unit_count", "contract_stage_count",
    "wave_count", "dominant_wave_arithmetic_work_estimate", "dominant_wave_useful_dpu_slots",
    "partial_wave_count", "arithmetic_weighted_dpu_slot_utilization", "arithmetic_weighted_tasklet_utilization",
    "estimated_host_h2d_bytes_four_real_products", "estimated_host_d2h_bytes_four_real_products",
    "estimated_native_request_count_four_real_products", "estimated_native_payload_record_count_four_real_products",
)


def _source_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    if result.returncode or len(value) != 40:
        raise ValueError("cannot determine source SHA")
    return value


def _dependency_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _circuit_descriptors(circuit: object) -> dict[str, object]:
    operations = tuple(getattr(circuit, "operations"))
    n_qubits = int(getattr(circuit, "n_qubits"))
    depths = [0] * n_qubits
    gates = Counter(operation.gate for operation in operations)
    edges: set[tuple[int, int]] = set()
    for operation in operations:
        depth = 1 + max(depths[wire] for wire in operation.wires)
        for wire in operation.wires:
            depths[wire] = depth
        for index, left in enumerate(operation.wires):
            for right in operation.wires[index + 1 :]:
                edges.add(tuple(sorted((left, right))))
    degrees = [0] * n_qubits
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    possible_edges = comb(n_qubits, 2)
    return {
        "n_qubits": n_qubits,
        "true_deterministic_depth": max(depths, default=0),
        "gate_count": len(operations),
        "gate_counts": {gate: gates[gate] for gate in sorted(gates)},
        "interaction_graph": {
            "edge_count": len(edges),
            "density": len(edges) / possible_edges if possible_edges else 0.0,
            "max_degree": max(degrees, default=0),
        },
    }


def _tensor_descriptors(network: object, dag: object, provenance: Mapping[str, object]) -> dict[str, object]:
    tensors = tuple(getattr(network, "tensors"))
    nodes = tuple(getattr(dag, "nodes"))
    output = getattr(dag, "output")
    contracts = tuple(node for node in nodes if isinstance(node, ContractNode))
    non_final = tuple(node for node in contracts if node.output.id != output.tensor_id)
    largest_non_final = max((prod(node.output.shape) for node in non_final), default=0)
    planner_peak = int(provenance["largest_intermediate"] or 0)
    final_elements = prod(output.shape)
    return {
        "tensor_count": len(tensors),
        "index_count": len({label for tensor in tensors for label in tensor.labels} | set(getattr(network, "output_labels"))),
        "contraction_count": len(contracts),
        "planner_flops_estimate": provenance["optimized_flops"],
        "planner_peak_elements_including_final_output_estimate": max(planner_peak, final_elements),
        "final_output_elements": final_elements,
        "largest_non_final_intermediate_elements": largest_non_final,
        "largest_non_final_intermediate_bytes_complex128": largest_non_final * COMPLEX128_BYTES,
        "max_intermediate_rank": max((len(node.output.shape) for node in contracts), default=0),
        "logical_plan_id": contraction_dag_hash(dag),
    }


def _topologies() -> Iterable[UpmemTopology]:
    for tasklets in range(1, 25):
        yield UpmemTopology(dpu_count=1, rank_count=1, tasklets_per_dpu=tasklets)
    for dpus in range(2, 5):
        yield UpmemTopology(dpu_count=dpus, rank_count=1, tasklets_per_dpu=8)


def _waves(plan: UpmemPlan) -> tuple[tuple[object, ...], ...]:
    result: list[tuple[object, ...]] = []
    for stage in plan.stages:
        if stage.kind != "contract_batch":
            continue
        for wave in sorted({unit.wave for unit in stage.work_units}):
            result.append(tuple(unit for unit in stage.work_units if unit.wave == wave))
    return tuple(result)


def _physical_descriptors(dag: object, topology: UpmemTopology) -> dict[str, object]:
    plan = plan_upmem(dag, numeric_policy=NUMERIC_POLICY, topology=topology)
    units = tuple(unit for stage in plan.stages for unit in stage.work_units)
    waves = _waves(plan)
    admission = collection_resource_admission(plan)
    active_rank_wave_count = sum(len({unit.logical_rank for unit in wave}) for wave in waves)
    partial = sum(len({(unit.logical_rank, unit.logical_dpu) for unit in wave}) < topology.dpu_count for wave in waves)
    return {
        "topology": {"dpu_count": topology.dpu_count, "rank_count": topology.rank_count, "tasklets_per_dpu": topology.tasklets_per_dpu},
        "physical_plan_id": physical_plan_id(plan),
        "work_unit_count": len(units),
        "contract_stage_count": sum(stage.kind == "contract_batch" for stage in plan.stages),
        "host_reduce_stage_count": sum(stage.kind == "host_reduce" for stage in plan.stages),
        "wave_count": len(waves),
        "dominant_wave_arithmetic_work_estimate": admission["dominant_work_wave_arithmetic_work"],
        "dominant_wave_useful_dpu_slots": admission["dominant_work_wave_populated_dpu_slots"],
        "dominant_wave_dpu_utilization": admission["dominant_work_wave_utilization"],
        "partial_wave_count": partial,
        "arithmetic_weighted_dpu_slot_utilization": admission["arithmetic_weighted_dpu_slot_utilization"],
        "arithmetic_weighted_tasklet_utilization": admission["arithmetic_weighted_tasklet_utilization"],
        "estimated_host_h2d_bytes_four_real_products": COMPLEX_LANE_PASS_COUNT * sum(unit.estimated_input_bytes for unit in units),
        "estimated_host_d2h_bytes_four_real_products": COMPLEX_LANE_PASS_COUNT * sum(unit.estimated_output_bytes for unit in units),
        "wram_panel_algorithm_facts": _wram_panel_operation_facts(units, numeric_policy=NUMERIC_POLICY, tasklets_per_dpu=topology.tasklets_per_dpu),
        "estimated_native_request_count_four_real_products": COMPLEX_LANE_PASS_COUNT * active_rank_wave_count,
        "estimated_native_payload_record_count_four_real_products": COMPLEX_LANE_PASS_COUNT * len(units),
        "native_request_estimate_basis": "four_complex_real_lanes_times_active_rank_wave_requests_v4",
        "native_payload_record_estimate_basis": "four_complex_real_lanes_times_assigned_work_units",
    }


def _candidate_record(candidate: Candidate) -> dict[str, object]:
    circuit = builtin_circuit(candidate.circuit_name, dict(candidate.parameters))
    job = make_simulation_job(circuit)
    network, _ = lower_tensor_network(job)
    path, provenance = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, path)
    return {
        "candidate_id": candidate.candidate_id,
        "circuit": {"kind": "builtin", "name": candidate.circuit_name, "parameters": dict(candidate.parameters), "descriptors": _circuit_descriptors(circuit)},
        "problem_id": problem_id(job),
        "tensor_network_structure_id": tensor_network_structure_id(network),
        "tensor_network": _tensor_descriptors(network, dag, provenance),
        "planner": {"engine": provenance["planner_engine"], "planner_id": provenance["planner_id"], "mode": provenance["optimize_mode"], "configuration": provenance["planner_config"], "configuration_sha256": provenance["planner_config_hash"], "dependency_versions": provenance["dependency_versions"]},
        "physical_plans": [_physical_descriptors(dag, topology) for topology in _topologies()],
    }


def build_characterization() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha": _source_sha(),
        "dependency_identity": {"opt_einsum": _dependency_version("opt_einsum")},
        "planner_identity": {"engine": "opt_einsum", "mode": "greedy"},
        "execution": {"kind": "source_only_characterization", "hardware_executed": False, "simulator_executed": False},
        "candidates": [_candidate_record(candidate) for candidate in CANDIDATES],
    }


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _csv_rows(record: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for candidate in record["candidates"]:
        circuit = candidate["circuit"]
        desc = circuit["descriptors"]
        graph = desc["interaction_graph"]
        network = candidate["tensor_network"]
        for physical in candidate["physical_plans"]:
            topology = physical["topology"]
            rows.append({
                "candidate_id": candidate["candidate_id"], "circuit_name": circuit["name"], "n_qubits": desc["n_qubits"], "circuit_depth": desc["true_deterministic_depth"], "gate_count": desc["gate_count"], "interaction_edge_count": graph["edge_count"], "interaction_density": graph["density"], "interaction_max_degree": graph["max_degree"], "tensor_count": network["tensor_count"], "index_count": network["index_count"], "contraction_count": network["contraction_count"], "planner_flops_estimate": network["planner_flops_estimate"], "planner_peak_elements_including_final_output_estimate": network["planner_peak_elements_including_final_output_estimate"], "largest_non_final_intermediate_elements": network["largest_non_final_intermediate_elements"], "largest_non_final_intermediate_bytes_complex128": network["largest_non_final_intermediate_bytes_complex128"], "max_intermediate_rank": network["max_intermediate_rank"], "logical_plan_id": network["logical_plan_id"], "dpu_count": topology["dpu_count"], "tasklets_per_dpu": topology["tasklets_per_dpu"], "physical_plan_id": physical["physical_plan_id"], "work_unit_count": physical["work_unit_count"], "contract_stage_count": physical["contract_stage_count"], "wave_count": physical["wave_count"], "dominant_wave_arithmetic_work_estimate": physical["dominant_wave_arithmetic_work_estimate"], "dominant_wave_useful_dpu_slots": physical["dominant_wave_useful_dpu_slots"], "partial_wave_count": physical["partial_wave_count"], "arithmetic_weighted_dpu_slot_utilization": physical["arithmetic_weighted_dpu_slot_utilization"], "arithmetic_weighted_tasklet_utilization": physical["arithmetic_weighted_tasklet_utilization"], "estimated_host_h2d_bytes_four_real_products": physical["estimated_host_h2d_bytes_four_real_products"], "estimated_host_d2h_bytes_four_real_products": physical["estimated_host_d2h_bytes_four_real_products"], "estimated_native_request_count_four_real_products": physical["estimated_native_request_count_four_real_products"], "estimated_native_payload_record_count_four_real_products": physical["estimated_native_payload_record_count_four_real_products"],
            })
    return rows


def write_characterization(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = build_characterization()
    json_path = output_dir / JSON_FILENAME
    csv_path = output_dir / CSV_FILENAME
    json_path.write_bytes(_json_bytes(record))
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_rows(record))
    return json_path, csv_path


def check_characterization(output_dir: Path) -> None:
    record = build_characterization()
    json_path = output_dir / JSON_FILENAME
    csv_path = output_dir / CSV_FILENAME
    if json_path.read_bytes() != _json_bytes(record):
        raise ValueError("characterization JSON differs from deterministic recomputation")
    expected_csv = csv_path.read_bytes()
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(_csv_rows(record))
    if expected_csv != stream.getvalue().encode("utf-8"):
        raise ValueError("characterization CSV differs from deterministic recomputation")


def _physical_for(candidate: Mapping[str, object], dpu_count: int, tasklets: int) -> Mapping[str, object]:
    for physical in candidate["physical_plans"]:
        if physical["topology"] == {"dpu_count": dpu_count, "rank_count": 1, "tasklets_per_dpu": tasklets}:
            return physical
    raise ValueError("characterization topology is missing")


def _selection_descriptor(candidate: Mapping[str, object]) -> dict[str, float]:
    circuit = candidate["circuit"]["descriptors"]
    graph = circuit["interaction_graph"]
    network = candidate["tensor_network"]
    t8 = _physical_for(candidate, 1, 8)
    d4 = _physical_for(candidate, 4, 8)
    return {
        "interaction_density": float(graph["density"]),
        "normalized_max_degree": float(graph["max_degree"]) / max(1, int(circuit["n_qubits"]) - 1),
        "log2_optimized_flops": log2(float(network["planner_flops_estimate"])),
        "log2_non_final_peak_elements": log2(max(1, int(network["largest_non_final_intermediate_elements"]))),
        "log2_t8_work_unit_count": log2(max(1, int(t8["work_unit_count"]))),
        "t8_tasklet_utilization": float(t8["arithmetic_weighted_tasklet_utilization"]),
        "d4t8_dpu_utilization": float(d4["arithmetic_weighted_dpu_slot_utilization"]),
    }


def build_selection(record: Mapping[str, object]) -> dict[str, object]:
    candidates = tuple(record["candidates"])
    anchor_id = "quantization_stress_18q_l2"
    non_anchor = tuple(candidate for candidate in candidates if candidate["candidate_id"] != anchor_id)
    low = min(non_anchor, key=lambda candidate: (int(candidate["tensor_network"]["largest_non_final_intermediate_elements"]), float(candidate["tensor_network"]["planner_flops_estimate"]), candidate["candidate_id"]))
    descriptors = {candidate["candidate_id"]: _selection_descriptor(candidate) for candidate in candidates}
    fields = tuple(next(iter(descriptors.values())))
    bounds = {field: (min(values[field] for values in descriptors.values()), max(values[field] for values in descriptors.values())) for field in fields}
    normalized = {cid: tuple(0.0 if bounds[field][0] == bounds[field][1] else (values[field] - bounds[field][0]) / (bounds[field][1] - bounds[field][0]) for field in fields) for cid, values in descriptors.items()}
    distances = {cid: {other: float(dist(normalized[cid], normalized[other])) for other in normalized if other != cid} for cid in normalized}
    distinct = min((candidate for candidate in non_anchor if candidate["candidate_id"] != low["candidate_id"]), key=lambda candidate: (-min(distances[candidate["candidate_id"]][anchor_id], distances[candidate["candidate_id"]][low["candidate_id"]]), candidate["candidate_id"]))
    basis = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_rule": {"anchor_case_id": anchor_id, "low_case_rule": ["largest_non_final_intermediate_elements", "planner_flops_estimate", "candidate_id"], "distinct_case_rule": "maximize_minimum_normalized_euclidean_distance_from_anchor_and_low_case", "distance_fields": list(fields), "normalization": "min_max_over_all_four_candidates_constant_fields_zero", "timing_used": False},
        "candidates": [{"candidate_id": candidate["candidate_id"], "circuit": candidate["circuit"], "problem_id": candidate["problem_id"], "tensor_network": candidate["tensor_network"], "selection_descriptors": descriptors[candidate["candidate_id"]], "distances": distances[candidate["candidate_id"]]} for candidate in candidates],
        "anchor_case_id": anchor_id, "low_case_id": low["candidate_id"], "distinct_case_id": distinct["candidate_id"], "selected_case_ids": [anchor_id, low["candidate_id"], distinct["candidate_id"]],
    }
    return {**basis, "schema_version": SELECTION_SCHEMA_VERSION, "source_sha": record["source_sha"], "characterization_sha256": hashlib.sha256(_json_bytes({key: value for key, value in record.items() if key != "source_sha"})).hexdigest(), "dependency_identity": record["dependency_identity"], "planner_identity": record["planner_identity"], "selection_basis_sha256": hashlib.sha256(json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def write_selection(output: Path, record: Mapping[str, object]) -> Path:
    if output.exists():
        raise ValueError(f"selection output must be absent: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(build_selection(record)))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-dir", type=Path)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args(argv)
    if (args.output_dir is None) == (args.check_dir is None):
        parser.error("specify exactly one of --output-dir or --check-dir")
    if args.selection_output is not None and args.output_dir is None:
        parser.error("--selection-output requires --output-dir")
    try:
        if args.output_dir is not None:
            paths = write_characterization(args.output_dir.resolve())
            result = {"status": "written", "json": str(paths[0]), "csv": str(paths[1])}
            if args.selection_output is not None:
                record = json.loads(paths[0].read_text(encoding="utf-8"))
                result["selection"] = str(write_selection(args.selection_output.resolve(), record))
        else:
            check_characterization(args.check_dir.resolve())
            result = {"status": "checked", "directory": str(args.check_dir.resolve())}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
