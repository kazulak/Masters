#!/usr/bin/env python3
"""Generate and check the preregistered M7C scaling-workload record.

This selector performs deterministic circuit lowering and physical-plan mapping
only.  It does not open a UPMEM session or collect a timing measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from math import prod
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from quantum_bench.circuits import builtin_circuit
from quantum_bench.evidence import canonical_json, problem_id, tensor_network_structure_id
from quantum_bench.experiment import load_experiment_config
from quantum_bench.lowering import build_contraction_dag, contraction_dag_hash, lower_tensor_network
from quantum_bench.model import ReduceNode, make_simulation_job
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.upmem.plan import (
    UpmemPlan,
    UpmemTopology,
    collection_resource_admission,
    physical_plan_id,
    plan_upmem,
)
from quantum_bench.upmem.runtime import _wram_panel_operation_facts


ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA = "m7c_workload_selection_v1"
NUMERIC_POLICY = "split_complex_float32_v1"
PLANNER_CONFIG = {
    "engine": "opt_einsum",
    "mode": "greedy",
    "max_repeats": 1,
    "seed": 0,
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    circuit_name: str
    parameters: Mapping[str, int]
    role: str


_CANDIDATES = (
    Candidate(
        candidate_id="quantization_stress_16q_l2",
        circuit_name="quantization_stress",
        parameters={"n_qubits": 16, "repeat_layers": 2},
        role="reserve",
    ),
    Candidate(
        candidate_id="quantization_stress_18q_l2",
        circuit_name="quantization_stress",
        parameters={"n_qubits": 18, "repeat_layers": 2},
        role="primary_candidate",
    ),
    Candidate(
        candidate_id="ghz_chain_18q",
        circuit_name="ghz_chain",
        parameters={"n_qubits": 18},
        role="secondary_candidate",
    ),
)
_TOPOLOGIES = (
    ("dpu1_tasklet1", UpmemTopology(dpu_count=1, rank_count=1, tasklets_per_dpu=1)),
    ("dpu1_tasklet8", UpmemTopology(dpu_count=1, rank_count=1, tasklets_per_dpu=8)),
    ("dpu2_tasklet8", UpmemTopology(dpu_count=2, rank_count=1, tasklets_per_dpu=8)),
    ("dpu4_tasklet8", UpmemTopology(dpu_count=4, rank_count=1, tasklets_per_dpu=8)),
)
_SELECTION_RULE = {
    "policy": "predeclared_circuit_structure_and_resource_admission_v1",
    "primary_candidate_id": "quantization_stress_18q_l2",
    "secondary_candidate_id": "ghz_chain_18q",
    "requirements": {
        "numeric_policy": NUMERIC_POLICY,
        "tasklet_route": "dpu1_tasklet8",
        "dpu_routes": ["dpu2_tasklet8", "dpu4_tasklet8"],
        "all_required_routes_must_pass_collection_resource_admission": True,
        "secondary_must_use_a_structurally_distinct_builtin_circuit": True,
    },
    "timing_selection_rule": "no_physical_or_simulator_timing_is_used",
}


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise ValueError("cannot determine selection source commit")
    return commit


def _dependency_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _plain(value: object) -> Any:
    return json.loads(canonical_json(value))


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _plan_candidate(candidate: Candidate) -> dict[str, object]:
    circuit = builtin_circuit(candidate.circuit_name, dict(candidate.parameters))
    job = make_simulation_job(circuit)
    network, _ = lower_tensor_network(job)
    path, provenance = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, path)
    stable_provenance = dict(provenance)
    # Planner elapsed time is an observation, not a deterministic selection
    # constraint. Persisting it would make the preregistration unreplayable.
    stable_provenance.pop("planning_time_s", None)
    topology_records = {
        topology_id: _topology_record(dag, topology)
        for topology_id, topology in _TOPOLOGIES
    }
    required_records = (
        topology_records["dpu1_tasklet8"],
        topology_records["dpu2_tasklet8"],
        topology_records["dpu4_tasklet8"],
    )
    eligible = all(
        record["collection_resource_admission_passed"] is True
        for record in required_records
    )
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_role": candidate.role,
        "circuit": {
            "kind": "builtin",
            "name": candidate.circuit_name,
            "parameters": dict(candidate.parameters),
            "query": job.query,
        },
        "problem_id": problem_id(job),
        "tensor_network_structure_id": tensor_network_structure_id(network),
        "logical_plan_id": contraction_dag_hash(dag),
        "planner": {
            "configuration": PLANNER_CONFIG,
            "provenance": stable_provenance,
            "path_pairs": [list(pair) for pair in path],
        },
        "peak_intermediate_estimate": provenance["largest_intermediate"],
        "final_output_elements": prod(dag.output.shape),
        "host_reduce_node_count": sum(
            isinstance(node, ReduceNode) for node in dag.nodes
        ),
        "host_reduce_input_count": sum(
            len(node.inputs) for node in dag.nodes if isinstance(node, ReduceNode)
        ),
        "topologies": topology_records,
        "selection_eligible": eligible,
        "selection_rejection_reason": (
            None if eligible else "required_tasklet_or_dpu_admission_failed"
        ),
    }


def _topology_record(dag: object, topology: UpmemTopology) -> dict[str, object]:
    plan = plan_upmem(dag, numeric_policy=NUMERIC_POLICY, topology=topology)
    units = tuple(
        unit for stage in plan.stages for unit in stage.work_units
    )
    admission = collection_resource_admission(plan)
    movement = _wram_panel_operation_facts(
        units,
        numeric_policy=NUMERIC_POLICY,
        tasklets_per_dpu=topology.tasklets_per_dpu,
    )
    wave_summary = _wave_summary(plan)
    return {
        "physical_plan_id": physical_plan_id(plan),
        "topology": {
            "dpu_count": topology.dpu_count,
            "rank_count": topology.rank_count,
            "tasklets_per_dpu": topology.tasklets_per_dpu,
        },
        "work_unit_count": len(units),
        "m_size_distribution": _distribution(unit.m_size for unit in units),
        "n_size_distribution": _distribution(unit.n_size for unit in units),
        "k_size_distribution": _distribution(unit.k_size for unit in units),
        "wave_count": wave_summary["wave_count"],
        "wave_arithmetic_work_distribution": wave_summary["distribution"],
        "wave_arithmetic_work_sha256": wave_summary["ordered_wave_hash"],
        "estimated_host_h2d_bytes_four_real_products": 4
        * sum(unit.estimated_input_bytes for unit in units),
        "estimated_host_d2h_bytes_four_real_products": 4
        * sum(unit.estimated_output_bytes for unit in units),
        "wram_panel_algorithm_facts": movement,
        "resource_admission": admission,
        "collection_resource_admission_passed": admission[
            "collection_resource_admission_passed"
        ],
    }


def _distribution(values: object) -> list[dict[str, int]]:
    counts = Counter(int(value) for value in values)
    return [{"value": value, "count": counts[value]} for value in sorted(counts)]


def _wave_summary(plan: UpmemPlan) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for stage in plan.stages:
        if stage.kind != "contract_batch":
            continue
        for wave in sorted({unit.wave for unit in stage.work_units}):
            units = tuple(unit for unit in stage.work_units if unit.wave == wave)
            rows.append(
                {
                    "stage_id": stage.stage_id,
                    "wave": wave,
                    "arithmetic_work": sum(
                        unit.estimated_arithmetic_work for unit in units
                    ),
                    "useful_dpu_slots": len(
                        {
                            (unit.logical_rank, unit.logical_dpu)
                            for unit in units
                            if unit.estimated_arithmetic_work > 0
                        }
                    ),
                }
            )
    distribution = Counter(
        (int(row["arithmetic_work"]), int(row["useful_dpu_slots"]))
        for row in rows
    )
    return {
        "wave_count": len(rows),
        "distribution": [
            {
                "arithmetic_work": arithmetic_work,
                "useful_dpu_slots": useful_slots,
                "count": distribution[(arithmetic_work, useful_slots)],
            }
            for arithmetic_work, useful_slots in sorted(distribution)
        ],
        "ordered_wave_hash": _hash(rows),
    }


def build_selection() -> dict[str, object]:
    """Build the immutable source-only workload-selection record."""

    candidates = [_plan_candidate(candidate) for candidate in _CANDIDATES]
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    primary = by_id[_SELECTION_RULE["primary_candidate_id"]]
    secondary = by_id[_SELECTION_RULE["secondary_candidate_id"]]
    if primary["selection_eligible"] is not True:
        raise ValueError("predeclared primary M7C workload no longer meets admission")
    if secondary["selection_eligible"] is not True:
        raise ValueError("predeclared secondary M7C workload no longer meets admission")
    selection_basis = {
        "schema_version": SELECTION_SCHEMA,
        "planner_configuration": PLANNER_CONFIG,
        "selection_rule": _SELECTION_RULE,
        "candidates": candidates,
        "selected_primary": primary["candidate_id"],
        "selected_secondary": secondary["candidate_id"],
    }
    return {
        **selection_basis,
        "source_commit": _source_commit(),
        "python_version": sys.version.split()[0],
        "dependency_versions": {"opt_einsum": _dependency_version("opt_einsum")},
        "constraints_hash": _hash(selection_basis),
        "selection_audit_hash": _hash(
            {
                "primary": primary,
                "secondary": secondary,
                "selection_rule": _SELECTION_RULE,
            }
        ),
    }


def write_selection(output: Path) -> Path:
    if output.exists():
        raise ValueError(f"selection output must be absent: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_selection(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def check_selection(selection_path: Path, config_path: Path | None = None) -> None:
    persisted = json.loads(selection_path.read_text(encoding="utf-8"))
    current = build_selection()
    if persisted != current:
        raise ValueError("persisted M7C selection differs from deterministic recomputation")
    if config_path is None:
        return
    config = load_experiment_config(config_path)
    selected_case = config["cases"].get("scaling_primary")
    selected_plan = config["plans"].get("greedy")
    if selected_case is None or selected_plan is None:
        raise ValueError("M7C scaling config must select scaling_primary and greedy")
    circuit_config = selected_case["circuit"]
    primary = next(
        entry
        for entry in persisted["candidates"]
        if entry["candidate_id"] == persisted["selected_primary"]
    )
    if dict(circuit_config["parameters"]) != primary["circuit"]["parameters"]:
        raise ValueError("M7C scaling config parameters drift from selected primary")
    if circuit_config["name"] != primary["circuit"]["name"]:
        raise ValueError("M7C scaling config circuit drift from selected primary")
    planner = selected_plan["planner"]
    if planner["engine"] != PLANNER_CONFIG["engine"] or planner["mode"] != PLANNER_CONFIG["mode"]:
        raise ValueError("M7C scaling config planner drift from selection")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if (args.output is None) == (args.check is None):
        parser.error("specify exactly one of --output or --check")
    try:
        if args.output is not None:
            output = write_selection(args.output.resolve())
            payload: Mapping[str, object] = {"status": "written", "output": str(output)}
        else:
            check_selection(args.check.resolve(), args.config.resolve() if args.config else None)
            payload = {"status": "checked", "selection": str(args.check.resolve())}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(_plain(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
