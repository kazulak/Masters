from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.result_artifacts import load_result_records
from quantum_bench.bench.simulation_backend_compare import run_simulation_backend_compare
from quantum_bench.circuits import quest_compatible_circuit
from quantum_bench.core.records import ExecutionProfile, RouteOutput, RouteResult
from quantum_bench.providers import route_registry
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config
from quantum_bench.validation import tensor_to_quest_statevector


ROOT = Path(__file__).resolve().parents[1]


def test_quest_exact_route_is_additive_and_benchmark_route_remains_metrics_only() -> None:
    routes = route_registry(ROOT)

    assert routes["quest_cpu_full_state_benchmark"].identity.output_contract == "metrics_only"
    assert routes["quest_cpu_full_state_benchmark"].identity.validation_mode == "benchmark_only"
    assert routes["quest_cpu_full_state_exact"].identity.output_contract == "statevector"
    assert routes["quest_cpu_full_state_exact"].identity.validation_mode == "compare_statevector"
    assert routes["quest_cpu_full_state_exact"].backend_family == "quest"


def test_quest_compatible_bv_sequence_matches_native_order() -> None:
    circuit = quest_compatible_circuit("BV", {"n_qubits": 4})
    gates = [(op.gate, op.wires) for op in circuit.operations]

    assert gates[:4] == [("h", (0,)), ("h", (1,)), ("h", (2,)), ("h", (3,))]
    assert gates[4:7] == [("cx", (0, 3)), ("cx", (1, 3)), ("cx", (2, 3))]
    assert gates[7] == ("x", (3,))
    assert gates[8:] == [("h", (0,)), ("h", (1,)), ("h", (2,))]


def test_tensor_to_quest_statevector_catches_basis_order() -> None:
    tensor = np.zeros((2, 2, 2), dtype=np.complex128)
    tensor[1, 0, 0] = 3.0 + 1.0j
    tensor[0, 1, 0] = 5.0 + 2.0j
    tensor[0, 0, 1] = 7.0 + 4.0j

    state = tensor_to_quest_statevector(tensor)

    assert state[1] == 3.0 + 1.0j
    assert state[2] == 5.0 + 2.0j
    assert state[4] == 7.0 + 4.0j
    assert not np.array_equal(state, tensor.ravel())


def test_simulation_backend_compare_writes_normalized_records(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_simulation_compare
defaults:
  repeats: 1
  planner:
    engine: opt_einsum
    optimize: greedy
workloads:
  - id: quest_bv_3q
    circuit:
      kind: quest_compatible
      name: BV
      n_qubits: 3
routes:
  - id: cpu_tn_einsum_exact
    role: baseline
    required: true
  - id: quest_cpu_full_state_exact
    role: baseline
    required: true
validation:
  reference_route: cpu_tn_einsum_exact
  require_output_for_roles:
    - baseline
  tolerances:
    max_abs_error: 1.0e-9
    l2_error: 1.0e-8
    max_rel_error: 1.0e-8
    norm_drift: 1.0e-8
    min_fidelity: 0.999999999
""",
        encoding="utf-8",
    )

    def fake_quest_route(root_dir, run_dir, suite, case_payload, graph, network):
        output, _ = execute_task_sequence_np_einsum(graph, network)
        state = tensor_to_quest_statevector(output)
        return RouteResult(
            "quest_cpu_full_state_exact",
            "quest",
            "passed",
            RouteOutput("statevector", array=state, shape=state.shape, dtype=str(state.dtype), metadata={}),
            ExecutionProfile(kernel_s=0.001, total_s=0.002),
            None,
            "unavailable",
            None,
            {"quest": {"status": "ok"}},
        )

    monkeypatch.setattr("quantum_bench.bench.simulation_backend_compare._run_quest_route", fake_quest_route)

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")

    assert result.status == "completed"
    assert (result.run_dir / "run_manifest.json").exists()
    assert (result.run_dir / "artifact_retention_manifest.json").exists()
    assert (result.run_dir / "normalized_records.jsonl").exists()
    records = load_result_records([result.run_dir])
    assert {record["execution_model"] for record in records} == {"full_state", "tensor_network"}
    assert {record["route_id"] for record in records} == {"quest_cpu_full_state_exact", "cpu_tn_einsum_exact"}
    assert all(record["validation_status"] == "passed" for record in records)
    assert all(record["contraction_execution_target"] == "cpu" for record in records)


def test_simulation_backend_compare_suite_loads() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_quick.yml")

    assert "cpu_tn_einsum_exact" in suite["route_policy"]["routes"]
    assert "quest_cpu_full_state_exact" in suite["route_policy"]["routes"]
    assert all(case["circuit"]["kind"] == "quest_compatible" for case in suite["cases"])
