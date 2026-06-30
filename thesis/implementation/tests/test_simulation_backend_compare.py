from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.result_artifacts import load_result_records
from quantum_bench.bench.simulation_backend_probe import probe_simulation_backends
from quantum_bench.bench.simulation_backend_compare import run_simulation_backend_compare
from quantum_bench.bench.config import route_config_for
from quantum_bench.circuits import quest_compatible_circuit
from quantum_bench.core.records import BenchmarkContext, ExecutionProfile, RouteCapabilities, RouteIdentity, RouteOutput, RouteProbe, RouteResult
from quantum_bench.providers.exact_tn import CpuTnEinsumExactRoute
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
  warmups: 1
  repeats: 2
  planner:
    engine: opt_einsum
    optimize: greedy
metadata:
  validation_method: full_statevector
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
    role: comparison_anchor
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

    class FakeQuestRoute:
        name = "quest_cpu_full_state_exact"
        backend_family = "quest"
        identity = RouteIdentity(
            route_id=name,
            display_name="fake quest",
            role="baseline",
            simulation_method="full_state_vector",
            kernel_family="full_state_vector",
            hardware_target="cpu",
            execution_mode="test",
            output_contract="statevector",
            validation_mode="compare_statevector",
        )

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover - not used by this harness
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            output, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            state = tensor_to_quest_statevector(output)
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("statevector", array=state, shape=state.shape, dtype=str(state.dtype), metadata={}),
                ExecutionProfile(kernel_s=0.001, total_s=0.002),
                None,
                "unavailable",
                None,
                {"quest": {"status": "ok"}},
            )

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {"cpu_tn_einsum_exact": CpuTnEinsumExactRoute(), "quest_cpu_full_state_exact": FakeQuestRoute()},
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")

    assert result.status == "completed"
    assert (result.run_dir / "run_manifest.json").exists()
    assert (result.run_dir / "artifact_retention_manifest.json").exists()
    assert (result.run_dir / "normalized_records.jsonl").exists()
    summary_md = (result.run_dir / "comparison_summary.md").read_text(encoding="utf-8")
    records = load_result_records([result.run_dir])
    assert {record["execution_model"] for record in records} == {"full_state", "tensor_network"}
    assert {record["route_id"] for record in records} == {"quest_cpu_full_state_exact", "cpu_tn_einsum_exact"}
    assert {record["repeat_id"] for record in records} == {0, 1}
    assert {record["measured_repeat_count"] for record in records} == {2}
    assert all(record["simulation_compute_time_s"] is not None for record in records)
    assert all(record["validation_method"] == "full_statevector" for record in records)
    assert all(record["timing_scope"] == "end_to_end_and_compute" for record in records)
    assert all(record["validation_status"] == "passed" for record in records)
    assert all(record["contraction_execution_target"] == "cpu" for record in records)
    assert "## Backend Metadata" in summary_md
    assert "## Output Agreement" in summary_md


def test_simulation_backend_compare_suite_loads() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_quick.yml")

    assert "cpu_tn_einsum_exact" in suite["route_policy"]["routes"]
    assert "quest_cpu_full_state_exact" in suite["route_policy"]["routes"]
    assert "quimb_tn_exact" in suite["route_policy"]["routes"]
    assert all(case["circuit"]["kind"] == "quest_compatible" for case in suite["cases"])

    thesis_small = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_thesis_small.yml")
    scaling = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_scaling.yml")
    compute_medium = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_compute_medium.yml")
    gpu_medium = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_gpu_medium.yml")
    for loaded in (thesis_small, scaling):
        assert loaded["route_policy"]["routes"] == ["cpu_tn_einsum_exact", "quest_cpu_full_state_exact", "quimb_tn_exact"]
        assert loaded["metadata"]["deterministic_unitary_only"] is True
        assert loaded["metadata"]["statevector_cap_qubits"] <= 8
        assert loaded["metadata"]["expected_runtime_class"]
        assert all(case["circuit"]["kind"] == "quest_compatible" for case in loaded["cases"])
    assert compute_medium["metadata"]["gpu_independent"] is True
    assert "gpu" not in " ".join(compute_medium["route_policy"]["routes"])
    assert compute_medium["repeats"] == 3
    assert compute_medium["warmups"] == 1
    assert gpu_medium["metadata"]["gpu_execution_suite"] is True
    assert "quest_gpu_full_state_exact" in gpu_medium["route_policy"]["routes"]
    assert gpu_medium["_route_configs"][-1]["required"] is False


def test_quimb_tn_exact_matches_internal_task_sequence() -> None:
    routes = route_registry(ROOT)
    route = routes["quimb_tn_exact"]
    assert route.probe().available

    suite = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_quick.yml")
    case = suite["cases"][0]
    circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    context = BenchmarkContext(
        ROOT,
        ROOT / "runs" / "test",
        suite,
        case,
        route_config_for(suite, "quimb_tn_exact"),
        0,
        suite["tolerances"],
        suite.get("timeout_s"),
        suite.get("memory_guard_gib"),
    )

    result = route.execute(route.prepare(graph, network, context), context)
    expected, _ = execute_task_sequence_np_einsum(graph, network)

    assert result.status == "passed"
    assert result.output.array is not None
    assert result.backend_family == "quimb"
    assert result.metadata["dependency_versions"]["quimb"]
    np.testing.assert_allclose(result.output.array, expected, atol=1.0e-12)


def test_simulation_backend_probe_reports_gpu_feasibility_without_records() -> None:
    report = probe_simulation_backends(ROOT)

    assert report["schema_version"] == "simulation_backend_probe_v1"
    assert any(route["route_id"] == "quimb_tn_exact" for route in report["routes"])
    assert report["gpu_probe"]["cuda_only_assumption_used"] is False
    assert report["gpu_probe"]["gpu_execution_backend_added"] is False
    assert report["gpu_probe"]["gpu_benchmark_records_emitted"] is False
    candidates = report["gpu_probe"]["gpu_candidates"]
    assert candidates
    assert {candidate["candidate_category"] for candidate in candidates} >= {"tailored_quantum_gpu", "cuda_quantum_stack", "generic_tensor_gpu"}
    quest_hip = next(candidate for candidate in candidates if candidate["candidate_id"] == "quest_gpu_full_state_hip")
    assert quest_hip["source_support_is_not_benchmark_evidence"] is True
    assert quest_hip["gpu_execution_verified"] is False
    assert all(candidate["benchmark_route_eligible"] == candidate["gpu_execution_verified"] for candidate in candidates)
