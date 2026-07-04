from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantum_bench.bench.config import load_suite
from quantum_bench.bench.result_artifacts import load_result_records
from quantum_bench.bench.simulation_backend_probe import _select_gpu_backend, _verify_gpu_backend, probe_simulation_backends
from quantum_bench.bench.simulation_backend_compare import _resource_guard_skip_reason, run_simulation_backend_compare
from quantum_bench.bench.config import route_config_for
from quantum_bench.circuits import quest_compatible_circuit
from quantum_bench.core.records import BenchmarkContext, ExecutionProfile, RouteCapabilities, RouteIdentity, RouteOutput, RouteProbe, RouteResult
from quantum_bench.providers.exact_tn import CpuTnEinsumExactRoute
from quantum_bench.providers.full_state.quest_gpu import QuestGpuFullStateExactRoute, quest_gpu_verification_path
from quantum_bench.providers import route_registry
from quantum_bench.providers.exact_tn.upmem_sdk_simulator import UpmemTnSdkSimulatorQuantizedRoute
from quantum_bench.tn import build_tensor_network, execute_task_frontier_np_einsum, execute_task_sequence_np_einsum, frontier_waves, plan_task_graph_with_config
from quantum_bench.validation import tensor_to_quest_statevector


ROOT = Path(__file__).resolve().parents[1]


def test_quest_exact_route_is_active_cpu_full_state_route() -> None:
    routes = route_registry(ROOT)

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


def test_quest_compatible_repeat_layers_repeat_ordered_gate_sequence() -> None:
    circuit = quest_compatible_circuit("XOR", {"n_qubits": 4, "repeat_layers": 3})
    gates = [(op.gate, op.wires) for op in circuit.operations]
    layer = [("cx", (0, 1)), ("cx", (1, 2)), ("cx", (2, 3))]

    assert gates == layer * 3
    assert circuit.source["repeat_layers"] == 3


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
    assert not (result.run_dir / "plots").exists()
    assert result.run_dir.parent == tmp_path / "runs" / "evidence" / "unit_simulation_compare" / "simulation_backend_compare"
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "evidence_run"
    assert manifest["route_label"] == "simulation_backend_compare"
    assert manifest["normalized_records"] == "normalized_records.jsonl"
    for derived_name in (
        "comparison_summary.md",
        "simulation_backend_compare_results.csv",
        "simulation_backend_compare_pairs.csv",
    ):
        assert not (result.run_dir / derived_name).exists()
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
    assert all("parallelism_mode" in record for record in records)
    assert all(record["parallelism_evidence_type"] == "executed" for record in records)
    assert all(record["execution_plan_executed"] is True for record in records)
    assert all(record["slicing_enabled"] is False for record in records)
    assert all(record["frontier_scheduler_enabled"] is False for record in records)
    assert all(record["modeled_parallelism_available"] is False for record in records)
    by_route = {record["route_id"]: record for record in records}
    assert by_route["quest_cpu_full_state_exact"]["parallelism_mode"] == "not_applicable"
    assert by_route["quest_cpu_full_state_exact"]["execution_plan_kind"] == "full_state_simulation"
    assert by_route["cpu_tn_einsum_exact"]["parallelism_mode"] == "sequential"
    assert by_route["cpu_tn_einsum_exact"]["execution_plan_kind"] == "sequential_taskgraph"
    roles = {record["route_id"]: record["benchmark_role"] for record in records}
    assert roles["quest_cpu_full_state_exact"] == "serious_full_state_baseline"
    assert roles["cpu_tn_einsum_exact"] == "internal_debug_baseline"


def test_full_state_only_compare_skips_tensor_network_planning(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_cpu_gpu_full_state_only
metadata:
  validation_method: full_statevector
defaults:
  warmups: 0
  repeats: 1
  planner:
    engine: opt_einsum
    optimize: greedy
workloads:
  - id: quest_qrng_4q
    circuit:
      kind: quest_compatible
      name: QRNG
      n_qubits: 4
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    benchmark_role: serious_full_state_baseline
    required: true
  - id: quest_gpu_full_state_exact
    role: gpu_baseline
    benchmark_role: serious_gpu_full_state_baseline
    required: true
validation:
  reference_route: quest_cpu_full_state_exact
  require_output_for_roles:
    - comparison_anchor
    - gpu_baseline
  tolerances:
    max_abs_error: 1.0e-9
    l2_error: 1.0e-8
    max_rel_error: 1.0e-8
    norm_drift: 1.0e-8
    min_fidelity: 0.999999999
""",
        encoding="utf-8",
    )

    class FakeFullStateRoute:
        def __init__(self, name: str, target: str, backend_family: str) -> None:
            self.name = name
            self.backend_family = backend_family
            self.identity = RouteIdentity(
                name,
                name,
                "baseline",
                "full_state_vector",
                "full_state_vector",
                target,
                "test",
                "statevector",
                "compare_statevector",
            )

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph}

        def execute(self, prepared, context):
            n_qubits = prepared["graph"].network.circuit.n_qubits
            state = np.zeros(1 << n_qubits, dtype=np.complex128)
            state[0] = 1.0
            metadata = {}
            if self.name == "quest_gpu_full_state_exact":
                metadata = {
                    "accelerator_kind": "amd_gpu",
                    "gpu_backend_verified": True,
                    "gpu_program_executed": True,
                    "gpu_device_name": "AMD Radeon RX 6600 (gfx1032)",
                    "gpu_runtime_stack": "hip",
                    "gpu_synchronized": True,
                }
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("statevector", array=state, shape=state.shape, dtype=str(state.dtype), metadata={}),
                ExecutionProfile(kernel_s=0.001, total_s=0.002),
                None,
                "unavailable",
                None,
                metadata,
            )

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.build_tensor_network",
        lambda circuit: (_ for _ in ()).throw(AssertionError("TN planning should not run for full-state-only suites")),
    )
    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeFullStateRoute("quest_cpu_full_state_exact", "cpu", "quest"),
            "quest_gpu_full_state_exact": FakeFullStateRoute("quest_gpu_full_state_exact", "amd_gpu", "quest_gpu"),
        },
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])

    assert {record["route_id"] for record in records} == {"quest_cpu_full_state_exact", "quest_gpu_full_state_exact"}
    assert {record["task_count"] for record in records} == {0}
    assert {record["tn_task_count"] for record in records} == {0}
    assert all(record["validation_status"] == "passed" for record in records)
    gpu = next(record for record in records if record["route_id"] == "quest_gpu_full_state_exact")
    assert gpu["contraction_execution_target"] == "gpu"
    assert gpu["gpu_backend_verified"] is True
    assert gpu["gpu_program_executed"] is True


def test_metrics_only_performance_tier_is_not_exact_output_evidence(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_cpu_gpu_performance
metadata:
  validation_method: native_status_gate_counts
  performance_tier: true
defaults:
  warmups: 0
  repeats: 1
  planner: {engine: opt_einsum, optimize: greedy}
workloads:
  - id: quest_qrng_4q_perf
    circuit: {kind: quest_compatible, name: QRNG, n_qubits: 4, repeat_layers: 4}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
    options: {state_output_mode: none, validation_method: native_status_gate_counts, performance_tier: true}
  - id: quest_gpu_full_state_exact
    role: gpu_baseline
    required: true
    options: {state_output_mode: none, validation_method: native_status_gate_counts, performance_tier: true}
validation:
  reference_route: quest_cpu_full_state_exact
  require_output_for_roles: []
  tolerances:
    max_abs_error: 1.0e-9
    l2_error: 1.0e-8
    max_rel_error: 1.0e-8
    norm_drift: 1.0e-8
    min_fidelity: 0.999999999
""",
        encoding="utf-8",
    )

    class FakeMetricsOnlyRoute:
        def __init__(self, name: str, target: str, backend_family: str) -> None:
            self.name = name
            self.backend_family = backend_family
            self.identity = RouteIdentity(
                name,
                name,
                "baseline",
                "full_state_vector",
                "full_state_vector",
                target,
                "test",
                "statevector",
                "compare_statevector",
            )

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph}

        def execute(self, prepared, context):
            metadata = {
                "quest": {"status": "ok", "one_qubit_gates": 16, "two_qubit_gates": 0},
                "state_output_mode": "none",
                "output_contract": "metrics_only",
                "output_contract_label": "metrics_only",
                "output_contract_is_exact": False,
                "output_contract_note": "Runtime/performance tier only; no full statevector artifact was requested.",
                "validation_method": "native_status_gate_counts",
                "validation_status": "passed_native_status",
                "performance_tier": True,
                "exact_output_comparable": False,
                "full_statevector_validation_available": False,
                "native_process_wall_time_s": 0.01,
                "quest_simulation_compute_time_s": 0.004,
                "state_dump_requested": False,
                "state_dump_time_s": 0.0,
                "repeat_layers": 4,
                "energy_measurement_status": "unavailable",
            }
            if self.name == "quest_gpu_full_state_exact":
                metadata.update(
                    {
                        "accelerator_kind": "amd_gpu",
                        "gpu_backend_verified": True,
                        "gpu_program_executed": True,
                        "gpu_device_name": "AMD Radeon RX 6600 (gfx1032)",
                        "gpu_runtime_stack": "hip",
                        "gpu_synchronized": True,
                    }
                )
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("metrics_only", array=None, metadata={"output_kind": "metrics_only", **metadata}),
                ExecutionProfile(kernel_s=0.004, total_s=0.01),
                None,
                "unavailable",
                None,
                metadata,
            )

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeMetricsOnlyRoute("quest_cpu_full_state_exact", "cpu", "quest"),
            "quest_gpu_full_state_exact": FakeMetricsOnlyRoute("quest_gpu_full_state_exact", "amd_gpu", "quest_gpu"),
        },
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])

    assert {record["output_kind"] for record in records} == {"metrics_only"}
    assert {record["validation_status"] for record in records} == {"passed_native_status"}
    assert {record["state_output_mode"] for record in records} == {"none"}
    assert all(record["performance_tier"] is True for record in records)
    assert all(record["exact_output_comparable"] is False for record in records)
    assert all(record["full_statevector_validation_available"] is False for record in records)
    assert all(record["statevector_bytes"] is None for record in records)
    assert all(record["validation_method"] == "native_status_gate_counts" for record in records)
    assert all(record["parallelism_mode"] == "not_applicable" for record in records)
    assert all(record["parallelism_evidence_type"] == "executed" for record in records)
    assert all(record["execution_plan_kind"] == "full_state_simulation" for record in records)
    assert all(record["execution_plan_executed"] is True for record in records)


def test_simulation_backend_compare_suite_loads() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "simulation_backend_compare_quick.yml")

    assert "cpu_tn_einsum_exact" in suite["route_policy"]["routes"]
    assert "quest_cpu_full_state_exact" in suite["route_policy"]["routes"]
    assert "quimb_tn_exact" in suite["route_policy"]["routes"]
    assert all(case["circuit"]["kind"] == "quest_compatible" for case in suite["cases"])

    thesis_small = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "simulation_backend_compare_thesis_small.yml")
    scaling = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "simulation_backend_compare_scaling.yml")
    compute_medium = load_suite(ROOT / "configs" / "suites" / "cpu_evidence.yml")
    compute_large = load_suite(ROOT / "configs" / "suites" / "manual_large.yml")
    gpu_medium = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "simulation_backend_compare_gpu_medium.yml")
    gpu_execution_only = load_suite(ROOT / "configs" / "suites" / "gpu_evidence.yml")
    cpu_gpu_sweep = load_suite(ROOT / "configs" / "suites" / "cpu_gpu_sweep.yml")
    cpu_gpu_tier1 = load_suite(ROOT / "configs" / "suites" / "manual" / "cpu_gpu_sweep_tier1.yml")
    cpu_gpu_tier2 = load_suite(ROOT / "configs" / "suites" / "manual" / "cpu_gpu_sweep_tier2.yml")
    cpu_gpu_correctness_deep = load_suite(ROOT / "configs" / "suites" / "manual" / "cpu_gpu_correctness_deep.yml")
    cpu_gpu_performance = load_suite(ROOT / "configs" / "suites" / "manual" / "cpu_gpu_performance.yml")
    upmem_sdk = load_suite(ROOT / "configs" / "suites" / "upmem_sim_evidence.yml")
    quimb_slicing = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "quimb_slicing_quick.yml")
    cpu_frontier = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "cpu_frontier_quick.yml")
    for loaded in (thesis_small, scaling):
        assert loaded["route_policy"]["routes"] == ["cpu_tn_einsum_exact", "quest_cpu_full_state_exact", "quimb_tn_exact"]
        assert loaded["metadata"]["deterministic_unitary_only"] is True
        assert loaded["metadata"]["statevector_cap_qubits"] <= 8
        assert loaded["metadata"]["expected_runtime_class"]
        assert all(case["circuit"]["kind"] == "quest_compatible" for case in loaded["cases"])
    assert compute_medium["metadata"]["gpu_independent"] is True
    assert "gpu" not in " ".join(compute_medium["route_policy"]["routes"])
    assert compute_medium["route_policy"]["routes"] == ["quest_cpu_full_state_exact", "quimb_tn_exact"]
    assert compute_medium["repeats"] == 3
    assert compute_medium["warmups"] == 1
    compute_routes = {route["id"]: route for route in compute_medium["_route_configs"]}
    assert "cpu_tn_einsum_exact" not in compute_routes
    assert compute_routes["quest_cpu_full_state_exact"]["benchmark_role"] == "serious_full_state_baseline"
    assert compute_routes["quimb_tn_exact"]["required"] is True
    assert compute_routes["quimb_tn_exact"]["benchmark_role"] == "serious_external_tn_baseline"
    assert compute_large["metadata"]["manual_invocation_required"] is True
    assert compute_large["metadata"]["expected_runtime_class"] == "manual_large"
    assert compute_large["metadata"]["expected_memory_class"] == "workstation_high"
    assert compute_large["metadata"]["intended_use"] == "thesis_evidence"
    assert compute_large["metadata"]["max_qubits"] == 12
    assert "high_memory" in compute_large["metadata"]["expected_risk"]
    large_routes = {route["id"]: route for route in compute_large["_route_configs"]}
    assert large_routes["quimb_tn_exact"]["benchmark_role"] == "serious_external_tn_baseline"
    assert large_routes["cpu_tn_einsum_exact"]["role"] == "optional_diagnostic"
    assert large_routes["cpu_tn_einsum_exact"]["benchmark_role"] == "internal_debug_baseline"
    assert "not a tensor-network approach limitation" in large_routes["cpu_tn_einsum_exact"]["route_limitation_scope"]
    for case in compute_large["cases"]:
        circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
        build_tensor_network(circuit)
    assert gpu_medium["metadata"]["gpu_execution_suite"] is True
    assert "quest_gpu_full_state_exact" in gpu_medium["route_policy"]["routes"]
    assert gpu_medium["_route_configs"][-1]["required"] is False
    assert gpu_execution_only["metadata"]["gpu_execution_suite"] is True
    assert gpu_execution_only["route_policy"]["routes"] == ["quest_cpu_full_state_exact", "quest_gpu_full_state_exact"]
    assert gpu_execution_only["warmups"] == 0
    assert gpu_execution_only["repeats"] == 1
    assert all(route["id"] not in {"cpu_tn_einsum_exact", "quimb_tn_exact"} for route in gpu_execution_only["_route_configs"])
    assert gpu_execution_only["_route_configs"][1]["required"] is False
    assert quimb_slicing["metadata"]["intended_use"] == "diagnostics"
    assert quimb_slicing["route_policy"]["routes"] == ["quest_cpu_full_state_exact", "quimb_tn_exact", "quimb_tn_sliced_exact"]
    sliced_route = route_config_for(quimb_slicing, "quimb_tn_sliced_exact")
    assert sliced_route["benchmark_role"] == "explicit_slicing_evidence"
    assert sliced_route["options"]["target_slices"] == 2
    assert sliced_route["options"]["require_slicing"] is True
    assert cpu_frontier["metadata"]["intended_use"] == "diagnostics"
    assert cpu_frontier["route_policy"]["routes"] == ["quest_cpu_full_state_exact", "cpu_tn_einsum_exact", "cpu_tn_frontier_exact"]
    frontier_route = route_config_for(cpu_frontier, "cpu_tn_frontier_exact")
    assert frontier_route["benchmark_role"] == "internal_frontier_diagnostic"
    assert frontier_route["options"]["frontier_worker_count"] == 2
    assert cpu_gpu_sweep["suite_id"] == "cpu_gpu_sweep"
    assert cpu_gpu_sweep["route_policy"]["routes"] == ["quest_cpu_full_state_exact", "quest_gpu_full_state_exact"]
    assert cpu_gpu_sweep["warmups"] == 1
    assert cpu_gpu_sweep["repeats"] == 3
    assert cpu_gpu_sweep["metadata"]["max_qubits"] == 18
    assert cpu_gpu_sweep["metadata"]["manual_invocation_required"] is True
    sweep_routes = {route["id"]: route for route in cpu_gpu_sweep["_route_configs"]}
    assert sweep_routes["quest_cpu_full_state_exact"]["required"] is True
    assert sweep_routes["quest_gpu_full_state_exact"]["required"] is True
    assert sweep_routes["quest_gpu_full_state_exact"]["benchmark_role"] == "serious_gpu_full_state_baseline"
    assert {case["circuit"]["name"] for case in cpu_gpu_sweep["cases"]} == {"QRNG", "BV", "XOR", "BB84", "EDC", "HS"}
    sweep_qubits = {
        int(case["circuit"].get("n_qubits", case["circuit"].get("allocated_qubits")))
        for case in cpu_gpu_sweep["cases"]
    }
    assert sweep_qubits == {4, 6, 8, 10, 12, 14, 16, 18}
    for case in cpu_gpu_sweep["cases"]:
        if case["circuit"]["name"] == "HS":
            assert case["circuit"]["logical_qubits"] == case["circuit"]["allocated_qubits"] // 2
        circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
        assert circuit.source["deterministic_unitary"] is True
        assert circuit.n_qubits <= 18
    assert cpu_gpu_tier1["suite_id"] == "cpu_gpu_sweep"
    assert cpu_gpu_tier2["suite_id"] == "cpu_gpu_sweep"
    tier1_qubits = {
        int(case["circuit"].get("n_qubits", case["circuit"].get("allocated_qubits")))
        for case in cpu_gpu_tier1["cases"]
    }
    tier2_qubits = {
        int(case["circuit"].get("n_qubits", case["circuit"].get("allocated_qubits")))
        for case in cpu_gpu_tier2["cases"]
    }
    assert tier1_qubits == {4, 6, 8, 10, 12}
    assert tier2_qubits == {14, 16, 18}
    assert {case["case_id"] for case in cpu_gpu_tier1["cases"]} | {case["case_id"] for case in cpu_gpu_tier2["cases"]} == {
        case["case_id"] for case in cpu_gpu_sweep["cases"]
    }
    assert cpu_gpu_correctness_deep["metadata"]["state_output_mode"] == "full_dump"
    assert cpu_gpu_correctness_deep["metadata"]["validation_method"] == "full_statevector"
    assert cpu_gpu_correctness_deep["metadata"]["performance_tier"] is False
    assert cpu_gpu_performance["metadata"]["state_output_mode"] == "none"
    assert cpu_gpu_performance["metadata"]["validation_method"] == "native_status_gate_counts"
    assert cpu_gpu_performance["metadata"]["performance_tier"] is True
    assert cpu_gpu_performance["validation"]["require_output_for_roles"] == []
    performance_routes = {route["id"]: route for route in cpu_gpu_performance["_route_configs"]}
    assert performance_routes["quest_cpu_full_state_exact"]["options"]["state_output_mode"] == "none"
    assert performance_routes["quest_gpu_full_state_exact"]["options"]["state_output_mode"] == "none"
    assert all(case["circuit"]["repeat_layers"] >= 64 for case in cpu_gpu_performance["cases"])
    assert upmem_sdk["route_policy"]["routes"] == [
        "quest_cpu_full_state_exact",
        "quimb_tn_exact",
        "upmem_tn_sdk_simulator_quantized",
    ]
    assert "quest_gpu_full_state_exact" not in upmem_sdk["route_policy"]["routes"]
    upmem_routes = {route["id"]: route for route in upmem_sdk["_route_configs"]}
    assert upmem_routes["upmem_tn_sdk_simulator_quantized"]["required"] is False
    assert upmem_routes["upmem_tn_sdk_simulator_quantized"]["options"]["execute_external"] is True
    assert upmem_routes["upmem_tn_sdk_simulator_quantized"]["options"]["quantization_mode"] == "per_task_input_quantize"
    assert upmem_sdk["metadata"]["validation_tier"] == "quantized_codepath"


def test_resource_guard_missing_estimate_policy_is_explicit() -> None:
    graph = SimpleNamespace(path_summary=SimpleNamespace(max_intermediate_bytes=None, total_estimated_flops=None))

    assert _resource_guard_skip_reason({"options": {"max_estimated_intermediate_bytes": 1}}, graph) == "unavailable_estimate"
    assert _resource_guard_skip_reason({"options": {"max_estimated_intermediate_bytes": 1, "allow_missing_estimate": True}}, graph) is None


def test_resource_guard_skips_optional_route_before_execution(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_guarded_compare
metadata:
  expected_runtime_class: manual_large
  expected_memory_class: workstation_high
  intended_use: thesis_evidence
  max_qubits: 3
  manual_invocation_required: true
  expected_risk: [high_memory]
  known_heavy_backends: [cpu_tn_einsum_exact]
defaults:
  warmups: 0
  repeats: 1
  planner:
    engine: opt_einsum
    optimize: greedy
workloads:
  - id: quest_bv_3q
    circuit: {kind: quest_compatible, name: BV, n_qubits: 3}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: cpu_tn_einsum_exact
    role: baseline
    required: false
    options:
      max_estimated_intermediate_bytes: 1
      resource_skip_reason: unit_guard
validation:
  reference_route: quest_cpu_full_state_exact
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
        identity = RouteIdentity(name, "fake quest", "baseline", "full_state_vector", "full_state_vector", "cpu", "test", "statevector", "compare_statevector")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
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
            )

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {"quest_cpu_full_state_exact": FakeQuestRoute(), "cpu_tn_einsum_exact": CpuTnEinsumExactRoute()},
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])
    skipped = next(record for record in records if record["route_id"] == "cpu_tn_einsum_exact")

    assert skipped["status"] == "not_executed"
    assert skipped["validation_status"] == "skipped"
    assert skipped["resource_guard_status"] == "resource_guard_skipped"
    assert "unit_guard" in skipped["resource_skip_reason"]
    assert skipped["manual_invocation_required"] is True
    assert skipped["expected_runtime_class"] == "manual_large"


def test_optional_route_memory_error_becomes_normalized_skip(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_memory_error_compare
metadata:
  expected_runtime_class: manual_large
workloads:
  - id: quest_bv_3q
    circuit: {kind: quest_compatible, name: BV, n_qubits: 3}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: quimb_tn_exact
    role: baseline
    required: false
validation:
  reference_route: quest_cpu_full_state_exact
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
        identity = RouteIdentity(name, "fake quest", "baseline", "full_state_vector", "full_state_vector", "cpu", "test", "statevector", "compare_statevector")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            output, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("statevector", array=tensor_to_quest_statevector(output), shape=(8,), dtype="complex128", metadata={}),
                ExecutionProfile(kernel_s=0.001, total_s=0.002),
                None,
                "unavailable",
            )

    class MemoryErrorRoute:
        name = "quimb_tn_exact"
        backend_family = "quimb"
        identity = RouteIdentity(name, "fake quimb", "baseline", "exact_tensor_network", "external_tn_contraction", "cpu", "test", "final_tensor", "compare_output")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {}

        def execute(self, prepared, context):
            raise MemoryError("unit memory limit")

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {"quest_cpu_full_state_exact": FakeQuestRoute(), "quimb_tn_exact": MemoryErrorRoute()},
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])
    failed_optional = next(record for record in records if record["route_id"] == "quimb_tn_exact")

    assert failed_optional["status"] == "failed"
    assert failed_optional["validation_status"] == "skipped"
    assert failed_optional["resource_guard_status"] == "execution_failed"
    assert failed_optional["resource_skip_reason"].startswith("memory_error:")


def test_required_internal_debug_memory_error_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_required_debug_memory_error_compare
metadata:
  expected_runtime_class: manual_large
workloads:
  - id: quest_bv_3q
    circuit: {kind: quest_compatible, name: BV, n_qubits: 3}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: cpu_tn_einsum_exact
    role: optional_diagnostic
    benchmark_role: internal_debug_baseline
    required: true
validation:
  reference_route: quest_cpu_full_state_exact
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
        identity = RouteIdentity(name, "fake quest", "baseline", "full_state_vector", "full_state_vector", "cpu", "test", "statevector", "compare_statevector")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            output, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("statevector", array=tensor_to_quest_statevector(output), shape=(8,), dtype="complex128", metadata={}),
                ExecutionProfile(kernel_s=0.001, total_s=0.002),
                None,
                "unavailable",
            )

    class DebugMemoryErrorRoute:
        name = "cpu_tn_einsum_exact"
        backend_family = "internal"
        identity = RouteIdentity(name, "fake cpu tn", "baseline", "exact_tensor_network", "cpu_einsum", "cpu", "test", "final_tensor", "compare_output")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {}

        def execute(self, prepared, context):
            raise MemoryError("diagnostic route memory limit")

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {"quest_cpu_full_state_exact": FakeQuestRoute(), "cpu_tn_einsum_exact": DebugMemoryErrorRoute()},
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])
    failed_debug = next(record for record in records if record["route_id"] == "cpu_tn_einsum_exact")

    assert failed_debug["status"] == "failed"
    assert failed_debug["benchmark_role"] == "internal_debug_baseline"
    assert failed_debug["validation_status"] == "skipped"
    assert failed_debug["resource_guard_status"] == "execution_failed"
    assert failed_debug["resource_skip_reason"].startswith("memory_error:")


def test_required_serious_memory_error_remains_fatal(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_required_serious_memory_error_compare
metadata:
  expected_runtime_class: manual_large
workloads:
  - id: quest_bv_3q
    circuit: {kind: quest_compatible, name: BV, n_qubits: 3}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: quimb_tn_exact
    role: baseline
    benchmark_role: serious_external_tn_baseline
    required: true
validation:
  reference_route: quest_cpu_full_state_exact
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
        identity = RouteIdentity(name, "fake quest", "baseline", "full_state_vector", "full_state_vector", "cpu", "test", "statevector", "compare_statevector")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            output, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("statevector", array=tensor_to_quest_statevector(output), shape=(8,), dtype="complex128", metadata={}),
                ExecutionProfile(kernel_s=0.001, total_s=0.002),
                None,
                "unavailable",
            )

    class SeriousMemoryErrorRoute:
        name = "quimb_tn_exact"
        backend_family = "quimb"
        identity = RouteIdentity(name, "fake quimb", "baseline", "exact_tensor_network", "external_tn_contraction", "cpu", "test", "final_tensor", "compare_output")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {}

        def execute(self, prepared, context):
            raise MemoryError("serious route memory limit")

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {"quest_cpu_full_state_exact": FakeQuestRoute(), "quimb_tn_exact": SeriousMemoryErrorRoute()},
    )

    with pytest.raises(MemoryError, match="serious route memory limit"):
        run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")


def test_optional_route_value_error_becomes_normalized_skip(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_value_error_compare
metadata:
  expected_runtime_class: manual_large
workloads:
  - id: quest_bv_3q
    circuit: {kind: quest_compatible, name: BV, n_qubits: 3}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: cpu_tn_einsum_exact
    role: baseline
    required: false
validation:
  reference_route: quest_cpu_full_state_exact
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
        identity = RouteIdentity(name, "fake quest", "baseline", "full_state_vector", "full_state_vector", "cpu", "test", "statevector", "compare_statevector")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            output, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            return RouteResult(
                self.name,
                self.backend_family,
                "passed",
                RouteOutput("statevector", array=tensor_to_quest_statevector(output), shape=(8,), dtype="complex128", metadata={}),
                ExecutionProfile(kernel_s=0.001, total_s=0.002),
                None,
                "unavailable",
            )

    class ValueErrorRoute:
        name = "cpu_tn_einsum_exact"
        backend_family = "internal"
        identity = RouteIdentity(name, "fake cpu tn", "baseline", "exact_tensor_network", "cpu_einsum", "cpu", "test", "final_tensor", "compare_output")

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {}

        def execute(self, prepared, context):
            raise ValueError("Too many tensor indices for NumPy einsum symbol set")

    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {"quest_cpu_full_state_exact": FakeQuestRoute(), "cpu_tn_einsum_exact": ValueErrorRoute()},
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])
    failed_optional = next(record for record in records if record["route_id"] == "cpu_tn_einsum_exact")

    assert failed_optional["status"] == "failed"
    assert failed_optional["validation_status"] == "skipped"
    assert failed_optional["resource_guard_status"] == "execution_failed"
    assert failed_optional["resource_skip_reason"].startswith("value_error:")


def test_quimb_tn_exact_matches_internal_task_sequence() -> None:
    routes = route_registry(ROOT)
    route = routes["quimb_tn_exact"]
    assert route.probe().available

    suite = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "simulation_backend_compare_quick.yml")
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


def test_cpu_tn_frontier_exact_matches_sequential_with_worker_counts() -> None:
    routes = route_registry(ROOT)
    route = routes["cpu_tn_frontier_exact"]
    suite = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "cpu_frontier_quick.yml")
    case = suite["cases"][0]
    circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    expected, _ = execute_task_sequence_np_einsum(graph, network)
    assert max(len(wave) for wave in frontier_waves(graph)) > 1

    for worker_count in (1, 2):
        route_config = {
            **route_config_for(suite, "cpu_tn_frontier_exact"),
            "options": {"frontier_worker_count": worker_count},
        }
        context = BenchmarkContext(
            ROOT,
            ROOT / "runs" / "test",
            suite,
            case,
            route_config,
            0,
            suite["tolerances"],
            suite.get("timeout_s"),
            suite.get("memory_guard_gib"),
        )
        result = route.execute(route.prepare(graph, network, context), context)

        assert result.status == "passed"
        assert result.output.array is not None
        assert result.metadata["parallelism_mode"] == "frontier"
        assert result.metadata["frontier_scheduler_enabled"] is True
        assert result.metadata["frontier_worker_count"] == worker_count
        assert result.metadata["frontier_parallel_execution"] is (worker_count > 1)
        assert result.metadata["frontier_wave_count"] > 1
        assert result.metadata["max_frontier_width"] > 1
        if worker_count > 1:
            assert 0 < result.metadata["executed_parallel_task_count"] <= len(graph.tasks)
        else:
            assert result.metadata["executed_parallel_task_count"] == 0
        assert result.metadata["duplicate_contraction_check"] == "passed"
        assert result.metadata["missing_dependency_check"] == "passed"
        np.testing.assert_allclose(result.output.array, expected, atol=1.0e-12)


def test_cpu_frontier_executor_rejects_unresolved_dependencies() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "cpu_frontier_quick.yml")
    case = suite["cases"][0]
    circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    broken_first_task = replace(graph.tasks[0], dependencies=("missing_task",))
    broken_graph = replace(graph, tasks=(broken_first_task, *graph.tasks[1:]))

    with pytest.raises(ValueError, match="cyclic or unresolved"):
        execute_task_frontier_np_einsum(broken_graph, network, frontier_worker_count=2)


def test_cpu_frontier_diagnostic_suite_records_frontier_metadata(tmp_path: Path, monkeypatch) -> None:
    class FakeQuestRoute:
        name = "quest_cpu_full_state_exact"
        backend_family = "quest"
        identity = RouteIdentity(
            route_id=name,
            display_name="Fake QuEST CPU",
            role="serious_full_state_baseline",
            simulation_method="exact_full_state",
            kernel_family="full_state_vector",
            hardware_target="cpu",
            execution_mode="in_process_fake",
            output_contract="statevector",
            validation_mode="compare_statevector",
        )

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(identity=self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            expected, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            statevector = tensor_to_quest_statevector(expected)
            return RouteResult(
                route=self.name,
                backend_family=self.backend_family,
                status="passed",
                output=RouteOutput(contract="statevector", array=statevector),
                profile=ExecutionProfile(total_s=0.01, kernel_s=0.01),
                energy_joules=None,
                energy_source="unavailable",
                metadata={},
            )

    real_routes = route_registry(ROOT)
    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeQuestRoute(),
            "cpu_tn_einsum_exact": real_routes["cpu_tn_einsum_exact"],
            "cpu_tn_frontier_exact": real_routes["cpu_tn_frontier_exact"],
            "quimb_tn_sliced_exact": real_routes["quimb_tn_sliced_exact"],
        },
    )
    result = run_simulation_backend_compare(
        tmp_path,
        suite_path=ROOT / "configs" / "suites" / "diagnostics" / "cpu_frontier_quick.yml",
        artifact_retention="compact",
    )
    records = load_result_records([result.run_dir])
    frontier_records = [record for record in records if record["route_id"] == "cpu_tn_frontier_exact"]
    sequential_records = [record for record in records if record["route_id"] == "cpu_tn_einsum_exact"]
    slicing_records = [record for record in records if record["route_id"] == "quimb_tn_sliced_exact"]

    assert frontier_records
    assert not slicing_records
    assert all(record["validation_status"] == "passed" for record in frontier_records)
    assert all(record["parallelism_mode"] == "frontier" for record in frontier_records)
    assert all(record["parallelism_evidence_type"] == "executed" for record in frontier_records)
    assert all(record["execution_plan_kind"] == "taskgraph_frontier_scheduler" for record in frontier_records)
    assert all(record["execution_plan_executed"] is True for record in frontier_records)
    assert all(record["frontier_scheduler_enabled"] is True for record in frontier_records)
    assert all(record["frontier_parallel_execution"] is True for record in frontier_records)
    assert all(record["frontier_worker_count"] == 2 for record in frontier_records)
    assert all(record["frontier_wave_count"] > 1 for record in frontier_records)
    assert all(record["max_frontier_width"] > 1 for record in frontier_records)
    assert all(record["executed_parallel_task_count"] > 0 for record in frontier_records)
    assert all(record["duplicate_contraction_check"] == "passed" for record in frontier_records)
    assert all(record["missing_dependency_check"] == "passed" for record in frontier_records)
    assert all(record["frontier_scheduler_enabled"] is False for record in sequential_records)
    assert all(record["parallelism_mode"] == "sequential" for record in sequential_records)


def test_quimb_tn_sliced_exact_executes_sliced_tree() -> None:
    routes = route_registry(ROOT)
    route = routes["quimb_tn_sliced_exact"]
    assert route.probe().available

    suite = load_suite(ROOT / "configs" / "suites" / "diagnostics" / "quimb_slicing_quick.yml")
    case = suite["cases"][0]
    circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    context = BenchmarkContext(
        ROOT,
        ROOT / "runs" / "test",
        suite,
        case,
        route_config_for(suite, "quimb_tn_sliced_exact"),
        0,
        suite["tolerances"],
        suite.get("timeout_s"),
        suite.get("memory_guard_gib"),
    )

    result = route.execute(route.prepare(graph, network, context), context)
    expected, _ = execute_task_sequence_np_einsum(graph, network)

    assert result.status == "passed"
    assert result.output.array is not None
    assert result.metadata["slicing_enabled"] is True
    assert result.metadata["slicing_backend"] == "cotengra"
    assert result.metadata["slice_count"] > 1
    assert result.metadata["sliced_indices"]
    assert result.metadata["slicing_reconstruction_status"] == "completed"
    assert result.metadata["slice_parallel_execution"] is False
    assert result.metadata["slice_worker_count"] == 1
    assert result.metadata["execution_plan_kind"] == "cotengra_sliced_contraction_tree"
    np.testing.assert_allclose(result.output.array, expected, atol=1.0e-12)


def test_quimb_slicing_diagnostic_suite_records_slicing_metadata(tmp_path: Path, monkeypatch) -> None:
    class FakeQuestRoute:
        name = "quest_cpu_full_state_exact"
        backend_family = "quest"
        identity = RouteIdentity(
            route_id=name,
            display_name="Fake QuEST CPU",
            role="serious_full_state_baseline",
            simulation_method="exact_full_state",
            kernel_family="full_state_vector",
            hardware_target="cpu",
            execution_mode="in_process_fake",
            output_contract="statevector",
            validation_mode="compare_statevector",
        )

        def probe(self):
            return RouteProbe(self.name, True)

        def capabilities(self):
            return RouteCapabilities(identity=self.identity, can_return_output=True)

        def can_execute(self, graph, context):
            return True, None

        def estimate(self, graph, context):  # pragma: no cover
            raise NotImplementedError

        def prepare(self, graph, network, context):
            return {"graph": graph, "network": network}

        def execute(self, prepared, context):
            expected, _ = execute_task_sequence_np_einsum(prepared["graph"], prepared["network"])
            statevector = tensor_to_quest_statevector(expected)
            return RouteResult(
                route=self.name,
                backend_family=self.backend_family,
                status="passed",
                output=RouteOutput(contract="statevector", array=statevector),
                profile=ExecutionProfile(total_s=0.01, kernel_s=0.01),
                energy_joules=None,
                energy_source="unavailable",
                metadata={},
            )

    real_routes = route_registry(ROOT)
    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeQuestRoute(),
            "quimb_tn_exact": real_routes["quimb_tn_exact"],
            "quimb_tn_sliced_exact": real_routes["quimb_tn_sliced_exact"],
        },
    )
    suite_path = tmp_path / "quimb_slicing_cpu_anchor.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: quimb_slicing_cpu_anchor
defaults:
  warmups: 0
  repeats: 1
  timeout_s: 60
  memory_guard_gib: 4
  planner: {engine: opt_einsum, optimize: greedy}
workloads:
  - id: quest_qrng_3q
    circuit: {kind: quest_compatible, name: QRNG, n_qubits: 3}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: quimb_tn_exact
    role: baseline
    required: true
  - id: quimb_tn_sliced_exact
    role: baseline
    benchmark_role: explicit_slicing_evidence
    required: true
    options:
      methods: greedy
      max_repeats: 1
      slicing_strategy: target_slices
      target_slices: 2
      require_slicing: true
validation:
  reference_route: quest_cpu_full_state_exact
  require_output_for_roles: [baseline, comparison_anchor]
  tolerances:
    max_abs_error: 1.0e-9
    l2_error: 1.0e-8
    max_rel_error: 1.0e-8
    norm_drift: 1.0e-8
    min_fidelity: 0.999999999
""",
        encoding="utf-8",
    )
    result = run_simulation_backend_compare(
        tmp_path,
        suite_path=suite_path,
        artifact_retention="compact",
    )
    records = load_result_records([result.run_dir])
    sliced_records = [record for record in records if record["route_id"] == "quimb_tn_sliced_exact"]
    unsliced_records = [record for record in records if record["route_id"] == "quimb_tn_exact"]

    assert sliced_records
    assert unsliced_records
    assert all(record["validation_status"] == "passed" for record in sliced_records)
    assert all(record["parallelism_mode"] == "slicing" for record in sliced_records)
    assert all(record["parallelism_evidence_type"] == "executed" for record in sliced_records)
    assert all(record["execution_plan_kind"] == "cotengra_sliced_contraction_tree" for record in sliced_records)
    assert all(record["execution_plan_executed"] is True for record in sliced_records)
    assert all(record["slicing_enabled"] is True for record in sliced_records)
    assert all(record["slicing_backend"] == "cotengra" for record in sliced_records)
    assert all(record["slice_count"] > 1 for record in sliced_records)
    assert all(record["sliced_indices"] for record in sliced_records)
    assert all(record["slicing_reconstruction_status"] == "completed" for record in sliced_records)
    assert all(record["slice_parallel_execution"] is False for record in sliced_records)
    assert all(record["slice_worker_count"] == 1 for record in sliced_records)
    assert all(record["slicing_enabled"] is False for record in unsliced_records)
    assert all(record["slice_parallel_execution"] is False for record in unsliced_records)
    assert all(record["parallelism_mode"] == "sequential" for record in unsliced_records)


def test_simulation_backend_probe_reports_gpu_feasibility_without_records() -> None:
    report = probe_simulation_backends(ROOT)

    assert report["schema_version"] == "simulation_backend_probe_v1"
    gpu_route = next(route for route in report["routes"] if route["route_id"] == "quest_gpu_full_state_exact")
    assert gpu_route["available"] == bool(gpu_route["metadata"].get("gpu_backend_verified", False))
    assert any(route["route_id"] == "quimb_tn_exact" for route in report["routes"])
    assert report["gpu_probe"]["cuda_only_assumption_used"] is False
    assert report["gpu_probe"]["gpu_benchmark_records_emitted"] is False
    candidates = report["gpu_probe"]["gpu_candidates"]
    assert candidates
    assert {candidate["candidate_category"] for candidate in candidates} >= {"tailored_quantum_gpu", "cuda_quantum_stack", "generic_tensor_gpu"}
    quest_hip = next(candidate for candidate in candidates if candidate["candidate_id"] == "quest_gpu_full_state_hip")
    assert quest_hip["source_support_is_not_benchmark_evidence"] is True
    assert report["gpu_probe"]["gpu_execution_backend_added"] == bool(quest_hip["gpu_execution_verified"])
    assert all((not candidate["benchmark_route_eligible"]) or candidate["gpu_execution_verified"] for candidate in candidates)


def test_gpu_verify_auto_selects_tailored_backend_by_hardware() -> None:
    assert _select_gpu_backend("auto", {"amd_gpu_pci_detected": True, "nvidia_gpu_pci_detected": False}) == "quest-hip"
    assert _select_gpu_backend("auto", {"amd_gpu_pci_detected": False, "nvidia_gpu_pci_detected": True}) == "quest-cuda"
    assert _select_gpu_backend("auto", {"amd_gpu_pci_detected": False, "nvidia_gpu_pci_detected": False}) is None
    assert _select_gpu_backend("torch-rocm", {"amd_gpu_pci_detected": True}) == "torch-rocm"


def test_failed_gpu_verification_writes_blocker_artifact(tmp_path: Path) -> None:
    report = _verify_gpu_backend(
        tmp_path,
        "auto",
        {
            "amd_gpu_pci_detected": True,
            "nvidia_gpu_pci_detected": False,
            "dev_kfd_present": False,
            "dev_dri_present": False,
        },
    )

    artifact = quest_gpu_verification_path(tmp_path)
    assert artifact.exists()
    assert report["status"] == "blocked"
    assert report["selected_backend"] == "quest-hip"
    assert report["gpu_backend_verified"] is False
    assert "missing_prerequisites" in report
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "quest_gpu_verification_v1"
    assert payload["attempted_steps"][0]["step"] == "preflight"


def test_gpu_verification_blocks_when_hip_smoke_fails(monkeypatch, tmp_path: Path) -> None:
    runner_root = tmp_path / "native" / "quest_gpu"
    runner_root.mkdir(parents=True)

    def fake_run_command(cmd, *, cwd, timeout_s):
        if "hip-smoke" in cmd:
            return {"command": cmd, "cwd": str(cwd), "returncode": 1, "stdout": "", "stderr": "compile failed"}
        raise AssertionError(f"unexpected command after failed HIP smoke build: {cmd}")

    monkeypatch.setattr("quantum_bench.bench.simulation_backend_probe._run_command", fake_run_command)

    report = _verify_gpu_backend(
        tmp_path,
        "quest-hip",
        {
            "amd_gpu_pci_detected": True,
            "nvidia_gpu_pci_detected": False,
            "dev_kfd_present": True,
            "dev_dri_present": True,
            "dev_dri_renderD128_present": True,
            "dev_dri_render_node_present": True,
            "dev_dri_render_nodes": ["/dev/dri/renderD128"],
            "rocminfo_gpu_agent_detected": True,
            "rocminfo_gfx_targets": ["gfx1032"],
            "rocminfo_returncode": 0,
        },
    )

    assert report["status"] == "failed"
    assert report["blocker_reason"] == "hip_smoke_build_failed"
    assert report["gpu_backend_verified"] is False
    assert report["attempted_steps"][1]["step"] == "build_hip_smoke"
    assert all(step["step"] != "build_quest_gpu_runner" for step in report["attempted_steps"])


def test_successful_gpu_verification_requires_hip_smoke_and_quest_run(monkeypatch, tmp_path: Path) -> None:
    runner_root = tmp_path / "native" / "quest_gpu"
    runner = runner_root / "bin" / "quest_gpu_runner"
    runner.parent.mkdir(parents=True)

    def fake_run_command(cmd, *, cwd, timeout_s):
        if "hip-smoke" in cmd:
            return {"command": cmd, "cwd": str(cwd), "returncode": 0, "stdout": "", "stderr": ""}
        if cmd and str(cmd[0]).endswith("hip_smoke"):
            return {
                "command": cmd,
                "cwd": str(cwd),
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "status": "ok",
                        "gpu_backend_verified": True,
                        "gpu_program_executed": True,
                        "gpu_device_name": "AMD Radeon RX 6600",
                        "gcn_arch_name": "gfx1032",
                    }
                ),
                "stderr": "",
            }
        if "clean-all" in cmd and "all" in cmd:
            runner.write_text("#!/bin/sh\n", encoding="utf-8")
            return {"command": cmd, "cwd": str(cwd), "returncode": 0, "stdout": "", "stderr": ""}
        if cmd and str(cmd[0]).endswith("quest_gpu_runner"):
            dump_path = Path(cmd[cmd.index("--dump-state-json") + 1])
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(json.dumps({"basis_order": "little_endian", "real": [1, 0, 0, 0], "imag": [0, 0, 0, 0]}), encoding="utf-8")
            return {"command": cmd, "cwd": str(cwd), "returncode": 0, "stdout": json.dumps({"status": "ok", "time_s": 0.001}), "stderr": ""}
        if cmd[:1] == ["rocm-smi"]:
            return {
                "command": cmd,
                "cwd": str(cwd),
                "returncode": 0,
                "stdout": "GPU[0] : Card Series: AMD Radeon RX 6600\nGPU[0] : GFX Version: gfx1032\n",
                "stderr": "",
            }
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr("quantum_bench.bench.simulation_backend_probe._run_command", fake_run_command)

    report = _verify_gpu_backend(
        tmp_path,
        "quest-hip",
        {
            "amd_gpu_pci_detected": True,
            "nvidia_gpu_pci_detected": False,
            "dev_kfd_present": True,
            "dev_dri_present": True,
            "dev_dri_renderD128_present": True,
            "dev_dri_render_node_present": True,
            "dev_dri_render_nodes": ["/dev/dri/renderD128"],
            "rocminfo_gpu_agent_detected": True,
            "rocminfo_gfx_targets": ["gfx1032"],
            "rocminfo_returncode": 0,
        },
    )

    assert report["status"] == "verified"
    assert report["gpu_backend_verified"] is True
    assert report["gpu_program_executed"] is True
    assert report["gpu_device_name"] == "AMD Radeon RX 6600 (gfx1032)"
    assert [step["step"] for step in report["attempted_steps"]] == [
        "preflight",
        "build_hip_smoke",
        "minimal_hip_smoke_run",
        "build_quest_gpu_runner",
        "minimal_quest_gpu_run",
    ]
    payload = json.loads(quest_gpu_verification_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["hip_smoke_payload"]["gcn_arch_name"] == "gfx1032"


def test_optional_unverified_gpu_route_emits_no_benchmark_record(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_gpu_optional_compare
defaults:
  warmups: 0
  repeats: 1
  planner:
    engine: opt_einsum
    optimize: greedy
metadata:
  validation_method: full_statevector
workloads:
  - id: quest_qrng_2q
    circuit: {kind: quest_compatible, name: QRNG, n_qubits: 2}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: quest_gpu_full_state_exact
    role: optional_gpu_candidate
    required: false
validation:
  reference_route: quest_cpu_full_state_exact
  require_output_for_roles:
    - comparison_anchor
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
            state = np.zeros(4, dtype=np.complex128)
            state[0] = 0.5
            state[1] = 0.5
            state[2] = 0.5
            state[3] = 0.5
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
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeQuestRoute(),
            "quest_gpu_full_state_exact": QuestGpuFullStateExactRoute(tmp_path),
        },
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])

    assert {record["route_id"] for record in records} == {"quest_cpu_full_state_exact"}
    assert all(record["contraction_execution_target"] == "cpu" for record in records)
    assert all(record["gpu_backend_verified"] is False for record in records)


def test_upmem_sdk_simulator_quantized_route_requires_execute_external(tmp_path: Path) -> None:
    route = UpmemTnSdkSimulatorQuantizedRoute()
    suite = load_suite(ROOT / "configs" / "suites" / "upmem_sim_evidence.yml")
    case = suite["cases"][0]
    circuit = quest_compatible_circuit(case["circuit"]["name"], case["circuit"])
    network = build_tensor_network(circuit)
    graph = plan_task_graph_with_config(network, suite["planner"])
    route_config = {"id": route.name, "options": {"execute_external": False}}
    context = BenchmarkContext(ROOT, tmp_path, suite, case, route_config, 0, suite["tolerances"], 30, 2)

    can_execute, reason = route.can_execute(graph, context)

    assert can_execute is False
    assert reason == "upmem_sdk_simulator_execute_external_required"
    preflight = tmp_path / "cases" / case["case_id"] / "routes" / route.name / "repeat_0" / "preflight" / "upmem_sdk_simulator_preflight.json"
    assert preflight.exists()
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    assert payload["status"] == "blocked"


def test_simulation_backend_compare_emits_quantized_upmem_sdk_row(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_upmem_sdk_compare
defaults:
  warmups: 0
  repeats: 1
  planner:
    engine: opt_einsum
    optimize: greedy
metadata:
  validation_method: full_statevector_quantized_tolerance
workloads:
  - id: quest_qrng_2q
    circuit: {kind: quest_compatible, name: QRNG, n_qubits: 2}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: quimb_tn_exact
    role: baseline
    required: true
  - id: upmem_tn_sdk_simulator_quantized
    role: optional_upmem_candidate
    required: true
    options:
      execute_external: true
      policy: dense-then-generic
      quantization_mode: per_task_input_quantize
validation:
  reference_route: quest_cpu_full_state_exact
  require_output_for_roles:
    - comparison_anchor
    - baseline
  tolerances:
    max_abs_error: 0.25
    l2_error: 2.0
    max_rel_error: 10.0
    norm_drift: 2.0
    min_fidelity: 0.0
""",
        encoding="utf-8",
    )

    def fake_upmem_runtime(*, graph, network, case_id, policy, quantization_mode, bridge_root, execute_external, reference_output=None, env=None):
        output = np.asarray(reference_output if reference_output is not None else 2.0 + 0.0j, dtype=np.complex128)
        task_count = max(1, len(graph.tasks))
        summary = {
            "schema_version": "upmem_taskgraph_runtime_v1",
            "case_id": case_id,
            "status": "completed",
            "reason": None,
            "policy": policy,
            "quantization_mode": quantization_mode,
            "whole_network_quantized_at_initialization": False,
            "contraction_execution_target": "upmem",
            "upmem_execution_mode": "sdk_simulator",
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
            "hardware_benchmark_result": False,
            "hardware_timing_available": False,
            "hardware_speedup_applicable": False,
            "cpu_fallback_used": False,
            "dpu_program_executed_all_tasks": True,
            "runtime_tensor_sources_all_upmem_output_blobs": True,
            "valid_primary_upmem_codepath_result": True,
            "total_tasks": task_count,
            "executed_tasks": task_count,
            "unsupported_tasks": 0,
            "failed_tasks": 0,
            "dpu_program_executed_task_count": task_count,
            "kernel_family_counts": {"generic_loop_fallback": task_count},
            "backend_counts": {"upmem_sdk_simulator_generic_loop": task_count},
            "final_validation": {"passed": True, "max_abs_error": 0.0, "l2_error": 0.0, "norm_drift": 0.0},
            "total_wall_time_s": 0.01,
            "total_bridge_time_s": 0.002,
            "total_kernel_time_s": 0.003,
            "total_build_time_s": 0.004,
        }
        return SimpleNamespace(status="completed", reason=None, output=output, output_labels=graph.network.output_labels, summary=summary, task_metrics=())

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
                {},
            )

    monkeypatch.setattr("quantum_bench.providers.exact_tn.upmem_sdk_simulator.execute_upmem_taskgraph_runtime", fake_upmem_runtime)
    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeQuestRoute(),
            "quimb_tn_exact": route_registry(ROOT)["quimb_tn_exact"],
            "upmem_tn_sdk_simulator_quantized": UpmemTnSdkSimulatorQuantizedRoute(),
        },
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])
    route_ids = {record["route_id"] for record in records}
    upmem_record = next(record for record in records if record["route_id"] == "upmem_tn_sdk_simulator_quantized")

    assert "upmem_tn_sdk_simulator_exact" not in route_ids
    assert {"quest_cpu_full_state_exact", "quimb_tn_exact", "upmem_tn_sdk_simulator_quantized"} <= route_ids
    assert upmem_record["contraction_execution_target"] == "upmem"
    assert upmem_record["upmem_execution_mode"] == "sdk_simulator"
    assert upmem_record["execution_backend"] == "upmem_sdk"
    assert upmem_record["hardware_execution"] is False
    assert upmem_record["hardware_timing_available"] is False
    assert upmem_record["hardware_speedup_applicable"] is False
    assert upmem_record["cpu_fallback_used"] is False
    assert upmem_record["native_sdk_control_path"] is True
    assert upmem_record["simplepim_api_used"] is False
    assert upmem_record["quantization_mode"] == "per_task_input_quantize"
    assert upmem_record["dpu_program_invocations"] == upmem_record["task_count"]
    assert upmem_record["upmem_program_executed"] is True
    assert upmem_record["parallelism_mode"] == "sequential"
    assert upmem_record["parallelism_evidence_type"] == "executed"
    assert upmem_record["execution_plan_kind"] == "sequential_upmem_taskgraph"
    assert upmem_record["execution_plan_executed"] is True
    assert upmem_record["slicing_enabled"] is False
    assert upmem_record["frontier_scheduler_enabled"] is False
    assert upmem_record["intra_contraction_parallelism_source"] == "none"
    assert upmem_record["modeled_parallelism_available"] is False


def test_blocked_upmem_sdk_preflight_emits_no_fake_benchmark_row(monkeypatch, tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        """
schema_version: 2
suite_id: unit_upmem_sdk_blocked_compare
defaults:
  warmups: 0
  repeats: 1
  planner:
    engine: opt_einsum
    optimize: greedy
workloads:
  - id: quest_qrng_2q
    circuit: {kind: quest_compatible, name: QRNG, n_qubits: 2}
routes:
  - id: quest_cpu_full_state_exact
    role: comparison_anchor
    required: true
  - id: upmem_tn_sdk_simulator_quantized
    role: optional_upmem_candidate
    required: false
    options:
      execute_external: true
validation:
  reference_route: quest_cpu_full_state_exact
  tolerances:
    max_abs_error: 0.25
    l2_error: 2.0
    max_rel_error: 10.0
    norm_drift: 2.0
    min_fidelity: 0.0
""",
        encoding="utf-8",
    )

    def blocked_runtime(*args, **kwargs):
        summary = {
            "status": "unsupported",
            "reason": "upmem_sdk_simulator_unavailable",
            "total_tasks": 0,
            "dpu_program_executed_task_count": 0,
            "dpu_program_executed_all_tasks": False,
            "native_sdk_control_path": True,
            "simplepim_api_used": False,
        }
        return SimpleNamespace(status="unsupported", reason="upmem_sdk_simulator_unavailable", output=None, output_labels=None, summary=summary, task_metrics=())

    class FakeQuestRoute:
        name = "quest_cpu_full_state_exact"
        backend_family = "quest"
        identity = RouteIdentity(name, "fake quest", "baseline", "full_state_vector", "full_state_vector", "cpu", "test", "statevector", "compare_statevector")

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
                {},
            )

    monkeypatch.setattr("quantum_bench.providers.exact_tn.upmem_sdk_simulator.execute_upmem_taskgraph_runtime", blocked_runtime)
    monkeypatch.setattr(
        "quantum_bench.bench.simulation_backend_compare.route_registry",
        lambda root_dir: {
            "quest_cpu_full_state_exact": FakeQuestRoute(),
            "upmem_tn_sdk_simulator_quantized": UpmemTnSdkSimulatorQuantizedRoute(),
        },
    )

    result = run_simulation_backend_compare(tmp_path, suite_path=suite_path, artifact_retention="compact")
    records = load_result_records([result.run_dir])
    summary = json.loads((result.run_dir / "simulation_backend_compare_summary.json").read_text(encoding="utf-8"))

    assert {record["route_id"] for record in records} == {"quest_cpu_full_state_exact"}
    assert summary["optional_backend_reports"][0]["route_id"] == "upmem_tn_sdk_simulator_quantized"
    assert summary["optional_backend_reports"][0]["reason"] == "upmem_sdk_simulator_unavailable"
