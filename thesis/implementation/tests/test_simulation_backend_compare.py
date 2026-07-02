from __future__ import annotations

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
    assert result.run_dir.parent == tmp_path / "runs" / "evidence" / "unit_simulation_compare" / "simulation_backend_compare"
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_kind"] == "evidence_run"
    assert manifest["route_label"] == "simulation_backend_compare"
    assert manifest["normalized_records"] == "normalized_records.jsonl"
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
    roles = {record["route_id"]: record["benchmark_role"] for record in records}
    assert roles["quest_cpu_full_state_exact"] == "serious_full_state_baseline"
    assert roles["cpu_tn_einsum_exact"] == "internal_debug_baseline"
    assert "## Backend Metadata" in summary_md
    assert "Benchmark role" in summary_md
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
    compute_large = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_compute_large.yml")
    gpu_medium = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_gpu_medium.yml")
    gpu_execution_only = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_gpu_execution_only.yml")
    upmem_sdk = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_upmem_sdk_simulator.yml")
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
    compute_routes = {route["id"]: route for route in compute_medium["_route_configs"]}
    assert compute_routes["quimb_tn_exact"]["required"] is True
    assert compute_routes["quimb_tn_exact"]["benchmark_role"] == "serious_external_tn_baseline"
    assert compute_routes["cpu_tn_einsum_exact"]["required"] is False
    assert compute_routes["cpu_tn_einsum_exact"]["role"] == "optional_diagnostic"
    assert compute_routes["cpu_tn_einsum_exact"]["benchmark_role"] == "internal_debug_baseline"
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
    suite = load_suite(ROOT / "configs" / "suites" / "simulation_backend_compare_upmem_sdk_simulator.yml")
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
