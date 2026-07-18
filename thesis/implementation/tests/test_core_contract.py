from __future__ import annotations

import json
from pathlib import Path

from quantum_bench.bench.config import load_suite
from quantum_bench.circuits import builtin_circuit, manifest
from quantum_bench.tn import build_tensor_network, plan_task_graph
from quantum_bench.providers import route_registry
from quantum_bench.targets.upmem.schedule import estimate_dense_task_graph
from quantum_bench.validation import compute_reference, validate


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_manifest_gate_counts() -> None:
    circuit = builtin_circuit("ghz_chain", {"n_qubits": 4})
    record = manifest(circuit)
    assert record["gate_counts"] == {"1q": 1, "2q": 3, "total": 4}
    assert record["gate_set"] == ["cx", "h"]


def test_task_graph_serializes_and_validates_reference() -> None:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    reference, _ = compute_reference(network)
    result = validate(reference, reference)
    json.dumps(graph, default=lambda obj: getattr(obj, "__dict__", str(obj)))
    assert graph.tasks
    assert result.passed
    assert result.l2_error == 0.0


def test_upmem_target_estimates_shared_task_graph() -> None:
    circuit = builtin_circuit("bell_2q")
    network = build_tensor_network(circuit)
    graph = plan_task_graph(network)
    schedule = estimate_dense_task_graph(graph)
    assert len(schedule.tasks) == len(graph.tasks)
    assert schedule.hardware.wram_bytes == 64 * 1024
    assert schedule.total_host_to_dpu_bytes > 0
    assert schedule.total_dpu_to_host_bytes > 0
    assert schedule.metadata()["target"] == "upmem"


def test_suite_config_loads() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "smoke.yml")
    assert suite["suite_id"] == "smoke"
    assert suite["_schema_version"] == 2
    assert suite["route_policy"]["routes"] == ["cpu_tn_einsum_exact"]


def test_smoke_suite_v2_schema_loads_with_separate_workloads_and_routes() -> None:
    suite = load_suite(ROOT / "configs" / "suites" / "smoke.yml")
    assert suite["_schema_version"] == 2
    assert [case["case_id"] for case in suite["cases"]] == ["bell_2q", "ghz_4q"]
    assert suite["route_policy"]["routes"] == ["cpu_tn_einsum_exact"]
    assert suite["_route_configs"][0]["required"] is True


def test_route_probe_and_upmem_skip_reason() -> None:
    routes = route_registry(ROOT)
    assert routes["cpu_tn_einsum_exact"].probe().available
    assert routes["cpu_tn_einsum_exact"].identity.role == "internal_debug_baseline"
    assert routes["cpu_tn_einsum_exact"].identity.output_contract == "final_tensor"
    assert routes["cpu_tn_einsum_exact"].identity.validation_mode == "compare_output"
    assert routes["quest_cpu_full_state_exact"].identity.output_contract == "statevector"
    assert routes["quest_cpu_full_state_exact"].identity.role == "serious_full_state_baseline"
    assert routes["quest_cpu_full_state_exact"].identity.validation_mode == "compare_statevector"
    assert routes["quest_cpu_full_state_exact"].backend_family == "quest"
    assert routes["quest_gpu_full_state_exact"].identity.output_contract == "statevector"
    assert routes["quest_gpu_full_state_exact"].identity.role == "optional_gpu_candidate"
    assert routes["quest_gpu_full_state_exact"].identity.hardware_target == "gpu"
    gpu_capabilities = routes["quest_gpu_full_state_exact"].capabilities()
    assert gpu_capabilities.can_return_output == bool(gpu_capabilities.metadata.get("gpu_backend_verified", False))
    assert gpu_capabilities.metadata["gpu_records_require_real_gpu_execution"] is True
    gpu_tn_routes = [
        route_id
        for route_id, route in routes.items()
        if route.identity.hardware_target == "gpu" and "tensor_network" in route.identity.simulation_method
    ]
    assert gpu_tn_routes == []
    assert routes["quimb_tn_exact"].identity.output_contract == "final_tensor"
    assert routes["quimb_tn_exact"].identity.role == "serious_external_tn_baseline"
    assert routes["quimb_tn_exact"].identity.validation_mode == "compare_output"
    assert routes["quimb_tn_exact"].backend_family == "quimb"
    assert routes["quimb_tn_sliced_exact"].identity.output_contract == "final_tensor"
    assert routes["quimb_tn_sliced_exact"].identity.role == "explicit_slicing_evidence"
    assert routes["quimb_tn_sliced_exact"].identity.validation_mode == "compare_output"
    assert routes["quimb_tn_sliced_exact"].backend_family == "quimb"
    assert routes["quimb_tn_sliced_exact"].capabilities().metadata["slicing_enabled"] is True
    assert "upmem_tn_sdk_simulator_quantized" in routes
    assert "upmem_tn_sdk_simulator_exact" not in routes
    assert routes["upmem_tn_sdk_simulator_quantized"].identity.hardware_target == "upmem"
    assert routes["upmem_tn_sdk_simulator_quantized"].identity.execution_mode == "sdk_simulator"
    assert routes["upmem_tn_sdk_simulator_quantized"].identity.output_contract == "final_tensor"
    assert routes["upmem_tn_sdk_simulator_quantized"].backend_family == "upmem_sdk"


def test_parallelization_docs_link_upmem_multi_dpu_design() -> None:
    roadmap = (ROOT / "docs" / "parallelization_roadmap.md").read_text(encoding="utf-8")
    strategy = (ROOT / "docs" / "parallelization_implementation_strategy.md").read_text(encoding="utf-8")
    design = (ROOT / "docs" / "upmem_multi_dpu_scheduling_design.md").read_text(encoding="utf-8")
    readiness = (ROOT / "docs" / "upmem_multi_dpu_prototype_readiness.md").read_text(encoding="utf-8")

    assert "upmem_multi_dpu_scheduling_design.md" in roadmap
    assert "upmem_multi_dpu_scheduling_design.md" in strategy
    assert "upmem_multi_dpu_prototype_readiness.md" in roadmap
    assert "upmem_multi_dpu_prototype_readiness.md" in strategy
    assert "upmem_multi_dpu_prototype_readiness.md" in design
    assert "upmem_parallelism_evidence_type" in design
    assert "upmem-multi-dpu-assignment" in strategy
    assert "upmem-multi-dpu-assignment" in design
    assert "modeled assignment report implemented" in roadmap
    assert "modeled_only" in design
    assert "sdk_simulator_executed" in design
    assert "hardware_executed" in design
    assert "cpu_fallback_used=false" in design
    assert "no hardware speedup claim from SDK simulator rows" in design
    assert "dpu_alloc(1" in readiness
    assert "frontier_worker_count=1" in readiness
    assert "hardware_speedup_applicable=false" in readiness
    assert "Status: implemented for `frontier_worker_count=1`" in readiness
    assert "not hardware multi-DPU execution" in readiness
