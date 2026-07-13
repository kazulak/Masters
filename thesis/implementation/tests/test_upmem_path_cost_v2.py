from __future__ import annotations

from dataclasses import replace

import numpy as np

from quantum_bench.circuits import builtin_circuit
from quantum_bench.core.records import ContractionTask
from quantum_bench.routing.generic_prepare import GenericTaskPreparationCaps
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config
from quantum_bench.tn.planner_motifs import build_planner_motif_workload
from quantum_bench.tn.upmem_path_cost_v2 import (
    UPMEM_PATH_OBJECTIVE_V2,
    UpmemPathCostPolicyV2,
    model_upmem_task_cost_v2,
    task_numeric_execution,
    upmem_path_cost_policy_v2,
)


def _task() -> ContractionTask:
    return ContractionTask(
        id="task",
        input_tensor_ids=("left", "right"),
        output_tensor_id="out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=((2, 3), (3, 4)),
        output_shape=(2, 4),
        left_labels=(0, 1),
        right_labels=(1, 2),
        contracted_labels=(1,),
        output_labels=(0, 2),
        gemm_m=2,
        gemm_k=3,
        gemm_n=4,
        structure="dense",
        estimated_flops=48,
        estimated_bytes=104,
    )


def _v2_config(profile: str = "balanced_literature_informed") -> dict[str, str]:
    return {
        "engine": "custom_upmem",
        "algorithm": "greedy",
        "objective_version": UPMEM_PATH_OBJECTIVE_V2,
        "selection_scope": "projected_prefix",
        "weight_profile": profile,
        "normalization": "fixed_log1p_generic_budgets_v2",
        "execution_policy": "generic_single_dpu_split_complex_v2",
    }


def _motif_network() -> object:
    return build_planner_motif_workload(
        {
            "case_id": "planner_motif_grid",
            "circuit": {"kind": "planner_motif", "name": "grid"},
            "metadata": {
                "workload_type": "synthetic_planner_motif",
                "execution_scope": "model_only",
                "not_real_quantum_circuit": True,
            },
        }
    ).network


def test_v2_task_cost_distinguishes_component_invocations_and_memory_scopes() -> None:
    real = model_upmem_task_cost_v2(_task())
    split = model_upmem_task_cost_v2(_task(), numeric_execution=task_numeric_execution(True, False, 8))

    assert real.feasibility is True
    assert real.numeric_component_invocations == 1
    assert split.numeric_component_invocations == 4
    assert split.host_to_dpu_payload_bytes == 4 * real.host_to_dpu_payload_bytes
    assert split.dpu_to_host_payload_bytes == 4 * real.dpu_to_host_payload_bytes
    assert split.host_completion_events == 4
    assert split.task_mram_payload_bytes == real.task_mram_payload_bytes
    assert 0.0 < real.mram_static_reservation_pressure_ratio < 1.0
    assert 0.0 < real.mram_max_region_payload_ratio <= 1.0
    assert real.to_json_dict()["memory_budget_scope"] == "configured_modeled_budget_not_measured_runtime_occupancy"


def test_v2_rejects_live_payload_that_exceeds_fixed_native_buffer_reservation() -> None:
    policy = UpmemPathCostPolicyV2(
        caps=GenericTaskPreparationCaps(max_tensor_elements=16, max_contracted_combinations=16),
        native_max_tensor_elements=2,
        mram_capacity_bytes=4096,
    )

    components = model_upmem_task_cost_v2(_task(), policy)

    assert components.feasibility is False
    assert components.rejection_reasons == ("mram_live_payload_exceeds_native_static_reservation",)


def test_v2_projected_prefix_planner_is_deterministic_and_traceable() -> None:
    network = _motif_network()
    first = plan_task_graph_with_config(network, _v2_config())
    second = plan_task_graph_with_config(network, _v2_config())

    assert first.path == second.path
    assert first.path_summary.objective == UPMEM_PATH_OBJECTIVE_V2
    assert first.path_summary.planner_kind == "native_target_projected_prefix_greedy"
    assert first.path_summary.planner_metadata["selection_scope"] == "projected_prefix"
    assert first.path_summary.planner_metadata["selection_claim"] == "greedy_projected_prefix_not_global_path_optimum"
    assert first.path_summary.options["planner_config_hash"] == second.path_summary.options["planner_config_hash"]

    trace = first.path_summary.planner_metadata["step_trace"]
    assert len({entry["step_index"] for entry in trace}) == len(first.tasks)
    for step_index in range(len(first.tasks)):
        entries = [entry for entry in trace if entry["step_index"] == step_index]
        selected = [entry for entry in entries if entry["selected"]]
        assert len(selected) == 1
        assert selected[0]["candidate_rank"] == 1
        assert selected[0]["local_step_score"] is not None
        assert selected[0]["projected_cumulative_score"] is not None
        assert selected[0]["left_tensor_id"]
        assert selected[0]["output_tensor_id"]


def test_v2_accepts_zero_imaginary_storage_and_executes_complex_networks_as_split_components() -> None:
    motif = _motif_network()
    zero_imag_network = replace(
        motif,
        tensors=[replace(tensor, array=np.asarray(tensor.array, dtype=np.complex64)) for tensor in motif.tensors],
    )
    zero_imag_graph = plan_task_graph_with_config(zero_imag_network, _v2_config())
    zero_imag_executions = zero_imag_graph.path_summary.planner_metadata["task_numeric_executions"]
    assert {entry["representation"] for entry in zero_imag_executions.values()} == {"real_float32"}

    network = build_tensor_network(builtin_circuit("quantization_stress", {"n_qubits": 2}))
    standard = plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})
    custom = plan_task_graph_with_config(network, _v2_config())
    expected, _ = execute_task_sequence_np_einsum(standard, network)
    actual, _ = execute_task_sequence_np_einsum(custom, network)

    executions = custom.path_summary.planner_metadata["task_numeric_executions"]
    assert any(entry["representation"] == "split_real_imag" for entry in executions.values())
    assert sum(entry["component_invocations"] for entry in executions.values()) > len(custom.tasks)
    assert np.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_v2_policy_preserves_current_application_caps_without_claiming_abi_expansion() -> None:
    policy = upmem_path_cost_policy_v2()
    serialized = policy.to_json_dict()

    assert serialized["caps"]["application_max_contracted_combinations"] == 4096
    assert serialized["caps"]["native_abi_max_tensor_elements"] == 65536
    assert serialized["numeric_contract"] == "real_float32_or_split_real_imag_v2"
