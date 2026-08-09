from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantum_bench.bench.upmem_multi_dpu_assignment import run_upmem_multi_dpu_assignment
from quantum_bench.circuits import builtin_circuit
from quantum_bench.targets.upmem.schedule import annotate_task_graph_with_upmem_estimates, estimate_dense_task
from quantum_bench.targets.upmem.tile_plan import (
    REQUIRES_TILING_NOT_IMPLEMENTED,
    UNSUPPORTED_DENSE_GEMM_SHAPE,
    UPMEM_DENSE_ESTIMATE_KEY,
    UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM,
    UPMEM_L2_EFFECTIVE_WRAM_BYTES,
    UPMEM_L2_KERNEL_STRATEGY,
    UPMEM_L2_MAX_HOST_BLOB_BYTES,
    UPMEM_L2_NATIVE_MAX_DIM,
    UPMEM_PROFILE,
    plan_dense_task,
    plan_l2_tiled_execution,
)
from quantum_bench.tn import build_tensor_network, execute_task_sequence_np_einsum, plan_task_graph_with_config
from quantum_bench.tn.planner_motifs import build_planner_motif_workload
from quantum_bench.tn.upmem_path_cost import (
    FIXED_LOG1P_GENERIC_CAPS_V1,
    PathCostComponents,
    fixed_log1p_generic_caps_v1,
    model_upmem_path_cost,
    model_upmem_task_cost,
    upmem_path_cost_policy,
    upmem_path_cost_profile,
)
from quantum_bench.tn.upmem_path_cost_v2 import (
    UPMEM_PATH_OBJECTIVE_V2,
    UpmemPathCostPolicyV2,
    model_upmem_task_cost_v2,
    task_numeric_execution,
    upmem_path_cost_policy_v2,
)
from quantum_bench.targets.upmem.hardware_taskgraph_resident import (
    ResidentCapacityError,
    RESIDENT_INVALID_SLOT,
    build_resident_graph_package,
    build_resident_slot_lifetime_map,
    resident_requantize,
    resident_round_nearest_even,
    resident_tile_ranges,
    validate_resident_graph_package_bytes,
    validate_resident_graph_package_file,
)

from .support import dense_task


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


def _motif_network(name: str = "grid"):
    return build_planner_motif_workload(
        {
            "case_id": f"planner_motif_{name}",
            "circuit": {"kind": "planner_motif", "name": name},
            "metadata": {
                "workload_type": "synthetic_planner_motif",
                "execution_scope": "model_only",
                "not_real_quantum_circuit": True,
            },
        }
    ).network


def _assignment_suite(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "schema_version: 2",
                "suite_id: assignment_fixture",
                "defaults:",
                "  planner: {engine: opt_einsum, optimize: greedy}",
                "workloads:",
                "  - id: bell_2q",
                "    circuit: {kind: builtin, name: bell_2q}",
                "routes:",
                "  - id: upmem_tn_sdk_simulator_quantized",
                "    required: false",
                "validation: {}",
            )
        ),
        encoding="utf-8",
    )
    return path


def test_v2_task_cost_expands_split_complex_components() -> None:
    task = dense_task("task", 2, 3, 4)
    real = model_upmem_task_cost_v2(task)
    split = model_upmem_task_cost_v2(task, numeric_execution=task_numeric_execution(True, False, 8))

    assert real.feasibility is True
    assert real.numeric_component_invocations == 1
    assert split.numeric_component_invocations == 4
    assert split.host_to_dpu_payload_bytes == 4 * real.host_to_dpu_payload_bytes
    assert split.dpu_to_host_payload_bytes == 4 * real.dpu_to_host_payload_bytes
    assert split.task_mram_payload_bytes == real.task_mram_payload_bytes
    assert real.to_json_dict()["memory_budget_scope"] == "configured_modeled_budget_not_measured_runtime_occupancy"


def test_v2_rejects_native_static_reservation_overflow() -> None:
    policy = UpmemPathCostPolicyV2(native_max_tensor_elements=2, mram_capacity_bytes=4096)

    components = model_upmem_task_cost_v2(dense_task("task", 2, 3, 4), policy)

    assert components.feasibility is False
    assert components.rejection_reasons == ("mram_live_payload_exceeds_native_static_reservation",)


def test_v2_projected_prefix_selection_is_deterministic_and_traceable() -> None:
    network = _motif_network()
    first = plan_task_graph_with_config(network, _v2_config())
    second = plan_task_graph_with_config(network, _v2_config())

    assert first.path == second.path
    assert first.path_summary.objective == UPMEM_PATH_OBJECTIVE_V2
    assert first.path_summary.planner_kind == "native_target_projected_prefix_greedy"
    assert first.path_summary.planner_metadata["selection_scope"] == "projected_prefix"
    assert "not_global_path_optimum" in first.path_summary.planner_metadata["selection_claim"]
    trace = first.path_summary.planner_metadata["step_trace"]
    assert len({entry["step_index"] for entry in trace}) == len(first.tasks)
    for step_index in range(len(first.tasks)):
        selected = [entry for entry in trace if entry["step_index"] == step_index and entry["selected"]]
        candidates = [
            entry
            for entry in trace
            if entry["step_index"] == step_index and entry["candidate_rank"] is not None
        ]
        assert len(selected) == 1
        assert candidates
        assert all(entry["feasible"] is True for entry in candidates)
        assert selected[0]["feasible"] is True
        assert selected[0]["candidate_rank"] == 1
        assert selected[0]["projected_cumulative_score"] is not None


def test_v2_split_complex_execution_matches_exact_task_sequence() -> None:
    network = build_tensor_network(builtin_circuit("quantization_stress", {"n_qubits": 2}))
    standard = plan_task_graph_with_config(network, {"engine": "opt_einsum", "optimize": "greedy"})
    custom = plan_task_graph_with_config(network, _v2_config())
    expected, _ = execute_task_sequence_np_einsum(standard, network)
    actual, _ = execute_task_sequence_np_einsum(custom, network)
    executions = custom.path_summary.planner_metadata["task_numeric_executions"]

    assert any(entry["representation"] == "split_real_imag" for entry in executions.values())
    assert sum(entry["component_invocations"] for entry in executions.values()) > len(custom.tasks)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_v2_policy_keeps_application_caps_and_numeric_contract() -> None:
    serialized = upmem_path_cost_policy_v2().to_json_dict()

    assert serialized["caps"]["application_max_contracted_combinations"] == 4096
    assert serialized["caps"]["native_abi_max_tensor_elements"] == 65536
    assert serialized["numeric_contract"] == "real_float32_or_split_real_imag_v2"


def test_v1_path_cost_has_exact_transfer_components() -> None:
    task = dense_task("task", 2, 3, 4)
    components = model_upmem_task_cost(task)
    path = model_upmem_path_cost((task, dense_task("second", 2, 3, 4)))

    assert components == PathCostComponents(
        flops=48,
        peak_bytes=48,
        intermediate_writes=32,
        host_to_dpu_bytes=72,
        dpu_to_host_bytes=32,
        host_dpu_bytes=104,
        mram_wram_bytes=224,
        local_work=24,
        sync_events=1,
        numeric_penalty=0.0,
        wram_pressure=8 / 65536,
        tiles=1,
    )
    assert path.flops == 96
    assert path.host_dpu_bytes == 208
    assert path.feasibility is True


def test_v1_rejection_and_normalization_are_stable() -> None:
    rejected = model_upmem_task_cost(dense_task("too_large", 65537, 1, 1))
    policy = upmem_path_cost_policy()
    normalized = fixed_log1p_generic_caps_v1(policy).normalize(model_upmem_task_cost(dense_task("task", 2, 3, 4), policy))

    assert rejected.feasibility is False
    assert rejected.rejection_reasons == ("element_count_cap_exceeded",)
    assert fixed_log1p_generic_caps_v1(policy).normalization_id == FIXED_LOG1P_GENERIC_CAPS_V1
    assert normalized["flops"] == np.log1p(48) / np.log1p(2 * 65536 * 4096)


@pytest.mark.parametrize(
    "profile_id",
    ["compute_oriented", "wram_constrained", "balanced_literature_informed"],
)
def test_v1_profiles_are_serializable(profile_id: str) -> None:
    profile = upmem_path_cost_profile(profile_id)

    assert profile.policy.policy_id == "generic_single_dpu_float32_v1"
    assert json.dumps(profile.to_json_dict(), sort_keys=True)


@pytest.mark.parametrize(
    ("shape", "tiles", "requires_tiling"),
    [((8, 8, 8), 1, False), ((256, 256, 256), 16, True), ((8, 65536, 8), 128, True)],
)
def test_dense_tile_plan_preserves_feasibility_and_tile_counts(shape, tiles: int, requires_tiling: bool) -> None:
    plan = plan_dense_task(dense_task("tile", *shape))

    assert plan.supported is True
    assert plan.tile_counts.total_tile_count == tiles
    assert plan.requires_tiling is requires_tiling
    assert plan.working_set_bytes <= UPMEM_PROFILE.wram_bytes
    if requires_tiling:
        assert plan.tiling_implemented is False
        assert plan.reject_reason == REQUIRES_TILING_NOT_IMPLEMENTED


def test_dense_tile_plan_rejects_unknown_structure() -> None:
    plan = plan_dense_task(dense_task("sparse", 8, 8, 8, structure="sparse"))

    assert plan.supported is False
    assert plan.reject_reason == UNSUPPORTED_DENSE_GEMM_SHAPE
    assert plan.tile_counts.total_tile_count == 0


@pytest.mark.parametrize("shape", [(96, 96, 96), (128, 128, 64), (72, 512, 32)])
def test_l2_tile_model_is_bounded(shape) -> None:
    plan = plan_l2_tiled_execution(*shape)

    assert plan.supported is True
    assert plan.reason is None
    assert plan.execution_class == UPMEM_EXECUTION_CLASS_L2_SINGLE_DPU_MRAM
    assert plan.kernel_strategy == UPMEM_L2_KERNEL_STRATEGY
    assert plan.conservative_full_task_bytes > UPMEM_L2_EFFECTIVE_WRAM_BYTES
    assert plan.estimated_wram_bytes_per_tile <= UPMEM_L2_EFFECTIVE_WRAM_BYTES
    assert plan.host_blob_bytes <= UPMEM_L2_MAX_HOST_BLOB_BYTES
    assert max(plan.gemm_m, plan.gemm_k, plan.gemm_n) <= UPMEM_L2_NATIVE_MAX_DIM
    assert plan.total_tile_steps == plan.output_tile_count * plan.k_tile_count


def test_l2_tile_model_rejects_l1_fit_and_blob_overflow() -> None:
    assert plan_l2_tiled_execution(16, 16, 16).reason == "not_l2_wram_resident"
    assert plan_l2_tiled_execution(96, 96, 96, max_l2_host_blob_bytes=1024).reason == "unsupported_l2_blob_size"


def test_schedule_estimate_is_attached_to_every_task(minimal_graph) -> None:
    annotated, summary = annotate_task_graph_with_upmem_estimates(minimal_graph.graph)

    metadata = summary.metadata()
    assert metadata["task_count"] == len(annotated.tasks)
    assert metadata["estimate_key"] == UPMEM_DENSE_ESTIMATE_KEY
    assert all(UPMEM_DENSE_ESTIMATE_KEY in task.target_estimates for task in annotated.tasks)
    assert all(estimate_dense_task(task).supported for task in annotated.tasks)


def test_resident_allocator_is_deterministic_and_owns_nonoverlapping_slots(minimal_graph, resident_hardware_suite) -> None:
    first = build_resident_slot_lifetime_map(minimal_graph.graph, minimal_graph.network, profile=resident_hardware_suite.profile)
    second = build_resident_slot_lifetime_map(minimal_graph.graph, minimal_graph.network, profile=resident_hardware_suite.profile)

    assert first.to_json_dict() == second.to_json_dict()
    assert first.mram_used_bytes <= first.mram_pool_bytes
    assert all(slot.offset_bytes % 8 == 0 for slot in first.slots)
    for slot in first.slots:
        for left, right in zip(slot.lifetimes, slot.lifetimes[1:]):
            assert left.end_task < right.start_task or right.end_task < left.start_task

    package = build_resident_graph_package(
        minimal_graph.graph,
        minimal_graph.network,
        case_id="fixture",
        suite_id="fixture",
        quantization_mode="none",
        profile=resident_hardware_suite.profile,
    )
    available_slots = set(package.initial_data)
    for operation in package.operations:
        inputs = {
            slot
            for slot in (
                operation.slot_a,
                operation.slot_b,
                operation.slot_c,
                operation.slot_d,
            )
            if slot != RESIDENT_INVALID_SLOT
        }
        assert inputs <= available_slots
        outputs = {
            slot
            for slot in (operation.slot_out_real, operation.slot_out_imag)
            if slot != RESIDENT_INVALID_SLOT
        }
        available_slots.update(outputs)


def test_resident_allocator_reports_capacity_without_spill(minimal_graph, resident_hardware_suite) -> None:
    profile = resident_hardware_suite.profile.__class__(
        **{**resident_hardware_suite.profile.__dict__, "max_slot_descriptors": 0}
    )

    with pytest.raises(ResidentCapacityError, match="slot_descriptor_cap_exceeded"):
        build_resident_slot_lifetime_map(minimal_graph.graph, minimal_graph.network, profile=profile)


def test_resident_allocator_can_disable_slot_reuse(minimal_graph, resident_hardware_suite) -> None:
    reused = build_resident_slot_lifetime_map(
        minimal_graph.graph,
        minimal_graph.network,
        profile=resident_hardware_suite.profile,
    )
    distinct = build_resident_slot_lifetime_map(
        minimal_graph.graph,
        minimal_graph.network,
        profile=resident_hardware_suite.profile,
        allow_slot_reuse=False,
    )

    assert distinct.slot_descriptor_count == len(distinct.lifetimes)
    assert distinct.slot_descriptor_count > reused.slot_descriptor_count
    assert len(set(distinct.logical_to_slot.values())) == len(distinct.lifetimes)

    package = build_resident_graph_package(
        minimal_graph.graph,
        minimal_graph.network,
        case_id="fixture",
        suite_id="fixture",
        quantization_mode="none",
        profile=resident_hardware_suite.profile,
        allow_slot_reuse=False,
    )
    assert package.allocation.slot_descriptor_count == len(package.allocation.lifetimes)

    limited_profile = resident_hardware_suite.profile.__class__(
        **{
            **resident_hardware_suite.profile.__dict__,
            "max_slot_descriptors": distinct.slot_descriptor_count - 1,
        }
    )
    with pytest.raises(ResidentCapacityError, match="slot_descriptor_cap_exceeded"):
        build_resident_slot_lifetime_map(
            minimal_graph.graph,
            minimal_graph.network,
            profile=limited_profile,
            allow_slot_reuse=False,
        )


def test_resident_package_round_trip_is_schema_checked(minimal_graph, resident_hardware_suite, tmp_path: Path) -> None:
    package = build_resident_graph_package(minimal_graph.graph, minimal_graph.network, case_id="case", suite_id="suite", quantization_mode="none", profile=resident_hardware_suite.profile)
    binary = tmp_path / "dpu"
    binary.write_bytes(b"fixture")
    written = package.write(tmp_path, dpu_binary=binary, request_id="request")
    metadata = validate_resident_graph_package_file(written.package_path, profile=resident_hardware_suite.profile)

    assert metadata["graph_request_count"] == 1
    assert metadata["operation_count"] == package.component_operation_count
    assert metadata["slot_count"] == package.allocation.slot_descriptor_count
    assert json.loads(written.manifest_path.read_text(encoding="utf-8"))["intermediate_output_paths"] == []


def test_resident_binary_rejects_bad_magic(minimal_graph, resident_hardware_suite) -> None:
    package = build_resident_graph_package(minimal_graph.graph, minimal_graph.network, case_id="case", suite_id="suite", quantization_mode="none", profile=resident_hardware_suite.profile)
    import quantum_bench.targets.upmem.hardware_taskgraph_resident as resident

    payload = bytearray(resident._encode_package(package.allocation.slots, package.operations))
    payload[0] ^= 1

    with pytest.raises(ValueError, match="bad_magic"):
        validate_resident_graph_package_bytes(bytes(payload))


def test_resident_split_complex_uses_component_operations(split_complex_graph_fixture, resident_hardware_suite) -> None:
    package = build_resident_graph_package(split_complex_graph_fixture.graph, split_complex_graph_fixture.network, case_id="complex", suite_id="suite", quantization_mode="none", profile=resident_hardware_suite.profile)
    reference, _ = execute_task_sequence_np_einsum(split_complex_graph_fixture.graph, split_complex_graph_fixture.network)

    assert [operation.component for operation in package.operations] == ["ar_br", "ai_bi", "ar_bi", "ai_br", "complex_combine"]
    assert all(operation.to_json_dict()["intermediate_output_path"] is None for operation in package.operations)
    assert package.allocation.final_components
    np.testing.assert_allclose(
        package.full_precision_output if package.full_precision_output is not None else reference,
        reference,
    )


def test_resident_tile_and_rounding_boundaries() -> None:
    assert resident_tile_ranges(255) == ((0, 254),)
    assert resident_tile_ranges(256) == ((0, 255),)
    assert resident_tile_ranges(257) == ((0, 255), (256, 256))
    np.testing.assert_array_equal(
        resident_round_nearest_even(np.array([0.5, 1.5, -0.5, -1.5, 2.5, -2.5], dtype=np.float32)),
        np.array([0, 2, 0, -2, 2, -2], dtype=np.float32),
    )
    quantized, scale, saturation = resident_requantize(np.zeros(4, dtype=np.float32))
    assert scale == 1.0
    assert saturation == 0
    np.testing.assert_array_equal(quantized, np.zeros(4, dtype=np.int8))
    for element_count in (0, 1, 255, 256, 257, 513):
        ranges = resident_tile_ranges(element_count)
        covered = [index for start, end in ranges for index in range(start, end + 1)]
        assert covered == list(range(element_count))
        assert len(covered) == len(set(covered))


@pytest.mark.parametrize(
    "strategy",
    ["frontier_round_robin_dpu_groups", "frontier_size_aware_dpu_groups"],
)
def test_multi_dpu_assignment_is_modeled_and_owns_each_task_once(
    tmp_path: Path, strategy: str
) -> None:
    result = run_upmem_multi_dpu_assignment(
        tmp_path,
        suite_path=_assignment_suite(tmp_path / "suite.yml"),
        dpu_group_count=2,
        strategy=strategy,
    )
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    assignments = [assignment for case in plan["cases"] for wave in case["frontier_waves"] for assignment in wave["assignments"]]
    task_ids = [assignment["task_id"] for assignment in assignments]

    assert result.status == "completed"
    assert len(task_ids) == len(set(task_ids))
    assert plan["metadata"]["modeled_only"] is True
    assert plan["metadata"]["dpu_programs_executed"] is False
    assert plan["summary"]["assigned_task_count"] == plan["summary"]["task_count"]
    assert plan["summary"]["executed_dpu_task_count"] == 0
    assert {assignment["dpu_group_id"] for assignment in assignments} <= {0, 1}
    assert all(
        case["dpu_assignment_validation_status"] == "passed"
        and case["assigned_task_count"] == case["task_count"]
        and case["duplicate_assignment_check"] == "passed"
        and case["missing_dependency_check"] == "passed"
        for case in plan["cases"]
    )


def test_multi_dpu_sequential_strategy_keeps_single_owner(tmp_path: Path) -> None:
    result = run_upmem_multi_dpu_assignment(
        tmp_path,
        suite_path=_assignment_suite(tmp_path / "suite.yml"),
        dpu_group_count=4,
        strategy="sequential_single_dpu",
    )
    plan = json.loads(result.plan_path.read_text(encoding="utf-8"))
    assignments = [assignment for case in plan["cases"] for wave in case["frontier_waves"] for assignment in wave["assignments"]]
    task_ids = [assignment["task_id"] for assignment in assignments]

    assert {assignment["dpu_group_id"] for assignment in assignments} == {0}
    assert plan["dpu_group_count"] == 4
    assert len(task_ids) == len(set(task_ids)) == plan["summary"]["task_count"]
    assert plan["summary"]["assigned_task_count"] == plan["summary"]["task_count"]
    assert all(case["dpu_assignment_validation_status"] == "passed" for case in plan["cases"])
