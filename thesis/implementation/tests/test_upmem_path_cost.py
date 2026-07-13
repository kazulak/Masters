from __future__ import annotations

import json

import numpy as np

from quantum_bench.core.records import ContractionTask, TensorSpec, TensorValue
from quantum_bench.routing.generic_prepare import (
    GenericTaskPreparationCaps,
    GenericTaskPreparationInput,
    generic_structural_feasibility,
    prepare_generic_task,
)
from quantum_bench.tn.upmem_path_cost import (
    FIXED_LOG1P_GENERIC_CAPS_V1,
    GENERIC_SINGLE_DPU_FLOAT32_V1,
    PathCostComponents,
    fixed_log1p_generic_caps_v1,
    model_upmem_path_cost,
    model_upmem_task_cost,
    upmem_path_cost_policy,
    upmem_path_cost_profile,
)


def _task(
    task_id: str = "generic",
    *,
    left_shape: tuple[int, ...] = (2, 3),
    right_shape: tuple[int, ...] = (3, 4),
    output_shape: tuple[int, ...] = (2, 4),
    left_labels: tuple[int, ...] = (0, 1),
    right_labels: tuple[int, ...] = (1, 2),
    contracted_labels: tuple[int, ...] = (1,),
    output_labels: tuple[int, ...] = (0, 2),
) -> ContractionTask:
    return ContractionTask(
        id=task_id,
        input_tensor_ids=(f"{task_id}_left", f"{task_id}_right"),
        output_tensor_id=f"{task_id}_out",
        dependencies=(),
        index_expression="ab,bc->ac",
        input_shapes=(left_shape, right_shape),
        output_shape=output_shape,
        left_labels=left_labels,
        right_labels=right_labels,
        contracted_labels=contracted_labels,
        output_labels=output_labels,
        gemm_m=0,
        gemm_k=0,
        gemm_n=0,
        structure="generic",
        estimated_flops=0,
        estimated_bytes=0,
    )


def _tensors(task: ContractionTask) -> tuple[TensorValue, TensorValue]:
    left = np.zeros(task.input_shapes[0], dtype=np.float32)
    right = np.zeros(task.input_shapes[1], dtype=np.float32)
    return (
        TensorValue(TensorSpec(task.input_tensor_ids[0], task.left_labels, task.input_shapes[0], "dense"), left),
        TensorValue(TensorSpec(task.input_tensor_ids[1], task.right_labels, task.input_shapes[1], "dense"), right),
    )


def test_public_feasibility_parity_for_supported_and_rejected_shapes() -> None:
    supported = _task()
    left, right = _tensors(supported)
    preparation = prepare_generic_task(GenericTaskPreparationInput(supported, left, right))
    feasibility = generic_structural_feasibility(supported)

    assert feasibility.feasible is True
    assert feasibility.reason is None
    assert preparation.status == "prepared"
    assert preparation.reason is None
    assert {key: preparation.metadata[key] for key in feasibility.metadata} == feasibility.metadata

    rejected = _task("rank", left_shape=(1,) * 17, right_shape=(1,), output_shape=(1,) * 16)
    rejected_left, rejected_right = _tensors(rejected)
    rejected_preparation = prepare_generic_task(GenericTaskPreparationInput(rejected, rejected_left, rejected_right))
    rejected_feasibility = generic_structural_feasibility(rejected)

    assert rejected_feasibility.feasible is False
    assert rejected_feasibility.reason == "rank_cap_exceeded"
    assert rejected_preparation.status == "unsupported_shape"
    assert rejected_preparation.reason == rejected_feasibility.reason


def test_structural_rejection_order_and_int32_reason_are_shared() -> None:
    element_task = _task("elements", left_shape=(65537, 1), right_shape=(1, 1), output_shape=(65537, 1))
    assert generic_structural_feasibility(element_task).reason == "element_count_cap_exceeded"

    contracted_task = _task(
        "contracted",
        left_shape=(4097, 1),
        right_shape=(4097, 1),
        output_shape=(1, 1),
        right_labels=(0, 2),
        contracted_labels=(0,),
        output_labels=(1, 2),
    )
    assert generic_structural_feasibility(contracted_task).reason == "contracted_combination_cap_exceeded"

    overflow_task = _task("overflow", left_shape=(1, 200000), right_shape=(200000, 1), output_shape=(1, 1))
    caps = GenericTaskPreparationCaps(max_tensor_elements=300000, max_contracted_combinations=300000)
    assert generic_structural_feasibility(overflow_task, caps).reason == "int32_accumulation_overflow_risk"
    assert generic_structural_feasibility(overflow_task, caps, check_int32_accumulation=False).feasible is True


def test_default_float32_components_account_for_task_movement() -> None:
    components = model_upmem_task_cost(_task())

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

    path = model_upmem_path_cost((_task("first"), _task("second")))
    assert path.flops == 96
    assert path.host_to_dpu_bytes == 144
    assert path.dpu_to_host_bytes == 64
    assert path.host_dpu_bytes == 208
    assert path.intermediate_writes == 64
    assert path.feasibility is True


def test_cost_components_preserve_structural_rejection_reason() -> None:
    task = _task("rejected", left_shape=(65537, 1), right_shape=(1, 1), output_shape=(65537, 1))
    components = model_upmem_task_cost(task)

    assert components.feasibility is False
    assert components.rejection_reasons == ("element_count_cap_exceeded",)
    assert components.flops == 0


def test_normalization_and_profiles_are_fixed_and_serializable() -> None:
    policy = upmem_path_cost_policy()
    normalization = fixed_log1p_generic_caps_v1(policy)
    components = model_upmem_task_cost(_task(), policy)
    first = normalization.normalize(components)
    second = fixed_log1p_generic_caps_v1(policy).normalize(components)

    assert normalization.normalization_id == FIXED_LOG1P_GENERIC_CAPS_V1
    assert first == second
    assert first["flops"] == np.log1p(48) / np.log1p(2 * 65536 * 4096)
    assert json.dumps(normalization.to_json_dict(), sort_keys=True) == json.dumps(
        fixed_log1p_generic_caps_v1(policy).to_json_dict(), sort_keys=True
    )

    profile_ids = (
        "compute_oriented",
        "host_transfer_oriented",
        "local_movement_oriented",
        "wram_constrained",
        "synchronization_constrained",
        "balanced_literature_informed",
    )
    profiles = [upmem_path_cost_profile(profile_id, policy=policy) for profile_id in profile_ids]
    assert all(profile.policy.policy_id == GENERIC_SINGLE_DPU_FLOAT32_V1 for profile in profiles)
    assert [profile.profile_id for profile in profiles] == list(profile_ids)
    assert all(json.dumps(profile.to_json_dict(), sort_keys=True) for profile in profiles)
