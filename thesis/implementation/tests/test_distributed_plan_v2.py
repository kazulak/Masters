from __future__ import annotations

from dataclasses import fields, replace

import pytest

from quantum_bench.targets.upmem.distributed_plan_v2 import (
    COMMUNICATION_HOST_MEDIATED_SUM_V1,
    COMMUNICATION_NONE,
    COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1,
    CONTRACTED_PARTIAL_SUM,
    DTYPE_FLOAT32,
    DTYPE_INT32,
    DISTRIBUTED_PLAN_V2_SCHEMA_VERSION,
    NativeM5Capability,
    OUTPUT_TILE,
    OUTPUT_OWNERSHIP_EXCLUSIVE,
    OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM,
    SYNCHRONIZATION_HOST_BARRIER_V1,
    SYNCHRONIZATION_NONE,
    DistributedSingleContractionPlanV2,
    build_contracted_partial_sum_plan_v2,
    build_output_tile_plan_v2,
    parse_distributed_plan_v2,
    serialize_distributed_plan_v2,
    native_m5_capability,
    validate_distributed_plan_v2,
)


@pytest.mark.parametrize("dpu_count", (1, 2, 4))
def test_builders_use_dense_ids_and_complete_ranges_for_each_dpu_count(
    dpu_count: int,
) -> None:
    plan = build_output_tile_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=11,
        total_contracted_elements=12,
        contraction_plan_hash="hash",
        dpu_count=dpu_count,
    )

    assert plan.schema_version == DISTRIBUTED_PLAN_V2_SCHEMA_VERSION
    assert plan.dpu_count == dpu_count
    assert [unit.dpu_id for unit in plan.work_units] == list(range(dpu_count))
    ranges = sorted(
        (unit.output_offset, unit.output_elements) for unit in plan.work_units
    )
    assert ranges[0][0] == 0
    assert all(
        offset + elements == next_offset
        for (offset, elements), (next_offset, _) in zip(ranges, ranges[1:])
    )
    assert ranges[-1][0] + ranges[-1][1] == plan.total_output_elements
    assert (
        max(elements for _, elements in ranges)
        - min(elements for _, elements in ranges)
        <= 1
    )
    assert all(unit.partition_kind == OUTPUT_TILE for unit in plan.work_units)
    assert all(
        unit.output_ownership == OUTPUT_OWNERSHIP_EXCLUSIVE for unit in plan.work_units
    )


def test_output_tile_serializes_deterministically_and_preserves_caller_hash() -> None:
    plan = build_output_tile_plan_v2(
        logical_operation_id="op-7",
        logical_task_id="task-7",
        total_output_elements=10,
        total_contracted_elements=12,
        contraction_plan_hash="caller-owned-contraction-hash",
        dpu_count=4,
    )
    encoded = serialize_distributed_plan_v2(plan)
    assert encoded == serialize_distributed_plan_v2(parse_distributed_plan_v2(encoded))
    assert plan.contraction_plan_hash == "caller-owned-contraction-hash"


@pytest.mark.parametrize("dpu_count", (1, 2, 4))
def test_contracted_builder_reconstructs_remainder_ranges_independent_of_unit_order(
    dpu_count: int,
) -> None:
    plan = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=5,
        total_contracted_elements=13,
        contraction_plan_hash="hash",
        dpu_count=dpu_count,
    )

    reordered = unchecked_replace(plan, work_units=tuple(reversed(plan.work_units)))
    validate_without_constructor(reordered)
    ranges = sorted(
        (unit.contracted_offset, unit.contracted_elements) for unit in plan.work_units
    )
    assert ranges[0][0] == 0
    assert all(
        offset + elements == next_offset
        for (offset, elements), (next_offset, _) in zip(ranges, ranges[1:])
    )
    assert ranges[-1][0] + ranges[-1][1] == plan.total_contracted_elements
    assert (
        max(elements for _, elements in ranges)
        - min(elements for _, elements in ranges)
        <= 1
    )


def test_contracted_partial_sum_builder_exposes_host_contract() -> None:
    plan = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op-1",
        logical_task_id="task-1",
        total_output_elements=5,
        total_contracted_elements=11,
        contraction_plan_hash="contraction-hash",
        dpu_count=2,
        dtype=DTYPE_FLOAT32,
    )

    assert [
        (unit.contracted_offset, unit.contracted_elements) for unit in plan.work_units
    ] == [
        (0, 6),
        (6, 5),
    ]
    assert all(unit.output_elements == 5 for unit in plan.work_units)
    assert all(
        unit.output_ownership == OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM
        for unit in plan.work_units
    )
    assert plan.communication.provider == COMMUNICATION_HOST_MEDIATED_SUM_V1
    assert plan.communication.dtype == DTYPE_FLOAT32
    assert plan.communication.participants == (0, 1)
    assert plan.communication.predicted_bytes == 2 * 5 * 4
    assert plan.communication.synchronization == SYNCHRONIZATION_HOST_BARRIER_V1


def test_pidcomm_contract_requires_int32_and_predicts_collective_bytes() -> None:
    plan = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op-1",
        logical_task_id="task-1",
        total_output_elements=8,
        total_contracted_elements=17,
        contraction_plan_hash="contraction-hash",
        dpu_count=4,
        dtype=DTYPE_INT32,
        communication_provider=COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1,
    )

    assert plan.communication.participants == (0, 1, 2, 3)
    assert plan.communication.predicted_bytes == 8 * 4
    assert plan.communication.dtype == DTYPE_INT32
    assert plan.communication.provider == COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1


def test_execution_hash_is_stable_and_changes_for_placement_or_provider() -> None:
    base = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op-1",
        logical_task_id="task-1",
        total_output_elements=8,
        total_contracted_elements=16,
        contraction_plan_hash="unchanged-caller-hash",
        dpu_count=2,
    )
    reordered = replace(base, work_units=tuple(reversed(base.work_units)))
    pid = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op-1",
        logical_task_id="task-1",
        total_output_elements=8,
        total_contracted_elements=16,
        contraction_plan_hash="unchanged-caller-hash",
        dpu_count=2,
        dtype=DTYPE_INT32,
        communication_provider=COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1,
    )
    placed_elsewhere = unchecked_replace(
        base,
        work_units=tuple(
            replace(unit, dpu_id=unit.dpu_id + 4) for unit in base.work_units
        ),
        communication=replace(base.communication, participants=(4, 5)),
    )

    assert base.execution_plan_hash == reordered.execution_plan_hash
    assert base.to_json_bytes() == reordered.to_json_bytes()
    assert base.execution_plan_hash != pid.execution_plan_hash
    assert base.execution_plan_hash != placed_elsewhere.execution_plan_hash
    assert (
        base.contraction_plan_hash
        == pid.contraction_plan_hash
        == placed_elsewhere.contraction_plan_hash
    )


def test_rejects_gaps_overlaps_zero_ranges_duplicates_and_mixed_kinds() -> None:
    valid = build_output_tile_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=8,
        total_contracted_elements=4,
        contraction_plan_hash="hash",
        dpu_count=2,
    )

    cases = [
        unchecked_replace(
            valid,
            work_units=(
                replace(valid.work_units[0], output_elements=3),
                valid.work_units[1],
            ),
        ),
        unchecked_replace(
            valid,
            work_units=(
                replace(valid.work_units[1], output_offset=3),
                valid.work_units[0],
            ),
        ),
        unchecked_replace(
            valid,
            work_units=(
                replace(valid.work_units[0], output_elements=0),
                valid.work_units[1],
            ),
        ),
        unchecked_replace(
            valid,
            work_units=(valid.work_units[0], replace(valid.work_units[1], dpu_id=0)),
        ),
        unchecked_replace(
            valid,
            work_units=(
                valid.work_units[0],
                replace(
                    valid.work_units[1],
                    partition_kind=CONTRACTED_PARTIAL_SUM,
                    output_ownership=OUTPUT_OWNERSHIP_SHARED_PARTIAL_SUM,
                    output_offset=0,
                    output_elements=8,
                    contracted_offset=0,
                    contracted_elements=4,
                ),
            ),
        ),
    ]
    for candidate in cases:
        with pytest.raises((ValueError, TypeError)):
            validate_without_constructor(candidate)


def test_rejects_wrong_provider_dtype_and_synchronization_contracts() -> None:
    host = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=8,
        total_contracted_elements=8,
        contraction_plan_hash="hash",
        dpu_count=2,
    )
    wrong_dtype = replace(
        host.communication, provider=COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1
    )
    wrong_sync = replace(host.communication, synchronization=SYNCHRONIZATION_NONE)
    output_with_reduction = unchecked_replace(
        build_output_tile_plan_v2(
            logical_operation_id="op",
            logical_task_id="task",
            total_output_elements=8,
            total_contracted_elements=8,
            contraction_plan_hash="hash",
            dpu_count=2,
        ),
        communication=replace(
            host.communication,
            participants=(0, 1),
            predicted_bytes=64,
        ),
    )

    for communication in (wrong_dtype, wrong_sync):
        with pytest.raises(ValueError):
            validate_without_constructor(
                unchecked_replace(host, communication=communication)
            )
    with pytest.raises(ValueError):
        validate_without_constructor(output_with_reduction)


def test_rejects_multi_dpu_partial_sum_without_provider() -> None:
    with pytest.raises(ValueError, match="explicit reducer"):
        build_contracted_partial_sum_plan_v2(
            logical_operation_id="op",
            logical_task_id="task",
            total_output_elements=4,
            total_contracted_elements=9,
            contraction_plan_hash="hash",
            dpu_count=2,
            communication_provider=COMMUNICATION_NONE,
        )


def test_rejects_sparse_ids_and_dpu_count_mismatch() -> None:
    valid = build_output_tile_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=8,
        total_contracted_elements=4,
        contraction_plan_hash="hash",
        dpu_count=2,
    )
    sparse = unchecked_replace(
        valid,
        work_units=tuple(
            replace(unit, dpu_id=unit.dpu_id + 1) for unit in valid.work_units
        ),
    )
    wrong_count = unchecked_replace(valid, dpu_count=4)

    for candidate in (sparse, wrong_count):
        with pytest.raises(ValueError):
            validate_without_constructor(candidate)


def test_rejects_identity_mismatch_between_work_units() -> None:
    valid = build_output_tile_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=8,
        total_contracted_elements=4,
        contraction_plan_hash="hash",
        dpu_count=2,
    )
    candidate = unchecked_replace(
        valid,
        work_units=(
            valid.work_units[0],
            replace(valid.work_units[1], logical_task_id="other-task"),
        ),
    )

    with pytest.raises(ValueError, match="one logical operation and task"):
        validate_without_constructor(candidate)


def test_numeric_contract_rejects_wrong_dtypes() -> None:
    with pytest.raises(ValueError, match="real float32"):
        build_output_tile_plan_v2(
            logical_operation_id="op",
            logical_task_id="task",
            total_output_elements=4,
            total_contracted_elements=4,
            contraction_plan_hash="hash",
            dtype=DTYPE_INT32,
        )
    with pytest.raises(ValueError, match="float32"):
        build_contracted_partial_sum_plan_v2(
            logical_operation_id="op",
            logical_task_id="task",
            total_output_elements=4,
            total_contracted_elements=8,
            contraction_plan_hash="hash",
            dpu_count=2,
            dtype=DTYPE_INT32,
        )
    with pytest.raises(ValueError, match="int32"):
        build_contracted_partial_sum_plan_v2(
            logical_operation_id="op",
            logical_task_id="task",
            total_output_elements=4,
            total_contracted_elements=8,
            contraction_plan_hash="hash",
            dpu_count=2,
            dtype=DTYPE_FLOAT32,
            communication_provider=COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1,
        )


def test_predicted_bytes_must_be_an_exact_non_negative_integer() -> None:
    plan = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=4,
        total_contracted_elements=8,
        contraction_plan_hash="hash",
        dpu_count=2,
    )
    for value in ("32", 32.0, True, -1):
        candidate = unchecked_replace(
            plan,
            communication=replace(plan.communication, predicted_bytes=value),
        )
        with pytest.raises(ValueError):
            validate_without_constructor(candidate)


def test_parser_rejects_malformed_bytes_types_and_v1_schema() -> None:
    plan = build_output_tile_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=4,
        total_contracted_elements=4,
        contraction_plan_hash="hash",
    )
    assert parse_distributed_plan_v2(plan.to_json_dict()).dpu_count == 1
    for malformed in (b"\xff", b"[]", None, 7):
        with pytest.raises(ValueError):
            parse_distributed_plan_v2(malformed)  # type: ignore[arg-type]

    v1_payload = plan.to_json_dict()
    v1_payload["schema_version"] = "distributed_single_contraction_plan_v1"
    with pytest.raises(ValueError, match="schema"):
        parse_distributed_plan_v2(v1_payload)


@pytest.mark.parametrize(
    ("provider", "partition_kind", "dtype"),
    (
        (COMMUNICATION_NONE, OUTPUT_TILE, DTYPE_FLOAT32),
        (COMMUNICATION_HOST_MEDIATED_SUM_V1, CONTRACTED_PARTIAL_SUM, DTYPE_FLOAT32),
        (COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1, CONTRACTED_PARTIAL_SUM, DTYPE_INT32),
    ),
)
def test_native_m5_capability_marks_output_and_host_reduction_executable(
    provider: str, partition_kind: str, dtype: str
) -> None:
    capability = native_m5_capability(provider, partition_kind, dtype)

    assert isinstance(capability, NativeM5Capability)
    assert capability.supported is True
    assert capability.executable is (
        (provider == COMMUNICATION_NONE and partition_kind == OUTPUT_TILE and dtype == DTYPE_FLOAT32)
        or (provider == COMMUNICATION_HOST_MEDIATED_SUM_V1 and partition_kind == CONTRACTED_PARTIAL_SUM and dtype == DTYPE_FLOAT32)
    )


@pytest.mark.parametrize(
    ("provider", "partition_kind", "dtype"),
    (
        (COMMUNICATION_NONE, OUTPUT_TILE, DTYPE_INT32),
        (COMMUNICATION_HOST_MEDIATED_SUM_V1, CONTRACTED_PARTIAL_SUM, DTYPE_INT32),
        (
            COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1,
            CONTRACTED_PARTIAL_SUM,
            DTYPE_FLOAT32,
        ),
        (COMMUNICATION_PIDCOMM_ALLREDUCE_INT32_V1, OUTPUT_TILE, DTYPE_INT32),
    ),
)
def test_native_m5_capability_rejects_unsupported_contracts(
    provider: str, partition_kind: str, dtype: str
) -> None:
    capability = native_m5_capability(provider, partition_kind, dtype)

    assert capability.supported is False
    assert capability.executable is False


def test_one_dpu_partial_sum_preserves_explicit_host_reduction() -> None:
    plan = build_contracted_partial_sum_plan_v2(
        logical_operation_id="op",
        logical_task_id="task",
        total_output_elements=3,
        total_contracted_elements=5,
        contraction_plan_hash="hash",
        dpu_count=1,
    )
    assert plan.work_units[0].partition_kind == CONTRACTED_PARTIAL_SUM
    assert plan.communication.provider == COMMUNICATION_HOST_MEDIATED_SUM_V1
    assert plan.communication.participants == (0,)
    assert plan.communication.predicted_bytes == 3 * 4
    assert plan.communication.synchronization == SYNCHRONIZATION_HOST_BARRIER_V1


def validate_without_constructor(plan: DistributedSingleContractionPlanV2) -> None:
    """Run validation on an invalid dataclass made with object.__new__."""

    validate_distributed_plan_v2(plan)


def unchecked_replace(
    plan: DistributedSingleContractionPlanV2, **changes: object
) -> DistributedSingleContractionPlanV2:
    candidate = object.__new__(type(plan))
    for item in fields(plan):
        object.__setattr__(
            candidate, item.name, changes.get(item.name, getattr(plan, item.name))
        )
    return candidate
