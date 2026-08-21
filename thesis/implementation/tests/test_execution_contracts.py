from dataclasses import FrozenInstanceError, replace

import pytest
import numpy as np

from quantum_bench.execution.contracts import (
    BackendFacts,
    CpuPlan,
    ExecutionPlan,
    ExecutionResult,
    NumericMode,
    Target,
    TimingBreakdown,
    UpmemPlan,
    UpmemRuntimeResources,
    UpmemTopology,
    execution_plan_hash,
    validate_execution_plan,
    validate_execution_result,
    validate_timing,
    validate_transfer_bytes,
    validate_upmem_runtime_resources,
)


def cpu_plan() -> ExecutionPlan:
    return ExecutionPlan(
        contraction_dag_hash="dag-1",
        target=Target.CPU,
        payload=CpuPlan(
            numeric_mode=NumericMode.FLOAT32,
            executor_id="cpu_numpy_v1",
            node_order=("node-1",),
        ),
    )


def upmem_plan() -> ExecutionPlan:
    return ExecutionPlan(
        contraction_dag_hash="dag-1",
        target=Target.UPMEM,
        payload=UpmemPlan(
            topology=UpmemTopology(
                dpu_count=1,
                tasklets_per_dpu=1,
                rank_count=1,
            ),
            numeric_mode=NumericMode.FLOAT32,
            kernel_id="kernel-v1",
            decomposition_id="output_tile_v1",
            placement_id="contiguous_dpu_v1",
            reduction_id="host_reduction_v1",
        ),
    )


def test_canonical_serialization_and_hash_are_deterministic():
    first = cpu_plan()
    second = ExecutionPlan(
        payload=CpuPlan(
            node_order=("node-1",),
            executor_id="cpu_numpy_v1",
            numeric_mode=NumericMode.FLOAT32,
        ),
        target=Target.CPU,
        contraction_dag_hash="dag-1",
    )

    assert execution_plan_hash(first) == execution_plan_hash(second)
    assert first == second


def test_target_and_payload_mismatch_is_rejected():
    malformed = ExecutionPlan(
        contraction_dag_hash="dag-1",
        target=Target.UPMEM,
        payload=CpuPlan(numeric_mode=NumericMode.FLOAT32, executor_id="cpu"),
    )

    with pytest.raises(ValueError, match="UPMEM execution plans require"):
        validate_execution_plan(malformed)


def test_plan_hash_changes_for_numeric_topology_and_kernel_changes():
    baseline = upmem_plan()
    numeric = replace(baseline.payload, numeric_mode=NumericMode.HOST_PACKED_INT8)
    topology = replace(
        baseline.payload,
        topology=replace(baseline.payload.topology, dpu_count=2, rank_count=1),
    )
    kernel = replace(baseline.payload, kernel_id="kernel-v2")

    assert len({
        execution_plan_hash(baseline),
        execution_plan_hash(replace(baseline, payload=numeric)),
        execution_plan_hash(replace(baseline, payload=topology)),
        execution_plan_hash(replace(baseline, payload=kernel)),
    }) == 4


def test_plan_hash_is_independent_of_runtime_rank_paths():
    baseline = upmem_plan()
    first = UpmemRuntimeResources(
        session_root="/run/a",
        host_binary="/bin/host",
        dpu_binary="/bin/dpu",
        initialization_binary="/bin/init",
        rank_paths=("/dev/dpu_rank0",),
    )
    second = replace(first, rank_paths=("/dev/dpu_rank99",))
    assert not hasattr(baseline.payload.topology, "rank_paths")
    assert execution_plan_hash(baseline) == execution_plan_hash(
        replace(baseline, payload=replace(baseline.payload))
    )
    assert first != second


def test_rank_paths_are_runtime_bindings_and_session_opener_is_not_identity():
    def opener(plan, context):
        return None

    resources = UpmemRuntimeResources(
        session_root="/run",
        host_binary="/bin/host",
        dpu_binary="/bin/dpu",
        initialization_binary="/bin/init",
        rank_paths=("/dev/dpu_rank0",),
        session_opener=opener,
    )
    validate_upmem_runtime_resources(resources, upmem_plan().payload.topology)
    assert "session_opener" not in repr(resources)
    assert resources == replace(resources, session_opener=lambda plan, context: None)
    with pytest.raises(ValueError, match="rank_count"):
        validate_upmem_runtime_resources(
            replace(resources, rank_paths=("/dev/dpu_rank0", "/dev/dpu_rank1")),
            upmem_plan().payload.topology,
        )


def test_timing_keeps_reference_separate_from_reduction():
    timing = TimingBreakdown(reduction_s=2.0, reference_s=7.0)

    assert timing.reduction_s == 2.0
    assert timing.reference_s == 7.0
    with pytest.raises(FrozenInstanceError):
        timing.reference_s = 0.0


def test_transfer_byte_invariant_is_enforced():
    validate_transfer_bytes(8, 4, 12)
    result = ExecutionResult(
        contraction_dag_hash="dag-1",
        target=Target.UPMEM,
        output=np.zeros((1,), dtype=np.float32),
        timing=TimingBreakdown(),
        h2d_bytes=8,
        d2h_bytes=4,
        transfer_bytes=12,
        backend_facts=BackendFacts(
            backend_id="test",
            profile_id="test",
            abi_id="test",
            session_id="test",
            dispatch_id="test",
            kernel_id="test",
            execution_class="test",
            intermediate_placement="test",
            intermediate_placement_origin="test",
            native_identity_verified=True,
        ),
    )
    validate_execution_result(result)

    with pytest.raises(ValueError, match="backend_facts"):
        validate_execution_result(replace(result, backend_facts=None))

    with pytest.raises(ValueError, match="transfer_bytes"):
        validate_transfer_bytes(8, 4, 11)

    with pytest.raises(ValueError, match="non-negative"):
        validate_transfer_bytes(-1, None, None)


def test_invalid_upmem_topology_and_negative_timing_are_rejected():
    malformed = replace(
        upmem_plan(),
        payload=replace(
            upmem_plan().payload,
            topology=UpmemTopology(dpu_count=0, tasklets_per_dpu=0, rank_count=1),
        ),
    )
    with pytest.raises(ValueError, match="positive"):
        validate_execution_plan(malformed)

    with pytest.raises(ValueError, match="kernel_s"):
        validate_timing(TimingBreakdown(kernel_s=-0.1))
