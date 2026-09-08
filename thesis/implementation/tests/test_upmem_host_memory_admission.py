"""Declared prepared-wave memory admission stays analytical and pre-native."""

import sys

import pytest

from quantum_bench.lowering import slice_contraction
from quantum_bench.model import ContractionDAG, TensorSpec, TensorView
from quantum_bench.results import UnsupportedExecution
from quantum_bench.upmem import runtime
from quantum_bench.upmem.execution_features import (
    _cheap_prepared_memory_bound,
    _declared_executor_memory_estimate,
    extract_execution_features,
)
from quantum_bench.upmem.plan import (
    UpmemResources,
    UpmemTopology,
    plan_upmem,
)
from tests.test_upmem_wave_runtime import fork_join
from tests.test_upmem_wave_work import _node


POLICIES = ("split_complex_float32_v1", "complex_int8_shared_scale_v1")
SCHEDULES = ("serial_nodes_v1", "static_dag_waves_v1")


def _plan(dag, policy, schedule, *, dpu_count=3):
    return plan_upmem(
        dag,
        numeric_policy=policy,
        schedule_policy=schedule,
        topology=UpmemTopology(dpu_count=dpu_count, tasklets_per_dpu=8, rank_count=1),
    )


def _single(node):
    tensors = tuple(
        TensorSpec(view.tensor_id, view.labels, view.shape, "dense")
        for view in (node.left, node.right)
    )
    return ContractionDAG(
        tensors=tensors,
        nodes=(node,),
        output=TensorView(
            tensor_id=node.output.id,
            labels=node.output.labels,
            shape=node.output.shape,
        ),
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("sliced", [False, True])
def test_declared_estimate_matches_analytical_cohort_facts(policy, schedule, sliced):
    dag, _ = fork_join(k=3 if sliced else 1)
    if sliced:
        dag = slice_contraction(dag, node_id="a", labels=(1,))
    facts = extract_execution_features(dag, _plan(dag, policy, schedule))
    buffers = facts["host_buffers"]
    assert buffers["declared_executor_memory_scope"] == (
        "declared_prepared_wave_executor_bytes_v1"
    )
    assert buffers["declared_executor_memory_estimate_bytes"] == (
        buffers["declared_executor_persistent_bytes"]
        + buffers["declared_executor_peak_workspace_bytes"]
    )
    assert buffers["declared_executor_memory_is_rss"] is False
    assert buffers["declared_executor_live_payload_bytes"] >= 0
    assert buffers["native_output_buffer_bytes"] == 256 * 256 * 4
    assert _cheap_prepared_memory_bound(dag, _plan(dag, policy, schedule)) <= buffers[
        "declared_executor_memory_estimate_bytes"
    ]


@pytest.mark.parametrize("policy", POLICIES)
def test_tail_geometry_is_included_without_payload_materialization(policy):
    dag = _single(_node("tail", 3, 5, 65))
    facts = extract_execution_features(
        dag, _plan(dag, policy, "static_dag_waves_v1"), fuse_complex=True
    )
    buffers = facts["host_buffers"]
    assert buffers["declared_executor_submit_workspace_bytes"] > 0
    assert buffers["declared_executor_memory_estimate_bytes"] > (
        buffers["declared_executor_persistent_bytes"]
    )


def test_budget_boundary_is_inclusive_and_reserve_is_added():
    dag, _ = fork_join()
    plan = _plan(dag, POLICIES[0], SCHEDULES[1])
    estimate = extract_execution_features(dag, plan)["host_buffers"][
        "declared_executor_memory_estimate_bytes"
    ]
    runtime._admit_prepared_snapshots(
        dag, plan, fuse_complex=False, geometry_policy="panel_only_v1",
        host_memory_budget_bytes=estimate + 7, host_memory_reserve_bytes=7,
    )
    with pytest.raises(UnsupportedExecution) as failure:
        runtime._admit_prepared_snapshots(
            dag, plan, fuse_complex=False, geometry_policy="panel_only_v1",
            host_memory_budget_bytes=estimate + 6, host_memory_reserve_bytes=7,
        )
    assert failure.value.capability == "prepared_wave_host_memory"


@pytest.mark.parametrize(
    ("budget", "reserve"),
    [
        (True, 0),
        (1.0, 0),
        (-1, 0),
        (1, True),
        (1, 1.0),
        (1, -1),
    ],
)
def test_budget_and_reserve_require_integer_ranges(budget, reserve):
    dag, _ = fork_join()
    plan = _plan(dag, POLICIES[0], SCHEDULES[1])
    with pytest.raises(ValueError, match="host_memory_(budget|reserve)_bytes"):
        runtime._admit_prepared_snapshots(
            dag,
            plan,
            fuse_complex=False,
            geometry_policy="panel_only_v1",
            host_memory_budget_bytes=budget,
            host_memory_reserve_bytes=reserve,
        )


def test_cheap_lower_bound_rejects_before_control_generation(monkeypatch):
    dag, _ = fork_join()
    plan = _plan(dag, POLICIES[0], SCHEDULES[1])

    def forbidden(*args, **kwargs):
        pytest.fail("cohort controls were generated before cheap rejection")

    monkeypatch.setattr(runtime, "build_cohort_controls", forbidden)
    with pytest.raises(UnsupportedExecution) as failure:
        runtime._admit_prepared_snapshots(
            dag,
            plan,
            fuse_complex=False,
            geometry_policy="panel_only_v1",
            host_memory_budget_bytes=1,
        )
    assert failure.value.capability == "prepared_wave_host_memory"


def test_small_fixture_hand_calculates_workspace_terms():
    dag = _single(_node("tiny", 1, 1, 1))
    plan = _plan(dag, POLICIES[0], SCHEDULES[1], dpu_count=1)
    buffers = extract_execution_features(dag, plan)["host_buffers"]
    cohort = buffers["cohorts"][0]

    # Two complex128 inputs (32), one complex64 graph output (8), two
    # float32 operand planes (16), one complex64 final copy (8).
    assert buffers["caller_input_declared_bytes"] == 32
    assert buffers["graph_output_array_bytes"] == 8
    assert buffers["encoded_operand_plane_bytes"] == 16
    assert buffers["retained_response_snapshot_bytes"] == 304
    assert buffers["retained_executor_bulk_bytes"] == 336
    assert buffers["declared_executor_persistent_bytes"] == 368

    # Four controls: 136-byte header + 112-byte operation + four 160-byte
    # tiles + 64-byte payload = 952-byte envelope; E-P = 888.
    assert cohort["control_count"] == 4
    assert cohort["wave_count"] == 4
    assert cohort["input_envelope_bytes"] == 952
    assert cohort["response_snapshot_bytes"] == 304
    assert cohort["live_wave_payload_bytes"] == 64
    assert buffers["declared_executor_preparation_workspace_bytes"] == 80
    assert buffers["declared_executor_descriptor_packing_workspace_bytes"] == 2 * (952 - 64)

    # Native output is 256*256 float32 bytes; submit uses max(descriptor,
    # native output) alongside the canonical input, payload, and envelope.
    assert buffers["native_input_snapshot_bytes"] == 952
    assert buffers["native_output_buffer_bytes"] == 256 * 256 * 4
    assert buffers["declared_executor_submit_workspace_bytes"] == 263168
    assert buffers["declared_executor_assembly_workspace_bytes"] == 56
    assert buffers["declared_executor_reduce_workspace_bytes"] == 0
    assert buffers["declared_executor_peak_workspace_bytes"] == 263168
    assert buffers["declared_executor_memory_estimate_bytes"] == 368 + 263168


def test_independent_node_transients_and_preceding_lanes_overlap():
    dag, _ = fork_join(k=1)
    buffers = extract_execution_features(
        dag,
        _plan(dag, POLICIES[0], SCHEDULES[1]),
        fuse_complex=True,
    )["host_buffers"]

    # The first static cohort contains independent 2x2 nodes a and b. Each
    # node transient is 64 materialized + 64 transform + 16 complex = 144;
    # cohort canonical is 16 + 16 = 32. The later join has a larger workspace.
    first = _declared_executor_memory_estimate(dag, 0, buffers["cohorts"][:1])
    assert first["declared_executor_preparation_workspace_bytes"] == 32 + 144
    assert buffers["declared_executor_preparation_workspace_bytes"] == 256

    # Current-node assembly is 64 + 32 + 32 + 64 = 192. The second node
    # overlaps the first node's four 2x2 float32 lane arrays: 4*4*4 = 64.
    assert buffers["declared_executor_assembly_workspace_bytes"] == 32 + 64 + 192


@pytest.mark.parametrize("fuse", [False, True])
def test_int8_assembly_accounts_for_retained_int64_casts(fuse):
    dag = _single(_node("packed", 2, 3, 257))
    plan = _plan(dag, POLICIES[1], SCHEDULES[1], dpu_count=1)
    buffers = extract_execution_features(
        dag, plan, fuse_complex=fuse,
    )["host_buffers"]
    cohort = buffers["cohorts"][0]

    # The physical tile limit is 256 (not the inner WRAM panel size of 64).
    assert [unit.k_size for stage in plan.stages for unit in stage.work_units] == [256, 1]
    # Two K tiles are retained per lane. Generic controls emit one product
    # per lane; fused controls emit four products in lane zero. Both paths
    # therefore have 2*2*3*4 product elements. The source-derived one-lane
    # int64 cast term is that total*2. Casts die at each assemble call return;
    # the four returned lane arrays are counted separately.
    aggregate = cohort["int8_assembly_cast_bytes_by_node"]["packed"]
    assert cohort["wave_count"] == (8 if not fuse else 2)
    assert aggregate == (2 * 2 * 3 * 4) * 2
    assert buffers["declared_executor_int8_assembly_cast_workspace_bytes"] == aggregate


def test_rejection_precedes_executor_and_session_creation(tmp_path, monkeypatch):
    dag, _ = fork_join()
    plan = _plan(dag, POLICIES[0], SCHEDULES[1])
    estimate = extract_execution_features(dag, plan)["host_buffers"][
        "declared_executor_memory_estimate_bytes"
    ]
    root = tmp_path / "must-not-exist"

    def forbidden(*args, **kwargs):
        pytest.fail("executor was created before declared memory rejection")

    monkeypatch.setattr(runtime, "UpmemV4Executor", forbidden)
    resources = UpmemResources(
        session_root=str(root),
        host_binary=sys.executable,
        dpu_binary=sys.executable,
        initialization_binary=sys.executable,
        request_transport="packed_wave_v1",
    )
    with pytest.raises(UnsupportedExecution) as failure:
        runtime.open_upmem_simulator(
            dag, plan, resources, host_memory_budget_bytes=estimate - 1
        )
    assert failure.value.capability == "prepared_wave_host_memory"
    assert not root.exists()
