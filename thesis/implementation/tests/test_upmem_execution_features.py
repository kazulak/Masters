"""Planning facts must describe actual prepared-wave execution, not old profiles."""

from dataclasses import replace
import json

import numpy as np
import pytest

from quantum_bench.cpu import replay_upmem_plan_once
from quantum_bench.lowering import slice_contraction
from quantum_bench.model import ContractionDAG, TensorSpec, TensorView
from quantum_bench.upmem.execution_features import extract_execution_features
from quantum_bench.upmem.plan import UpmemResources, UpmemTopology, plan_upmem
from quantum_bench.upmem.runtime import _prepare_complex_operation, open_upmem_simulator
from quantum_bench.upmem.wave_protocol import COMPLETION, CONTROL, FOUR_PRODUCT_KERNELS, IDLE
from quantum_bench.upmem.wave_work import build_cohort_waves
from tests.test_upmem_wave_runtime import binaries as _sdk_binaries, fork_join
from tests.test_upmem_wave_work import _node


POLICIES = ("split_complex_float32_v1", "complex_int8_shared_scale_v1")
SCHEDULES = ("serial_nodes_v1", "static_dag_waves_v1")
binaries = _sdk_binaries


def _plan(dag, policy=POLICIES[0], schedule=SCHEDULES[1], dpus=3):
    return plan_upmem(dag, numeric_policy=policy, schedule_policy=schedule,
                      topology=UpmemTopology(dpu_count=dpus, tasklets_per_dpu=8, rank_count=1))


@pytest.mark.parametrize("fuse", [False, True])
def test_parallel_work_is_maximum_per_wave_not_sum_of_independent_nodes(fuse):
    dag, _ = fork_join(k=1)
    serial = extract_execution_features(dag, _plan(dag, schedule=SCHEDULES[0]), fuse_complex=fuse)
    concurrent = extract_execution_features(dag, _plan(dag), fuse_complex=fuse)
    assert serial["totals"]["real_mac_count"] == concurrent["totals"]["real_mac_count"] == 64
    assert serial["totals"]["wave_critical_real_mac_sum"] == 64
    assert concurrent["totals"]["wave_critical_real_mac_sum"] == 48
    assert serial["totals"]["launch_count"] == (3 if fuse else 12)
    assert concurrent["totals"]["launch_count"] == (2 if fuse else 8)
    assert concurrent["totals"]["idle_slot_launch_count"] == (3 if fuse else 12)
    assert concurrent["totals"]["h2d_bytes"] == (128 + 6 * 144 if fuse else 256 + 24 * 144)
    assert concurrent["totals"]["d2h_bytes"] == 192 + (6 if fuse else 24) * 72
    # Serializable, deterministic facts cannot depend on tensor data or timing.
    assert json.dumps(concurrent, sort_keys=True) == json.dumps(
        extract_execution_features(dag, _plan(dag), fuse_complex=fuse), sort_keys=True)


def test_one_dpu_schedule_degenerates_without_inventing_parallel_gain():
    dag, _ = fork_join()
    rows = [extract_execution_features(dag, _plan(dag, schedule=s, dpus=1)) for s in SCHEDULES]
    assert rows[0]["totals"] == rows[1]["totals"]
    assert rows[0]["totals"]["wave_critical_real_mac_sum"] == rows[0]["totals"]["real_mac_count"]


@pytest.mark.parametrize("fuse", [False, True])
def test_multiple_original_waves_are_not_counted_as_one_cohort(fuse):
    dag = _single(_node("many_rows", 257, 2, 1))
    facts = extract_execution_features(dag, _plan(dag, dpus=1), fuse_complex=fuse)
    assert facts["totals"]["cohort_count"] == 1
    assert facts["totals"]["original_wave_count"] == 2
    assert facts["totals"]["launch_count"] == (2 if fuse else 8)
    assert facts["totals"]["real_mac_count"] == 2056


@pytest.mark.parametrize("kwargs", [{"fuse_complex": 1}, {"geometry_policy": "unknown"}])
def test_invalid_execution_policy_is_rejected(kwargs):
    dag, _ = fork_join()
    with pytest.raises((ValueError, TypeError)):
        extract_execution_features(dag, _plan(dag), **kwargs)


def test_forged_schedule_is_not_accepted_as_feature_input():
    dag, _ = fork_join()
    plan = _plan(dag)
    first = plan.stages[0]
    forged = replace(first.work_units[0], logical_dpu=2)
    invalid = replace(plan, stages=(replace(first, work_units=(forged, *first.work_units[1:])), *plan.stages[1:]))
    with pytest.raises(ValueError, match="recomputation"):
        extract_execution_features(dag, invalid)


@pytest.mark.parametrize("policy,local_bytes", [(POLICIES[0], 10304), (POLICIES[1], 4896)])
@pytest.mark.parametrize("fuse", [False, True])
def test_panel_tail_spans_and_k_panel_barriers(policy, local_bytes, fuse):
    node = _node("one", 3, 5, 65)
    dag = _single(node)
    facts = extract_execution_features(dag, _plan(dag, policy=policy, dpus=1), fuse_complex=fuse)
    assert facts["totals"]["real_mac_count"] == 3900
    # Two K panels per product, three wrapper barriers per physical launch.
    assert facts["totals"]["barrier_events"] == (19 if fuse else 28)
    assert facts["totals"]["local_traffic"]["mram_aligned_transfer_bytes_estimate"] == local_bytes


@pytest.mark.parametrize("policy,panel_bytes,outer_bytes", [(POLICIES[0], 480, 416), (POLICIES[1], 416, 320)])
def test_outer_tail_changes_local_traversal_not_macs_or_transport(policy, panel_bytes, outer_bytes):
    dag = _single(_node("outer", 3, 5, 1))
    plan = _plan(dag, policy=policy, dpus=1)
    panel = extract_execution_features(dag, plan, fuse_complex=True)
    outer = extract_execution_features(dag, plan, fuse_complex=True, geometry_policy="outer_k1_v1")
    for name in ("real_mac_count", "h2d_bytes", "d2h_bytes", "launch_count"):
        assert panel["totals"][name] == outer["totals"][name]
    assert panel["totals"]["local_traffic"]["mram_aligned_transfer_bytes_estimate"] == panel_bytes
    assert outer["totals"]["local_traffic"]["mram_aligned_transfer_bytes_estimate"] == outer_bytes
    assert outer["totals"]["barrier_events"] == 11


def _single(node):
    return ContractionDAG(
        tensors=tuple(TensorSpec(v.tensor_id, v.labels, v.shape, "dense") for v in (node.left, node.right)),
        nodes=(node,), output=TensorView(tensor_id=node.output.id, labels=node.output.labels, shape=node.output.shape))


def test_mixed_fusion_fallback_matches_real_control_sequence():
    original, _ = fork_join(k=1)
    shapes = (((256, 1), (1, 256), (256, 256)),
              ((256, 1), (1, 1), (256, 1)),
              ((256, 256), (256, 1), (256, 1)))
    nodes = tuple(replace(node, left=replace(node.left, shape=left),
                          right=replace(node.right, shape=right),
                          output=replace(node.output, shape=output))
                  for node, (left, right, output) in zip(original.nodes, shapes, strict=True))
    tensors = tuple(TensorSpec(view.tensor_id, view.labels, view.shape, "dense", dtype="complex128")
                    for node in nodes[:2] for view in (node.left, node.right))
    dag = ContractionDAG(tensors=tensors, nodes=nodes, output=replace(original.output, shape=(256, 1)))
    plan = _plan(dag)
    facts = extract_execution_features(dag, plan, fuse_complex=True, geometry_policy="outer_k1_v1")
    node_map = {node.node_id: node for node in dag.nodes}
    all_waves = []
    for stage in plan.stages:
        lowerings, operands = {}, {}
        for node_id in stage.node_ids:
            node = node_map[node_id]
            node_stage = replace(stage, node_ids=(node_id,),
                                 work_units=tuple(u for u in stage.work_units if u.node_id == node_id))
            lowering, left, right, _, _ = _prepare_complex_operation(
                node, np.ones(node.left.shape, dtype=np.complex64),
                np.ones(node.right.shape, dtype=np.complex64), node_stage, POLICIES[0])
            lowerings[node_id] = lowering
            operands[node_id] = (left, right)
        waves, _ = build_cohort_waves(stage, lowerings, operands, dpu_count=3, tasklets=8,
                                      numeric_mode=0, request_start=0, fuse=True,
                                      geometry_policy="outer_k1_v1")
        all_waves.extend(waves)
    actual = {"h2d_bytes": 0, "d2h_bytes": 0, "launch_count": len(all_waves),
              "idle_slot_launch_count": 0, "real_mac_count": 0, "wave_critical_real_mac_sum": 0}
    for wave in all_waves:
        works = []
        for tile in wave:
            c = tile.control
            actual["h2d_bytes"] += CONTROL.size + sum(size for _, size in c.planes[:4])
            actual["d2h_bytes"] += COMPLETION.size + sum(size for _, size in c.planes[4:])
            actual["idle_slot_launch_count"] += int(c.flags == IDLE)
            works.append(c.m * c.n * c.k * (4 if c.kernel in FOUR_PRODUCT_KERNELS else 1))
        actual["real_mac_count"] += sum(works)
        actual["wave_critical_real_mac_sum"] += max(works)
    assert actual["launch_count"] == 8
    assert actual["idle_slot_launch_count"] == 15
    assert actual["real_mac_count"] == 525312
    assert actual == {key: facts["totals"][key] for key in actual}
    assert set(facts["totals"]["local_traffic"]["algorithms"]) == {"outer_compute_v1", "panel_compute_v1"}
    for wave, described in zip(all_waves, facts["waves"], strict=True):
        for tile, slot in zip(wave, described["slots"], strict=True):
            c = tile.control
            assert slot["kernel"] == c.kernel
            assert (slot["m"], slot["n"], slot["k"]) == (c.m, c.n, c.k)
            assert slot["product_count"] == (0 if c.flags == IDLE else 4 if c.kernel in FOUR_PRODUCT_KERNELS else 1)


@pytest.mark.parametrize("numeric_policy", POLICIES)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("geometry", ["panel_only_v1", "outer_k1_v1"])
@pytest.mark.parametrize("sliced", [False, True])
def test_composed_sdk_execution_matches_planned_counts_and_cpu_policy(
    tmp_path, binaries, numeric_policy, schedule, fuse, geometry, sliced,
):
    dag, inputs = fork_join(k=3 if sliced else 1)
    if sliced:
        dag = slice_contraction(dag, node_id="a", labels=(1,))
    plan = _plan(dag, policy=numeric_policy, schedule=schedule)
    facts = extract_execution_features(dag, plan, fuse_complex=fuse, geometry_policy=geometry)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    host, dpu, init = binaries[8]
    resources = UpmemResources(session_root=str(tmp_path / "session"), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init,
                               request_transport="packed_wave_v1")
    with open_upmem_simulator(dag, plan, resources, fuse_complex=fuse, geometry_policy=geometry) as session:
        for _ in range(2):
            result = session.run_once(inputs)
            np.testing.assert_array_equal(result.output, expected.output)
            assert result.numeric_facts["raw_lane_records"] == expected.numeric_facts["raw_lane_records"]
            assert result.measurement.h2d_bytes == facts["totals"]["h2d_bytes"]
            assert result.measurement.d2h_bytes == facts["totals"]["d2h_bytes"]
            operations = result.backend_facts["operation_facts"]
            assert sum(op["cohort_native_launch_count"] for op in operations) == facts["totals"]["launch_count"]
            assert result.backend_facts["cpu_fallback_used"] is False
    assert facts["totals"]["host_reduce_count"] == int(sliced)
