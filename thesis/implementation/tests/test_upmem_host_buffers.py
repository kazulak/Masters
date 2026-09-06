"""Retained-buffer counts must follow the actual prepared-wave lifetimes."""

from math import prod

import numpy as np
import pytest

from quantum_bench.cpu import replay_upmem_plan_once
from quantum_bench.lowering import slice_contraction
from quantum_bench.model import ContractNode
from quantum_bench.upmem.execution_features import extract_execution_features
from quantum_bench.upmem.plan import UpmemResources
from quantum_bench.upmem.runtime import open_upmem_simulator
from tests.test_upmem_execution_features import POLICIES, SCHEDULES, _plan, _single
from tests.test_upmem_wave_runtime import binaries as _sdk_binaries, fork_join
from tests.test_upmem_wave_work import _node


binaries = _sdk_binaries


@pytest.mark.parametrize("policy,encoded_bytes", [(POLICIES[0], 128), (POLICIES[1], 32)])
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("fuse", [False, True])
def test_retained_buffers_include_idle_completion_snapshots(policy, encoded_bytes, schedule, fuse):
    dag, inputs = fork_join(k=1)
    facts = extract_execution_features(dag, _plan(dag, policy, schedule), fuse_complex=fuse)
    buffers = facts["host_buffers"]
    cohorts = 3 if schedule == SCHEDULES[0] else 2
    controls = cohorts * 3 * (1 if fuse else 4)
    assert buffers["caller_input_declared_bytes"] == sum(value.nbytes for value in inputs.values()) == 128
    assert buffers["graph_output_array_bytes"] == 3 * 4 * 8
    assert buffers["final_output_copy_bytes"] == 4 * 8
    assert buffers["encoded_operand_plane_bytes"] == encoded_bytes
    assert buffers["retained_response_snapshot_bytes"] == 3 * 4 * 4 * 4 + controls * 72
    assert buffers["retained_executor_bulk_bytes"] == (
        128 + encoded_bytes + buffers["retained_response_snapshot_bytes"])
    assert len(buffers["cohorts"]) == cohorts
    assert buffers["full_host_memory_bound"] is False
    assert buffers["peak_rss_measured"] is False
    assert buffers["retained_executor_bulk_bytes"] > buffers["max_response_snapshot_bytes"]


def test_odd_logical_results_do_not_include_dma_padding():
    dag = _single(_node("odd", 3, 5, 1))
    facts = extract_execution_features(dag, _plan(dag, dpus=1), fuse_complex=True)
    buffers = facts["host_buffers"]
    assert buffers["retained_response_snapshot_bytes"] == 72 + 4 * 3 * 5 * 4
    assert buffers["retained_response_snapshot_bytes"] < facts["totals"]["d2h_bytes"]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_slices_count_every_branch_and_host_reduction_output(schedule):
    original, _ = fork_join()
    dag = slice_contraction(original, node_id="a", labels=(1,))
    facts = extract_execution_features(dag, _plan(dag, schedule=schedule))
    buffers = facts["host_buffers"]
    assert facts["totals"]["host_reduce_count"] > 0
    assert buffers["graph_output_array_bytes"] == 8 * sum(prod(node.output.shape) for node in dag.nodes)
    assert buffers["retained_response_snapshot_bytes"] == sum(
        row["response_snapshot_bytes"] for row in buffers["cohorts"])
    assert buffers["final_output_copy_bytes"] == 32


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("sliced", [False, True])
def test_predicted_retained_arrays_and_snapshots_match_sdk_objects(
    tmp_path, binaries, monkeypatch, policy, schedule, fuse, sliced,
):
    dag, inputs = fork_join(k=3 if sliced else 1)
    if sliced:
        dag = slice_contraction(dag, node_id="a", labels=(1,))
    plan = _plan(dag, policy, schedule)
    facts = extract_execution_features(dag, plan, fuse_complex=fuse, geometry_policy="outer_k1_v1")
    buffers = facts["host_buffers"]
    host, dpu, init = binaries[8]
    resources = UpmemResources(session_root=str(tmp_path / "run"), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init, request_transport="packed_wave_v1")
    observed = {"snapshots": [], "envelopes": [], "outputs": 0, "encoded": 0}
    with open_upmem_simulator(dag, plan, resources, fuse_complex=fuse,
                              geometry_policy="outer_k1_v1") as session:
        native = session._low_level.ranks[0].session
        submit = native.submit_waves
        execute = session._low_level._execute_cohort

        def observe_submit(**kwargs):
            response = submit(**kwargs)
            backing = {id(payload.obj): payload.obj for wave in response["results"]
                       for products in wave for payload in products}
            assert len(backing) == 1
            assert all(type(value) is bytes for value in backing.values())
            observed["snapshots"].append(sum(len(value) for value in backing.values()))
            observed["envelopes"].append(response["envelope_bytes"])
            return response

        def observe_execute(*args, **kwargs):
            outcomes = execute(*args, **kwargs)
            for output, _, _, operands in outcomes.values():
                observed["outputs"] += output.nbytes
                observed["encoded"] += sum(operand.real.nbytes + operand.imag.nbytes for operand in operands)
            return outcomes

        monkeypatch.setattr(native, "submit_waves", observe_submit)
        monkeypatch.setattr(session._low_level, "_execute_cohort", observe_execute)
        result = session.run_once(inputs)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    np.testing.assert_array_equal(result.output, expected.output)
    assert result.numeric_facts["raw_lane_records"] == expected.numeric_facts["raw_lane_records"]
    assert observed["snapshots"] == [row["response_snapshot_bytes"] for row in buffers["cohorts"]]
    assert observed["envelopes"] == [row["input_envelope_bytes"] for row in buffers["cohorts"]]
    reduction_output_bytes = 8 * sum(prod(node.output.shape) for node in dag.nodes
                                     if not isinstance(node, ContractNode))
    assert observed["outputs"] + reduction_output_bytes == buffers["graph_output_array_bytes"]
    assert observed["encoded"] == buffers["encoded_operand_plane_bytes"]
    assert result.output.nbytes == buffers["final_output_copy_bytes"]


def test_split_k_retains_all_raw_partials_not_only_reconstructed_output():
    dag = _single(_node("split", 2, 3, 257))
    facts = extract_execution_features(dag, _plan(dag), fuse_complex=True)
    buffers = facts["host_buffers"]
    assert facts["totals"]["original_wave_count"] == 2
    assert buffers["retained_response_snapshot_bytes"] == 2 * 3 * 72 + 2 * 4 * 2 * 3 * 4
    assert buffers["graph_output_array_bytes"] == 2 * 3 * 8
