"""Whole-DAG correctness through the real persistent host and SDK simulator."""

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from quantum_bench.cpu import replay_upmem_plan_once
from quantum_bench.cpu import run_complex128_reference
from quantum_bench.circuits import builtin_circuit
from quantum_bench.lowering import build_contraction_dag, lower_tensor_network, slice_contraction
from quantum_bench.model import make_simulation_job
from quantum_bench.planning import plan_opt_einsum
from quantum_bench.results import ExecutionFailed
from quantum_bench.upmem.native_session import V4ProtocolError
from quantum_bench.upmem.path_heuristic import extract_plan_features
from quantum_bench.model import ContractNode, ContractionDAG, TensorSpec, TensorView
from quantum_bench.upmem.plan import (
    UpmemResources, UpmemTopology, physical_plan_id, plan_upmem, validate_upmem_plan,
)
from quantum_bench.upmem.runtime import open_upmem_simulator


NATIVE = Path(__file__).resolve().parents[1] / "native/upmem/runtime"


@pytest.fixture(scope="module")
def binaries():
    missing = [tool for tool in ("make", "gcc", "dpu-pkg-config", "dpu-upmem-dpurte-clang")
               if not shutil.which(tool)]
    if missing:
        reason = f"SDK wave runtime prerequisites missing: {missing}"
        if os.environ.get("UPMEM_REQUIRE_SDK_SIMULATOR") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    result = {}
    for tasklets in (3, 7, 8, 12, 24):
        subprocess.run(["make", "-C", str(NATIVE), "v4", f"NR_TASKLETS={tasklets}",
                        f"bin/dpu_wave_v5_t{tasklets}"], check=True, capture_output=True)
        result[tasklets] = tuple(str(NATIVE / "bin" / name) for name in (
            f"host_upmem_execution_plan_v4_t{tasklets}", f"dpu_wave_v5_t{tasklets}",
            f"dpu_simplepim_management_init_t{tasklets}"))
    return result


def fork_join(k=3):
    tensors, nodes, inputs = [], [], {}
    for index, name in enumerate(("a", "b")):
        l_labels, r_labels, out_labels = ((0, 1), (1, 2), (0, 2)) if not index else ((2, 4), (4, 3), (2, 3))
        left = TensorSpec(name + "l", l_labels, (2, k), "dense", dtype="complex128")
        right = TensorSpec(name + "r", r_labels, (k, 2), "dense", dtype="complex128")
        out = TensorSpec(name + "o", out_labels, (2, 2), "dense", produced_by=name)
        tensors.extend((left, right))
        nodes.append(ContractNode(node_id=name,
                     left=TensorView(tensor_id=left.id, labels=left.labels, shape=left.shape),
                     right=TensorView(tensor_id=right.id, labels=right.labels, shape=right.shape),
                     output=out, contracted_labels=(l_labels[1],), output_labels=out_labels, dependencies=()))
        for tensor in (left, right):
            values = np.arange(np.prod(tensor.shape)).reshape(tensor.shape)
            inputs[tensor.id] = ((values + index) % 5 - 2 + 1j * ((values + 2 * index) % 3 - 1)).astype(np.complex128)
    # Join by a dependent contraction, not merely an independent launch fixture.
    join = ContractNode(node_id="join",
        left=TensorView(tensor_id="ao", labels=(0, 2), shape=(2, 2)),
        right=TensorView(tensor_id="bo", labels=(2, 3), shape=(2, 2)),
        output=TensorSpec("out", (0, 3), (2, 2), "dense", produced_by="join"),
        contracted_labels=(2,), output_labels=(0, 3), dependencies=("a", "b"))
    dag = ContractionDAG(tensors=tuple(tensors), nodes=(*nodes, join),
                         output=TensorView(tensor_id="out", labels=(0, 3), shape=(2, 2)))
    return dag, inputs


def test_schedule_is_a_distinct_recomputed_physical_plan():
    dag, _ = fork_join()
    kwargs = dict(numeric_policy="split_complex_float32_v1",
                  topology=UpmemTopology(dpu_count=3, tasklets_per_dpu=8, rank_count=1))
    serial = plan_upmem(dag, **kwargs)
    waves = plan_upmem(dag, **kwargs, schedule_policy="static_dag_waves_v1")
    assert serial.logical_plan_id == waves.logical_plan_id
    assert physical_plan_id(serial) != physical_plan_id(waves)
    assert waves.stages[0].node_ids == ("a", "b")
    validate_upmem_plan(dag, waves)
    with pytest.raises(ValueError, match="wave cost extraction is not qualified"):
        extract_plan_features(waves)
    with pytest.raises(ValueError, match="recomputation"):
        validate_upmem_plan(dag, replace(waves, schedule_policy="serial_nodes_v1"))
    first = waves.stages[0]
    bad = replace(first.work_units[0], logical_dpu=2)
    with pytest.raises(ValueError, match="recomputation"):
        validate_upmem_plan(dag, replace(waves, stages=(replace(first, work_units=(bad, *first.work_units[1:])), *waves.stages[1:])))


@pytest.mark.parametrize("dpus,tasklets", [(1, 8), (3, 8), (4, 8), (1, 3), (1, 7), (1, 12), (1, 24)])
@pytest.mark.parametrize("numeric_policy", ["split_complex_float32_v1", "complex_int8_shared_scale_v1"])
@pytest.mark.parametrize("fuse", [False, True])
def test_full_dag_wave_runtime_preserves_policy_and_dependency_publication(
    tmp_path, binaries, dpus, tasklets, numeric_policy, fuse
):
    dag, inputs = fork_join()
    topology = UpmemTopology(dpu_count=dpus, tasklets_per_dpu=tasklets, rank_count=1)
    plan = plan_upmem(dag, numeric_policy=numeric_policy, topology=topology,
                      schedule_policy="static_dag_waves_v1")
    expected = replay_upmem_plan_once(dag, plan, inputs)
    host, dpu, init = binaries[tasklets]
    resources = UpmemResources(session_root=str(tmp_path / "session"), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init,
                               request_transport="packed_wave_v1")
    with open_upmem_simulator(dag, plan, resources, fuse_complex=fuse) as session:
        samples = [session.run_once(inputs), session.run_once(inputs)]
    assert session.close()["native_identity_verified"]
    for sample in samples:
        np.testing.assert_array_equal(sample.output, expected.output)
        assert sample.numeric_facts["raw_lane_records"] == expected.numeric_facts["raw_lane_records"]
        facts = sample.backend_facts
        assert facts["physical_plan_id"] == physical_plan_id(plan)
        assert facts["schedule_policy"] == "static_dag_waves_v1"
        assert facts["request_transport"] == "packed_wave_v1"
        assert facts["kernel_implementation_id"] == "dpu_panel_dispatch_v5_v1"
        assert facts["cpu_fallback_used"] is False
        assert facts["timing_claim_applicable"] is False
        operations = facts["operation_facts"]
        assert sum(op["cohort_native_launch_count"] for op in operations) == (
            (3 if dpus == 1 else 2) * (1 if fuse else 4))
        assert sample.measurement.kernel_s == sum(op["timing"]["rank_response_kernel_max_sum_s"] for op in operations)
        assert sample.measurement.h2d_bytes == sum(op["application_visible_h2d_bytes"] for op in operations)
        for op in operations:
            assert op["transfer_attribution_scope"] == "cohort_idle_overhead_on_first_node_v1"
            assert op["application_visible_h2d_bytes"] >= op["cohort_idle_h2d_bytes"]
            assert op["application_visible_d2h_bytes"] >= op["cohort_idle_d2h_bytes"]
        if dpus > 1:
            assert operations[0]["cohort_idle_h2d_bytes"] > 0
            assert operations[1]["cohort_idle_h2d_bytes"] == 0
            assert operations[1]["cohort_idle_d2h_bytes"] == 0
            assert operations[0]["cohort_node_ids"] == ("a", "b")
            assert set(operations[0]["active_dpu_ids"]).isdisjoint(operations[1]["active_dpu_ids"])
            assert operations[2]["cohort_id"] != operations[0]["cohort_id"]
    assert not list((tmp_path / "session").rglob("wave-*.bin"))


def test_failed_cohort_preserves_context_and_never_submits_consumer(tmp_path, binaries, monkeypatch):
    dag, inputs = fork_join()
    plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1",
                      topology=UpmemTopology(dpu_count=3, tasklets_per_dpu=8, rank_count=1),
                      schedule_policy="static_dag_waves_v1")
    host, dpu, init = binaries[8]
    resources = UpmemResources(session_root=str(tmp_path / "failure"), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init,
                               request_transport="packed_wave_v1")
    calls = []

    def fail_submission(**kwargs):
        calls.append(kwargs["operations"])
        error = V4ProtocolError("completion_failed", "injected partial cohort failure")
        error.backend_facts = {"failed_operation_index": 1, "completed_wave_count": 0,
                               "completed_result_count": 1}
        raise error

    session = open_upmem_simulator(dag, plan, resources)
    try:
        monkeypatch.setattr(session._low_level.ranks[0].session, "submit_waves", fail_submission)
        with pytest.raises(ExecutionFailed) as failure:
            session.run_once(inputs)
        facts = failure.value.backend_facts
        assert failure.value.stage == "completion_failed"
        assert facts["plan_stage_id"] == plan.stages[0].stage_id
        assert facts["branch_stage_id"] == plan.stages[0].stage_id
        assert facts["branch_node_id"] == "b"
        assert facts["cohort_node_ids"] == ("a", "b")
        assert facts["completed_result_count"] == 1
        with pytest.raises(ExecutionFailed, match="closed or failed"):
            session.run_once(inputs)
    finally:
        with pytest.raises(ExecutionFailed, match="session_close"):
            session.close()
    assert len(calls) == 1
    assert len(calls[0]) == 2


@pytest.mark.parametrize("numeric_policy", ["split_complex_float32_v1", "complex_int8_shared_scale_v1"])
def test_wave_runtime_preserves_split_k_order_and_serial_control(tmp_path, binaries, numeric_policy):
    dag, inputs = fork_join(k=257)
    host, dpu, init = binaries[8]
    outputs = []
    for schedule in ("serial_nodes_v1", "static_dag_waves_v1"):
        plan = plan_upmem(dag, numeric_policy=numeric_policy,
                          topology=UpmemTopology(dpu_count=3, tasklets_per_dpu=8, rank_count=1),
                          schedule_policy=schedule)
        expected = replay_upmem_plan_once(dag, plan, inputs)
        resources = UpmemResources(session_root=str(tmp_path / schedule), host_binary=host,
                                   dpu_binary=dpu, initialization_binary=init,
                                   request_transport="packed_wave_v1")
        with open_upmem_simulator(dag, plan, resources, fuse_complex=True) as session:
            sample = session.run_once(inputs)
        np.testing.assert_array_equal(sample.output, expected.output)
        outputs.append(sample.output)
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("numeric_policy", ["split_complex_float32_v1", "complex_int8_shared_scale_v1"])
def test_sliced_wave_cohorts_finish_before_host_reduction_and_consumer(tmp_path, binaries, numeric_policy):
    dag, inputs = fork_join()
    dag = slice_contraction(dag, node_id="a", labels=(1,))
    plan = plan_upmem(dag, numeric_policy=numeric_policy,
                      topology=UpmemTopology(dpu_count=3, tasklets_per_dpu=8, rank_count=1),
                      schedule_policy="static_dag_waves_v1")
    assert any(stage.kind == "host_reduce" for stage in plan.stages)
    expected = replay_upmem_plan_once(dag, plan, inputs)
    host, dpu, init = binaries[8]
    resources = UpmemResources(session_root=str(tmp_path / "sliced"), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init,
                               request_transport="packed_wave_v1")
    with open_upmem_simulator(dag, plan, resources, fuse_complex=True) as session:
        sample = session.run_once(inputs)
    np.testing.assert_array_equal(sample.output, expected.output)
    assert sample.measurement.host_reduce_s is not None
    assert sample.numeric_facts["raw_lane_records"] == expected.numeric_facts["raw_lane_records"]


@pytest.mark.parametrize("name,params", [("bell_2q", {}), ("ghz_4q", {}),
                                         ("quantization_stress", {"n_qubits": 4})])
def test_quantum_full_statevector_uses_frozen_path_and_dag_waves(tmp_path, binaries, name, params):
    network, inputs = lower_tensor_network(make_simulation_job(builtin_circuit(name, params)))
    path, _ = plan_opt_einsum(network, optimize="greedy")
    dag = build_contraction_dag(network, path)
    plan = plan_upmem(dag, numeric_policy="split_complex_float32_v1",
                      topology=UpmemTopology(dpu_count=4, tasklets_per_dpu=8, rank_count=1),
                      schedule_policy="static_dag_waves_v1")
    expected = replay_upmem_plan_once(dag, plan, inputs)
    reference = run_complex128_reference(dag, inputs)
    host, dpu, init = binaries[8]
    resources = UpmemResources(session_root=str(tmp_path / name), host_binary=host,
                               dpu_binary=dpu, initialization_binary=init,
                               request_transport="packed_wave_v1")
    with open_upmem_simulator(dag, plan, resources, fuse_complex=True) as session:
        sample = session.run_once(inputs)
    np.testing.assert_array_equal(sample.output, expected.output)
    np.testing.assert_allclose(sample.output, reference, atol=2e-6, rtol=2e-6)
    assert sample.output.size == 2 ** (2 if name == "bell_2q" else 4)
    assert sample.backend_facts["logical_plan_id"] == plan.logical_plan_id
